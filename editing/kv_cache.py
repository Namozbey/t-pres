import os
import torch
import torch.nn.functional as F

class TrellisKVCacheManager:
    def __init__(self, model):
        self.model = model
        self.kv_cache = {}
        self.use_cache = False
        self.use_spatial_blend = False
        self.external_grid_mask = None  # NEW: Holds the mask passed from the outside
        self.call_counters = {}
        self.hooks = []

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name.endswith('self_attn.to_qkv'):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._self_attn_hook(n, out)
                )
                self.hooks.append(hook)
            # elif name.endswith('cross_attn.to_kv'):
            #     hook = module.register_forward_hook(
            #         lambda m, inp, out, n=name: self._cross_attn_hook(n, out)
            #     )
            #     self.hooks.append(hook)

    def set_spatial_mask(self, bbox_dict):
        """
        Accepts a dictionary: {'min': [z, y, x], 'max': [z, y, x]}
        Values are normalized between 0.0 and 1.0.
        """
        self.external_grid_mask = bbox_dict

    def _resolve_spatial_mask(self, tensor, is_sparse, output_obj):
        if self.external_grid_mask is None:
            return torch.ones_like(tensor[..., :1])
            
        bbox = self.external_grid_mask
        z_min, y_min, x_min = bbox['min']
        z_max, y_max, x_max = bbox['max']

        if is_sparse:
            # 1. Get raw float coordinates [N, 3] -> (z, y, x)
            coords = output_obj.indices.float()[:, 1:4] 
            spatial_shape = torch.tensor(output_obj.spatial_shape, device=coords.device, dtype=coords.dtype)
            
            # 2. Normalize coordinates into a clean [0, 1] range
            norm_coords = coords / (spatial_shape - 1.0)
            
            # 3. Check which points fall INSIDE the bounding box limits
            mask_cond = (
                (norm_coords[:, 0] >= z_min) & (norm_coords[:, 0] <= z_max) &
                (norm_coords[:, 1] >= y_min) & (norm_coords[:, 1] <= y_max) &
                (norm_coords[:, 2] >= x_min) & (norm_coords[:, 2] <= x_max)
            )
            return mask_cond.to(tensor.dtype).unsqueeze(-1) # Shape: [N, 1]
            
        else:
            S = tensor.shape[-2] 
            G = int(round(S**(1/3))) 
            
            # 1. Recreate the 3D voxel grid indices for the sequence
            idx = torch.arange(S, device=tensor.device)
            z_idx = idx % G
            y_idx = (idx // G) % G
            x_idx = idx // (G * G)
            
            # Safe protection against single-voxel layer sizes
            denom = float(G - 1) if G > 1 else 1.0

            # 2. Normalize grid positions into a [0, 1] range
            y_norm = y_idx.float() / denom
            z_norm = z_idx.float() / denom
            x_norm = x_idx.float() / denom
            
            # 3. Apply the bounding box criteria to the flattened grid
            mask_cond = (
                (z_norm >= z_min) & (z_norm <= z_max) &
                (y_norm >= y_min) & (y_norm <= y_max) &
                (x_norm >= x_min) & (x_norm <= x_max)
            )
            
            W = mask_cond.to(tensor.dtype).unsqueeze(-1) # Shape: [S, 1]
            return W.expand(tensor.shape[:-1] + (1,))

    
    def _align_and_blend(self, new_tensor, cache_tensor, W):
        # Because we force coordinates to match, min_len truncation is no longer 
        # a destructive hack. The sequences will match perfectly.
        min_len = min(new_tensor.shape[-2], cache_tensor.shape[-2])
        t_new = new_tensor[..., :min_len, :]
        t_cache = cache_tensor[..., :min_len, :]
        w_align = W[..., :min_len, :]
            
        blended = w_align * t_new + (1 - w_align) * t_cache
        
        out = new_tensor.clone()
        out[..., :min_len, :] = blended
        return out

    def _self_attn_hook(self, name, output):
        step = self.call_counters.get(name, 0)
        self.call_counters[name] = step + 1
        cache_key = f"{name}_step_{step}"
        is_sparse = hasattr(output, 'feats')
        tensor = output.feats if is_sparse else output

        if not self.use_cache:
            q, k, v = tensor.chunk(3, dim=-1)
            self.kv_cache[cache_key] = (k.detach().to(torch.float16).cpu(), 
                                        v.detach().to(torch.float16).cpu())
            return output
        else:
            cached_item = self.kv_cache.get(cache_key)
            if not cached_item or not isinstance(cached_item, tuple):
                return output

            k_cached, v_cached = cached_item
            k_cached = k_cached.to(tensor.device).to(tensor.dtype)
            v_cached = v_cached.to(tensor.device).to(tensor.dtype)
            q, k_new, v_new = tensor.chunk(3, dim=-1)

            if self.use_spatial_blend:
                W = self._resolve_spatial_mask(tensor, is_sparse, output)
                k_final = self._align_and_blend(k_new, k_cached, W)
                v_final = self._align_and_blend(v_new, v_cached, W)
            else:
                k_final, v_final = k_cached, v_cached

            new_tensor = torch.cat([q, k_final, v_final], dim=-1)
            return output.replace(new_tensor) if is_sparse else new_tensor

    def _cross_attn_hook(self, name, output):
        step = self.call_counters.get(name, 0)
        self.call_counters[name] = step + 1
        cache_key = f"{name}_step_{step}"
        is_sparse = hasattr(output, 'feats')
        tensor = output.feats if is_sparse else output

        if not self.use_cache:
            self.kv_cache[cache_key] = tensor.detach().to(torch.float16).cpu()
            return output
        else:
            # 🚨 THE FIX: If we are blending, DO NOT touch cross attention!
            # Let the flow network see image_2's conditioning natively.
            if self.use_spatial_blend:
                return output
                
            # (Only do full replace if we are doing a strict Sanity Check)
            kv_cached = self.kv_cache.get(cache_key)
            if kv_cached is None or isinstance(kv_cached, tuple):
                return output
            
            kv_cached = kv_cached.to(tensor.device).to(tensor.dtype)
            return output.replace(kv_cached) if is_sparse else kv_cached

    def reset_counters(self):
        self.call_counters = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []