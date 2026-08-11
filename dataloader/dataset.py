#dataset.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import json

class SketchMeshDataset(Dataset):
    def __init__(self, root_dir, category="chair", image_size=518, transform=None, split="train"):
        """
        Args:
            root_dir (str): Path to the base data directory (e.g., 'data')
            category (str): The object category (e.g., 'chair')
            image_size (int): Resolution to resize the images to
            transform (callable, optional): Custom torchvision transforms
        """
        self.root_dir = root_dir
        self.category = category
        self.base_path = os.path.join(root_dir, category)
        self.split = split

        split_path = os.path.join(self.base_path, "split.json")
        with open(split_path, "r") as f:
            self.split_info = json.load(f)
        
        self.img_dir = os.path.join(self.base_path, "images")
        self.sketch_dir = os.path.join(self.base_path, "sketches")
        self.mesh_dir = os.path.join(self.base_path, "meshes")
        self.latent_dir = os.path.join(self.base_path, "latents", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
        self.ss_latent_dir = os.path.join(self.base_path, "ss_latents", "ss_enc_conv3d_16l8_fp16")
        self.cond_cache_dir = os.path.join(self.base_path, "sketch_cache")
        
        # Base transforms: Convert PIL Image to PyTorch Tensor (scales 0-255 to 0.0-1.0)
        # and resizes to the expected model input size.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Gather all image files
        self.image_paths = glob.glob(os.path.join(self.img_dir, "*.png"))
        self.data_pairs = self._validate_and_pair_data()


    def _validate_and_pair_data(self):
        """Ensures every image has a matching sketch and SLAT latent."""
        valid_pairs = []
        
        for img_path in self.image_paths:

            filename = os.path.basename(img_path).split('.')[0]
            uid, view_id = filename.split('_')
            view_id = int(view_id)

            if uid not in self.split_info.keys() or view_id not in self.split_info[uid][self.split]:
                continue

            sketch_path = os.path.join(self.sketch_dir, f"{filename}.png")
            cond_path = os.path.join(self.cond_cache_dir, f"{filename}.pt")
            
            mesh_path = os.path.join(self.mesh_dir, f"{uid}.glb")
            latent_path = os.path.join(self.latent_dir, f"{uid}.npz")
            ss_latent_path = os.path.join(self.ss_latent_dir, f"{uid}.npz")
            
            if os.path.exists(sketch_path) and os.path.exists(cond_path) and os.path.exists(ss_latent_path):
                valid_pairs.append({
                    "uid": uid,
                    "view_id": filename.split('_')[1], 
                    "image_path": img_path,
                    "sketch_path": sketch_path,
                    "mesh_path": mesh_path,
                    "latent_path": latent_path,
                    "ss_latent_path": ss_latent_path,
                    "cond_path": cond_path,
                })
            else:
                print(f"Warning: Missing sketch or latent for {filename}. Skipping.")
                
        print(f"Loaded {len(valid_pairs)} valid data pairs for category '{self.category}'.")
        return valid_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        
        # Load Images
        img = Image.open(item["image_path"]).convert('RGB')
        sketch = Image.open(item["sketch_path"]).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
            sketch_tensor = self.transform(sketch)

        # Load Latents
        latent_data = np.load(item["latent_path"])
        ss_latent_data = np.load(item["ss_latent_path"])
        
        # Convert numpy arrays to torch tensors
        # feats are float32, coords are uint8 (need to be int for sparse tensors)
        feats = torch.from_numpy(latent_data['feats']).float()
        coords = torch.from_numpy(latent_data['coords']).int()
        ss_latent = torch.from_numpy(ss_latent_data['mean']).float()

        cond_data = torch.load(item["cond_path"], map_location="cpu")

        cond_tokens = cond_data["cond"].float().squeeze(0).squeeze(0)
        neg_cond_tokens = cond_data["neg_cond"].float().squeeze(0).squeeze(0)

        data = {
            ## Tensors
            "image": img_tensor,
            "sketch": sketch_tensor,
            "latent_feats": feats,
            "latent_coords": coords,
            "ss_latent": ss_latent,
            "cond_tokens": cond_tokens,
            "neg_cond_tokens": neg_cond_tokens,
            
            ## Strings
            "uid": item["uid"],
            "view_id": item["view_id"],
            "mesh_path": item["mesh_path"],
            "image_path": item["image_path"],
            "sketch_path": item["sketch_path"]
        }

        return data

    def sparse_collate_fn(self, batch):
        """
        Custom collate function to handle variable-length sparse 3D latents,
        while preserving all 2D images and string paths.
        """
        # 1. Stack the standard fixed-size tensors (Images and Sketches)
        batched_data = {
            "uid": [item["uid"] for item in batch],
            "view_id": [item["view_id"] for item in batch],
            "image_path": [item["image_path"] for item in batch],
            "sketch_path": [item["sketch_path"] for item in batch],
            "mesh_path": [item["mesh_path"] for item in batch],
            # Stack 2D tensors into [B, C, H, W]
            "image": torch.stack([item["image"] for item in batch]),
            "sketch": torch.stack([item["sketch"] for item in batch]),
        }

        batched_data.update({
            "ss_latent": torch.stack([item["ss_latent"] for item in batch]),
            "cond_tokens": torch.stack([item["cond_tokens"] for item in batch]),
            "neg_cond_tokens": torch.stack([item["neg_cond_tokens"] for item in batch])
        })
    
        batched_feats = []
        batched_coords = []
    
        # 2. Process the variable-length 3D sparse tensors
        for batch_idx, item in enumerate(batch):
            feats = item["latent_feats"]
            coords = item["latent_coords"]
            
            # Create a column of the current batch index: [N, 1]
            batch_idx_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
            
            # Append the batch index to the coordinates: [N, 4] -> (batch_idx, x, y, z)
            coords_with_batch = torch.cat([batch_idx_col, coords], dim=1)
            
            batched_feats.append(feats)
            batched_coords.append(coords_with_batch)
            
        # Concatenate all sparse points into single massive lists
        batched_data["latent_feats"] = torch.cat(batched_feats, dim=0)
        batched_data["latent_coords"] = torch.cat(batched_coords, dim=0)
        
        return batched_data