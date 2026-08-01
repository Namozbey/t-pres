import os
import gc
import torch
import imageio
from PIL import Image

# Import your pipeline and utils
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import postprocessing_utils, render_utils
from editing.kv_cache import TrellisKVCacheManager

# Import your custom manager (assuming it's in a file named kv_cache_manager.py)
# from kv_cache_manager import TrellisKVCacheManager 

class TrellisEditor:
    def __init__(self, model_name="E:/Nov1/AI/TRELLIS/TRELLIS-image-large", device="cuda"):
        print("Initializing Trellis Pipeline...")
        self.device = device
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self.pipeline.to(self.device)
        
        print("Registering KV Cache Managers...")
        self.ss_manager = TrellisKVCacheManager(self.pipeline.models['sparse_structure_flow_model'])
        # self.slat_manager = TrellisKVCacheManager(self.pipeline.models['slat_flow_model'])
        
        self.ss_manager.register_hooks()
        # self.slat_manager.register_hooks()

    def _save_state(self, state_path, seed):
        """Saves the current KV caches and seed to a file."""
        state = {
            'ss_cache': self.ss_manager.kv_cache,
            # 'slat_cache': self.slat_manager.kv_cache,
            'seed': seed
        }
        torch.save(state, state_path)
        print(f"[State Saved] -> {state_path}")

    def _load_state(self, state_path):
        """Loads KV caches from a file into the managers."""
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Cannot load state. File not found: {state_path}")
            
        # weights_only=False is required to load python dictionaries containing tensors
        state = torch.load(state_path, weights_only=False) 
        
        self.ss_manager.kv_cache = state['ss_cache']
        # self.slat_manager.kv_cache = state['slat_cache']
        print(f"[State Loaded] <- {state_path}")
        
        return state['seed']

    def process(self, image_path, out_glb_path, state_out_path, state_in_path=None, mask_bb=None, seed=123):
        """
        Runs the Trellis pipeline. 
        If state_in_path is None: Runs a fresh generation.
        If state_in_path exists: Loads the state, applies the mask, and blends the new image.
        """
        # 1. Load the Image
        image = Image.open(image_path).convert("RGB")
        
        # 2. Reset counters for a fresh run
        self.ss_manager.reset_counters()
        # self.slat_manager.reset_counters()

        # 3. Configure Pipeline State
        if state_in_path is None:
            print("--- MODE: INITIAL GENERATION ---")
            self.ss_manager.use_cache = False
            # self.slat_manager.use_cache = False
            self.ss_manager.use_spatial_blend = False
            # self.slat_manager.use_spatial_blend = False
            
            current_seed = seed
            
        else:
            print("--- MODE: EDITING / BLENDING ---")
            if mask_bb is None:
                raise ValueError("A bounding box mask (mask_bb) is required when loading a previous state.")
                
            # Load the previous K,V state and force the seed to match the original
            current_seed = self._load_state(state_in_path)
            
            self.ss_manager.use_cache = True
            # self.slat_manager.use_cache = True
            self.ss_manager.use_spatial_blend = True
            # self.slat_manager.use_spatial_blend = True
            
            self.ss_manager.set_spatial_mask(mask_bb)
            # self.slat_manager.set_spatial_mask(mask_bb)

        # 4. Run Inference
        print(f"Running pipeline with seed {current_seed}...")
        outputs = self.pipeline.run(image, seed=current_seed)

        # 5. Export 3D Mesh & Video
        out_video_path = out_glb_path.split(".")[0] + ".mp4"
        mesh_video = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(out_video_path, mesh_video, fps=30)
        
        print(f"Exporting Mesh to {out_glb_path}...")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(out_glb_path)

        # 6. Save the resulting state for future chains
        self._save_state(state_out_path, current_seed)

        # 7. Cleanup VRAM
        torch.cuda.empty_cache()
        gc.collect()
        print("Done.\n")