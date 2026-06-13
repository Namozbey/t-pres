#dataset.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SketchMeshDataset(Dataset):
    def __init__(self, root_dir, category="chair", image_size=518, transform=None):
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
            filename = os.path.basename(img_path)
            sketch_path = os.path.join(self.sketch_dir, filename)
            
            # Use the UID to find the target latent .npz file
            uid = filename.split('_')[0]
            mesh_path = os.path.join(self.mesh_dir, f"{uid}.glb")
            latent_path = os.path.join(self.latent_dir, f"{uid}.npz")
            ss_latent_path = os.path.join(self.ss_latent_dir, f"{uid}.npz")
            
            if os.path.exists(sketch_path) and os.path.exists(latent_path) and os.path.exists(ss_latent_path):
                valid_pairs.append({
                    "uid": uid,
                    "view_id": filename.split('_')[1].split('.')[0], 
                    "image_path": img_path,
                    "sketch_path": sketch_path,
                    "mesh_path": mesh_path,
                    "latent_path": latent_path,
                    "ss_latent_path": ss_latent_path,
                })
            else:
                print(f"Warning: Missing sketch or latent for {filename}. Skipping.")
                
        print(f"Loaded {len(valid_pairs)} valid data pairs for category '{self.category}'.")
        return valid_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        
        # 1. Load Images
        img = Image.open(item["image_path"]).convert('RGB')
        sketch = Image.open(item["sketch_path"]).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
            sketch_tensor = self.transform(sketch)
            
        # 2. Load Latents
        latent_data = np.load(item["latent_path"])
        ss_latent_data = np.load(item["ss_latent_path"])
        
        # Convert numpy arrays to torch tensors
        # feats are float32, coords are uint8 (need to be int for sparse tensors)
        feats = torch.from_numpy(latent_data['feats']).float()
        coords = torch.from_numpy(latent_data['coords']).int()
        ss_latent = torch.from_numpy(ss_latent_data['mean']).float()

        cond_path = os.path.join(self.cond_cache_dir, f"{item["uid"]}.pt")
        cond_data = torch.load(cond_path, map_location="cpu")

        cond_tokens = {
            "cond": cond_data["cond"].float(),
            "neg_cond": cond_data["neg_cond"].float()
        }
            
        return {
            ## Tensors
            "image": img_tensor,
            "sketch": sketch_tensor,
            "latent_feats": feats,
            "latent_coords": coords,
            "ss_latent": ss_latent,
            "cond_tokens": cond_tokens,
            
            ## Strings
            "uid": item["uid"],
            "view_id": item["view_id"],
            "mesh_path": item["mesh_path"],
            "image_path": item["image_path"],
            "sketch_path": item["sketch_path"]
        }