import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
import numpy as np
from PIL import Image
import trimesh
import imageio

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
from config import TRAINING_CONFIG

def get_custom_sketch_tokens(pipeline, sketch_image):
    """Extracts the exact conditioning dict required for Phase 2."""
    processed_image = pipeline.preprocess_image(sketch_image)
    cond_encoder = pipeline.models['image_cond_model']
    cond_encoder.eval()
    
    # Disable dropout for maximum determinism
    for module in cond_encoder.modules():
        if hasattr(module, 'drop_path'):   module.drop_path = 0.0
        if hasattr(module, 'dropout'):     module.dropout = 0.0

    print("Extracting conditioning tokens from sketch...")
    with torch.no_grad():
        final_cond = pipeline.get_cond([processed_image])
    return final_cond

def load_and_decode_ss_latent(pipeline, npz_path, device):
    """Loads the continuous GT ss_latent and decodes it into physical coordinates."""
    print(f"Loading continuous GT ss_latent from: {npz_path}")
    data = np.load(npz_path)
    
    print(f"   -> Available keys in npz: {data.files}")
    
    # 🛑 FIX: Explicitly target the 'mean' key just like your dataloader
    if 'mean' in data.files:
        key = 'mean'
    else:
        # Fallback just in case
        possible_keys = [k for k in data.files if 'latent' in k or 'data' in k or 'arr_' in k or 'ss_' in k]
        key = possible_keys[0] if possible_keys else data.files[0]
        
    print(f"   -> Using key: '{key}'")
    
    # Load and cast to float exactly like dataloader
    ss_latent = torch.from_numpy(data[key]).float().to(device)
    
    # Ensure batched format [1, C, ...]
    if ss_latent.dim() == 4:
        ss_latent = ss_latent.unsqueeze(0)
        
    print(f"Continuous Latent Shape: {ss_latent.shape}")
    
    # Pass the continuous latent through the small structure decoder to get binary coordinates
    print("Decoding continuous latent into binary spatial coordinates (Voxels)...")
    with torch.no_grad():
        logits = pipeline.models['sparse_structure_decoder'](ss_latent)
        
        # 🛑 FIX: The decoder outputs a dense 3D grid of logits: [B, C, X, Y, Z]
        # We binarize it (> 0 for logits is > 50% probability) to get the solid blocks
        occupancy = logits > 0
        
        # Get the multidimensional indices of all True (solid) blocks
        indices = torch.argwhere(occupancy)
        
        # Extract [batch_idx, x, y, z], ignoring the channel index (col 1)
        coords = indices[:, [0, 2, 3, 4]].to(torch.int32)
        
    print(f"-> Extracted {coords.shape[0]} physical voxels from the latent.")
    return coords

def export_raw_voxels(coords, output_path):
    """Converts the sparse coordinates directly into a 3D blocky Minecraft-style mesh."""
    print("Building raw Voxel representation...")
    # Strip the batch index (col 0), keep x, y, z
    points = coords[:, 1:].cpu().numpy()
    
    boxes = []
    # Create a 1x1x1 physical cube for every coordinate
    for p in points:
        box = trimesh.creation.box(extents=(1, 1, 1))
        box.apply_translation(p)
        boxes.append(box)
        
    # Combine all cubes into a single mesh
    if boxes:
        voxel_mesh = trimesh.util.concatenate(boxes)
        voxel_mesh.export(output_path)
        print(f"[SUCCESS] Exported raw scaffolding voxels to: {output_path}")
    else:
        print("[WARNING] Zero voxels found. Cannot export scaffolding.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode GT ss_latent directly to Voxels and Mesh")
    parser.add_argument("--ss_latent", type=str, required=True, help="Path to the GT ss_latent .npz file")
    parser.add_argument("--sketch", type=str, required=True, help="Path to the original sketch image")
    parser.add_argument("--output_dir", type=str, default="./diagnostic_outputs", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    sketch_image = Image.open(args.sketch).convert("RGB")
    
    print("\n====================================================")
    print("LOADING BASE TRELLIS PIPELINE")
    print("====================================================")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRAINING_CONFIG["model_backbone"])
    pipeline.to(device)
    
    for name, model in pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()

    # 1. Image Conditioning
    cond_dict = get_custom_sketch_tokens(pipeline, sketch_image)
    
    # 2. Extract Phase 1 Coordinates
    gt_coords = load_and_decode_ss_latent(pipeline, args.ss_latent, device)
    
    # 3. VISUALIZATION 1: Export the Raw Scaffolding
    base_name = os.path.splitext(os.path.basename(args.ss_latent))[0]
    voxel_path = os.path.join(args.output_dir, f"gt_{base_name}_RAW_VOXELS.glb")
    export_raw_voxels(gt_coords, voxel_path)
    
    # 4. VISUALIZATION 2: Pass into Phase 2 (SLAT Flow Matching)
    print("\n====================================================")
    print("PASSING GT COORDS INTO PHASE 2 (SLAT FLOW MODEL)")
    print("====================================================")
    with torch.no_grad():
        print("Sampling high-frequency SLAT features onto the GT coordinates...")
        slat = pipeline.sample_slat(cond=cond_dict, coords=gt_coords)
        
        print("Decoding final phase 2 features to mesh and gaussians...")
        outputs = pipeline.decode_slat(slat, formats=['mesh', 'gaussian'])
        
    print("Rendering Phase 2 completion...")
    try:
        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95)
        phase2_path = os.path.join(args.output_dir, f"gt_{base_name}_PHASE2_COMPLETION.glb")
        glb.export(phase2_path)
        print(f"[SUCCESS] Exported Phase 2 completion mesh to: {phase2_path}")
    except Exception as e:
        print(f"[ERROR] Failed exporting Phase 2 mesh: {e}")