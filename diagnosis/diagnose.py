"""
TRELLIS OVERFITTING DIAGNOSTIC SCRIPT
=====================================
DESCRIPTION:
This script acts as a sandbox to test your model's ability to overfit a known 
sample from your dataset. It bypasses complex batching loops to load index 0 
directly from your hard drive data layers. It then sets up three distinct 3D 
mesh renders side-by-side to pinpoint exactly where your training pipeline 
or data formatting might be breaking down.

VISUAL OUTPUTS CREATED (in your output folder):
1. '1_ground_truth.glb'       - Decodes your raw saved dataset latent directly. 
                                 If this look broken, your offline preprocessing is flawed.
2. '2_base_model_pretrain.glb' - What stock TRELLIS outputs given your sketch condition.
3. '3_finetuned_checkpoint.glb'- What your LoRA-adapted model outputs. This should 
                                 ideally match the ground truth (Visual 1).

HOW TO RUN:
python diagnose.py --checkpoint ./checkpoints/trellis_lora_epoch_300
"""

import os
# Enables this your env has 'xformers' instead of 'flash-attn'
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from peft import PeftModel
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from dataloader.dataset import SketchMeshDataset
import trimesh
# Import the custom SparseTensor class that the model signature explicitly hunts for
from TRELLIS.trellis.modules.sparse.basic import SparseTensor
from config import TRAINING_CONFIG

def load_diagnostic_pipeline(base_model_path: str, lora_checkpoint_path: str, device: str):
    """
    Loads two pipelines: one completely vanilla (Base) and one hot-swapped with LoRA (Fine-tuned).
    """
    print("Loading Base TRELLIS Pipeline...")
    base_pipeline = TrellisImageTo3DPipeline.from_pretrained(base_model_path)
    base_pipeline.to(device)
    
    for model_name, model in base_pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()
    
    print("Loading Fine-Tuned TRELLIS Pipeline...")
    ft_pipeline = TrellisImageTo3DPipeline.from_pretrained(base_model_path)
    ft_pipeline.to(device)
    
    # Inject LoRA into the second pipeline instance
    base_dit = ft_pipeline.models['sparse_structure_flow_model']
    lora_dit = PeftModel.from_pretrained(base_dit, lora_checkpoint_path)
    ft_pipeline.models['sparse_structure_flow_model'] = lora_dit.merge_and_unload()
    
    for model_name, model in ft_pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()
    
    for model in base_pipeline.models.values(): 
        if hasattr(model, "parameters"):
            for p in model.parameters(): p.requires_grad_(False)
            
    for model in ft_pipeline.models.values(): 
        if hasattr(model, "parameters"):
            for p in model.parameters(): p.requires_grad_(False)
    
    return base_pipeline, ft_pipeline

def extract_conditioning(pipeline, sketch_tensor, device):
    """
    Replicates the sequence handling logic and packages it into the expected dictionary.
    """
    with torch.no_grad():
        cond_encoder = pipeline.models['image_cond_model']
        cond_output = cond_encoder(sketch_tensor.to(device))
        cond_tokens = cond_output.get("cond_tokens") if isinstance(cond_output, dict) else cond_output

        if cond_tokens.ndim == 2:
            cond_tokens = cond_tokens.unsqueeze(1)
            
        target_seq_len = 1024
        current_seq_len = cond_tokens.shape[1]
        
        if current_seq_len != target_seq_len:
            if current_seq_len == 1:
                cond_tokens = cond_tokens.repeat(1, target_seq_len, 1)
            else:
                cond_tokens = cond_tokens.permute(0, 2, 1)
                cond_tokens = torch.nn.functional.interpolate(
                    cond_tokens, size=target_seq_len, mode='linear', align_corners=False
                )
                cond_tokens = cond_tokens.permute(0, 2, 1)
                
    neg_cond = torch.zeros_like(cond_tokens)
    return {
        'cond': cond_tokens,
        'neg_cond': neg_cond
    }

def main():
    parser = argparse.ArgumentParser(description="TRELLIS Overfitting Diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./diagnosis/visuals")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load pipelines safely
    base_pipe, ft_pipe = load_diagnostic_pipeline(TRAINING_CONFIG["model_backbone"], args.checkpoint, device)
    
    # 2. Grab index 0 sample from your hard drive data layers
    print("\nExtracting a single sample from data layers...")
    dataset = SketchMeshDataset(root_dir=TRAINING_CONFIG["data_root"], category=TRAINING_CONFIG["category"], image_size=TRAINING_CONFIG["image_size"])
    
    batch = dataset[0]
    sketch_tensor = batch['sketch'].unsqueeze(0).to(device)
    gt_slat_dense = batch['ss_latent'].to(device) # Shape: [8, 16, 16, 16] or similar dense structure
    
    # -------------------------------------------------------------
    # VISUAL 1: Ground Truth (Dense to Sparse Conversion)
    # -------------------------------------------------------------
    print("\n[Visual 1/3] Converting and Decoding Ground Truth Latent...")
    with torch.no_grad():
        # Handle optional batch dimension from dataset sample safely
        if gt_slat_dense.ndim == 5:  # [B, C, X, Y, Z]
            gt_slat_dense = gt_slat_dense[0]
            
        # 🚨 FIX: Find coordinates where features are active to build the sparse network layout
        occupancy_mask = torch.sum(torch.abs(gt_slat_dense), dim=0) > 0.0
        coords = torch.argwhere(occupancy_mask).int() # Generates structural [N, 3] layout coordinates
        
        if coords.shape[0] == 0:
            print("WARNING: Ground truth spatial data appears empty! Using fallback mock coordinates.")
            coords = torch.argwhere(torch.ones_like(occupancy_mask)).int()

        # Add batch index 0 column to convert layout from [N, 3] to [N, 4] format for spconv compliance
        batch_indices = torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=device)
        coords_with_batch = torch.cat([batch_indices, coords], dim=1)
        
        # Isolate and harvest structural features corresponding directly to those discovered coordinates
        feats = gt_slat_dense[:, coords[:, 0], coords[:, 1], coords[:, 2]].permute(1, 0)
        
        # Wrap everything into a valid SparseTensor instance to appease the forward signature validation
        gt_sparse_tensor = SparseTensor(feats=feats, coords=coords_with_batch)
        
        # Run through the structural mesh decoder module safely
        decoder_model = base_pipe.models['slat_decoder_mesh']
        gt_mesh_result = decoder_model(gt_sparse_tensor)
        
        # Unpack the native MeshExtractResult wrapper properties
        gt_mesh_data = gt_mesh_result[0] if isinstance(gt_mesh_result, list) else gt_mesh_result
        v = gt_mesh_data.vertices.detach().cpu().numpy()
        f = gt_mesh_data.faces.detach().cpu().numpy()
        
        gt_mesh = trimesh.Trimesh(vertices=v, faces=f)
        gt_mesh.export(os.path.join(args.output_dir, "1_ground_truth.glb"))
        
    # Process sketch images into core pipeline input formats
    base_cond_dict = extract_conditioning(base_pipe, sketch_tensor, device)
    ft_cond_dict = extract_conditioning(ft_pipe, sketch_tensor, device)
    
    # -------------------------------------------------------------
    # VISUAL 2: Native Base Output (Before training)
    # -------------------------------------------------------------
    print("[Visual 2/3] Generating from Base Model...")
    with torch.no_grad():
        base_coords = base_pipe.sample_sparse_structure(cond=base_cond_dict, num_samples=1)
        base_slat = base_pipe.sample_slat(cond=base_cond_dict, coords=base_coords)
        base_decoded = base_pipe.decode_slat(base_slat, formats=['mesh'])
        
        base_mesh_data = base_decoded['mesh'][0] if isinstance(base_decoded['mesh'], list) else base_decoded['mesh']
        bv = base_mesh_data.vertices.detach().cpu().numpy()
        bf = base_mesh_data.faces.detach().cpu().numpy()
        
        base_mesh = trimesh.Trimesh(vertices=bv, faces=bf)
        base_mesh.export(os.path.join(args.output_dir, "2_base_model_pretrain.glb"))
        
    # -------------------------------------------------------------
    # VISUAL 3: Current Fine-Tuned Output (After training)
    # -------------------------------------------------------------
    print("[Visual 3/3] Generating from Fine-Tuned Checkpoint...")
    with torch.no_grad():
        ft_coords = ft_pipe.sample_sparse_structure(cond=ft_cond_dict, num_samples=1)
        ft_slat = ft_pipe.sample_slat(cond=ft_cond_dict, coords=ft_coords)
        ft_decoded = ft_pipe.decode_slat(ft_slat, formats=['mesh'])
        
        ft_mesh_data = ft_decoded['mesh'][0] if isinstance(ft_decoded['mesh'], list) else ft_decoded['mesh']
        fv = ft_mesh_data.vertices.detach().cpu().numpy()
        ff = ft_mesh_data.faces.detach().cpu().numpy()
        
        ft_mesh = trimesh.Trimesh(vertices=fv, faces=ff)
        ft_mesh.export(os.path.join(args.output_dir, "3_finetuned_checkpoint.glb"))
        
    print(f"\nDiagnostic execution complete! Open the files in '{args.output_dir}' using any 3D viewer.")

if __name__ == "__main__":
    main()