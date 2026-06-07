"""
TRELLIS SKETCH-TO-3D INFERENCE PIPELINE
=======================================
DESCRIPTION:
This is your production generation script. It takes a raw path to any user-provided 
sketch image via the terminal, normalizes it, and feeds it through your fine-tuned 
LoRA-adapted flow model. 

It breaks generation into the precise sub-steps expected by TRELLIS:
Phase 1: Generates sparse geometric layout coordinates (with point-density guards).
Phase 2: Samples continuous structured latents into those coordinates.
Phase 3: Synthesizes those sparse latents into explicit surface meshes.

OUTPUT:
Generates a single, high-fidelity 'generated_mesh.glb' in your target directory.

HOW TO RUN:
python inference.py \
  --sketch ./dataloader/data/chair/sketches/8a4a3a90bc104f11b82cedd9b4e5ab6b_0.png \
  --checkpoint ./checkpoints/trellis_lora_epoch_300 \
  --output_dir ./inference_outputs
"""

import os
import argparse
import torch
import random
from PIL import Image
import torchvision.transforms as T
from peft import PeftModel

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline

# =====================================================================
# INFERENCE EXECUTION PIPELINE
# =====================================================================

def load_inference_pipeline(base_model_path: str, lora_checkpoint_path: str, device: str):
    """
    Loads the master TRELLIS pipeline and hot-swaps the base structural DiT
    with your trained LoRA adapter.
    """
    print(f"Initializing Base TRELLIS Pipeline from {base_model_path}...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(base_model_path)
    pipeline.to(device)
    
    print(f"Injecting trained LoRA adapters from {lora_checkpoint_path}...")
    base_dit = pipeline.models['sparse_structure_flow_model']
    
    # This automatically finds your adapter config and merges/loads weights
    lora_dit = PeftModel.from_pretrained(base_dit, lora_checkpoint_path)
    
    # Merge weights permanently into the base model structure
    print("Merging weights into base model layers...")
    pipeline.models['sparse_structure_flow_model'] = lora_dit.merge_and_unload()
    
    # Ensure entire pipeline is locked down for evaluation
    for model_name, model in pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for p in model.parameters(): p.requires_grad_(False)
        
    print("Pipeline successfully customized for Sketch-to-3D inference.")
    return pipeline

def preprocess_sketch(image_path: str, image_size: int = 518) -> torch.Tensor:
    """
    Preprocesses a raw sketch image exactly how the training loop expects it.
    """
    image = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)

def run_inference(pipeline, sketch_tensor, device, seed: int = 42):
    """
    Executes the generation steps based on the verified TRELLIS Pipeline workflow.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        
    print("\n--- Generating 3D Structure Latents via Custom LoRA Flow Matching ---")
    
    # 1. Extract and format conditioning tokens explicitly
    with torch.no_grad():
        cond_encoder = pipeline.models['image_cond_model']
        cond_output = cond_encoder(sketch_tensor.to(device).float())
        
        if isinstance(cond_output, dict):
            cond_tokens = cond_output.get("cond_tokens")
        else:
            cond_tokens = cond_output

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

        cond_dict = {
            'cond': cond_tokens,
            'neg_cond': torch.zeros_like(cond_tokens)
        }

    # 2. Execute generation pipeline phases sequentially using verified model attributes
    with torch.no_grad():
        # Phase A: Sample spatial coordinate topologies via your fine-tuned LoRA block
        print("Sampling sparse structural layouts (Phase 1)...")
        coords = pipeline.sample_sparse_structure(cond=cond_dict, num_samples=1)
        
        print(f" -> Active sparse coordinates generated: {coords.shape[0]}")
        
        # 🚨 HARDENED SAFETY GUARD: Prevent low-volume structural core dumps
        MIN_SAFE_POINTS = 500
        if coords.shape[0] < MIN_SAFE_POINTS:
            print(f"\n [POINT DENSITY WARNING]")
            print(f"Generated points ({coords.shape[0]}) are below the multi-head attention safety threshold ({MIN_SAFE_POINTS}).")
            print("Padding layout with localized structural voxels to prevent terminal floating-point exception...")
            
            if coords.shape[0] == 0:
                # Absolute fallback if completely empty
                coords = torch.tensor([[0, 8, 8, 8]], dtype=torch.int32, device=device)
            
            # Pad by duplicating existing points with minor random offsets to maintain shape localization
            padding_needed = MIN_SAFE_POINTS - coords.shape[0]
            pad_indices = torch.randint(0, coords.shape[0], (padding_needed,), device=device)
            padded_coords = coords[pad_indices].clone()
            
            # Apply tiny random coordinate translations within a 1-voxel neighborhood space 
            # while protecting the leading batch index column (index 0)
            offsets = torch.randint(-1, 2, (padding_needed, 3), dtype=torch.int32, device=device)
            padded_coords[:, 1:] = torch.clamp(padded_coords[:, 1:] + offsets, 0, 63)
            
            # Combine original coordinates and padded coordinates
            coords = torch.cat([coords, padded_coords], dim=0)
            print(f" -> Stabilized coordinate volume: {coords.shape[0]} points safely loaded.")
        
        # Phase B: Sample detailed continuous features into the discovered sparse layouts
        print("Sampling structured latents (Phase 2)...")
        slat = pipeline.sample_slat(cond=cond_dict, coords=coords)
        
        # Phase C: Synthesize structural sparse latents into explicit surface geometry
        print("Synthesizing structural representation into explicit mesh geometry...")
        decoded_outputs = pipeline.decode_slat(slat, formats=['mesh'])
        
    return decoded_outputs

# =====================================================================
# SCRIPT ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRELLIS Sketch-to-3D Inference Pipeline")
    parser.add_argument("--sketch", type=str, required=True, help="Path to input sketch image file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved LoRA directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save generated assets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare pipeline and load custom LoRA architectures
    pipeline = load_inference_pipeline(
        base_model_path="microsoft/TRELLIS-image-large",
        lora_checkpoint_path=args.checkpoint,
        device=device
    )

    # 2. Process our target user input sketch
    sketch_tensor = preprocess_sketch(args.sketch, image_size=518)

    # 3. Predict vector fields and extract geometry safely
    generated_assets = run_inference(pipeline, sketch_tensor, device, seed=args.seed)

    # 4. Save file outputs to disk by unpacking properties safely
    output_path = os.path.join(args.output_dir, "generated_mesh.glb")
    
    mesh_output = generated_assets.get("mesh", generated_assets)
    if isinstance(mesh_output, list):
        mesh_output = mesh_output[0]
        
    if hasattr(mesh_output, "to_trimesh"):
        mesh_output = mesh_output.to_trimesh()
    elif hasattr(mesh_output, "mesh"):
        mesh_output = mesh_output.mesh

    mesh_output.export(output_path)
    print(f"\n[SUCCESS] 3D Asset completely recovered and written to: {output_path}")