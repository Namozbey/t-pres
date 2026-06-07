import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from PIL import Image
import imageio
from peft import PeftModel
import torchvision.transforms as T

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
from config import TRAINING_CONFIG

# =====================================================================
# EXTRACTION LAYER (Matches training loop perfectly)
# =====================================================================
def get_custom_sketch_tokens(pipeline, sketch_image, device):
    """
    Leverages native TRELLIS pipeline wrappers to extract tokens,
    ensuring all hidden structural tokens (1374 length) are perfectly appended.
    """
    # 1. Use TRELLIS's native image preprocessor (handles padding/canvas scaling)
    processed_image = pipeline.preprocess_image(sketch_image)
    
    # Temporarily patch pipeline's image_cond_model to run eval behaviors
    cond_encoder = pipeline.models['image_cond_model']
    cond_encoder.eval()
    for module in cond_encoder.modules():
        if hasattr(module, 'drop_path'):   module.drop_path = 0.0
        if hasattr(module, 'dropout'):     module.dropout = 0.0

    print("   [Token Alignment] Passing preprocessed list through native get_cond context...")
    with torch.no_grad():
        # 2. Let native TRELLIS call encode_image and append its hidden structural tokens
        final_cond = pipeline.get_cond([processed_image])
        
    print(f"   [Token Alignment] Aligned sequence output shape: {final_cond['cond'].shape}")
    return final_cond

# =====================================================================
# EXECUTION LIFECYCLE
# =====================================================================
def run_manual_inference_flow(pipeline, conditioning_dict):
    """Executes TRELLIS sequentially using explicit conditioning injection with safety guards."""
    with torch.no_grad():
        # Phase 1: Sample spatial coordinate topologies
        coords = pipeline.sample_sparse_structure(cond=conditioning_dict, num_samples=1)
        
        # Prevent zero-volume structural crashes
        MIN_SAFE_POINTS = 500
        device = coords.device if hasattr(coords, 'device') else "cuda"
        
        if coords.numel() == 0 or coords.shape[0] < MIN_SAFE_POINTS:
            print(f"   [POINT DENSITY WARNING] Generated points ({coords.shape[0] if coords.numel() > 0 else 0}) are below safe threshold.")
            print("   Padding layout with fallback structural voxels to prevent terminal floating-point exceptions...")
            
            if coords.numel() == 0 or coords.shape[0] == 0:
                # If completely empty, seed a baseline bounding box structure so the shape calculation passes
                # Format: [batch_index, x, y, z] -> using a centered voxel distribution
                fallback_list = []
                for x in range(24, 40, 2):
                    for y in range(24, 40, 2):
                        for z in range(24, 40, 2):
                            fallback_list.append([0, x, y, z])
                coords = torch.tensor(fallback_list, dtype=torch.int32, device=device)
            else:
                # If it generated a few points, pad by duplicating them with minor random offsets
                padding_needed = MIN_SAFE_POINTS - coords.shape[0]
                pad_indices = torch.randint(0, coords.shape[0], (padding_needed,), device=device)
                padded_coords = coords[pad_indices].clone()
                
                # Apply tiny random spatial translations while keeping batch index (col 0) untouched
                offsets = torch.randint(-1, 2, (padding_needed, 3), dtype=torch.int32, device=device)
                padded_coords[:, 1:] = torch.clamp(padded_coords[:, 1:] + offsets, 0, 63)
                coords = torch.cat([coords, padded_coords], dim=0)

        print(f"   -> Sub-stage coordinates stabilized: {coords.shape[0]} spatial nodes.")
        
        # Phase 2: Sample detailed continuous features into the layouts
        slat = pipeline.sample_slat(cond=conditioning_dict, coords=coords)
        
        # Phase 3: Synthesize representations into explicit mesh/gaussians
        outputs = pipeline.decode_slat(slat, formats=['mesh', 'gaussian'])
        
    return outputs

def export_assets(outputs, prefix, output_dir):
    print(f"Rendering diagnostics for {prefix}...")
    try:
        if 'gaussian' in outputs:
            gs_video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(output_dir, f"{prefix}_sample_gs.mp4"), gs_video, fps=30)
        if 'mesh' in outputs:
            mesh_video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(os.path.join(output_dir, f"{prefix}_sample_mesh.mp4"), mesh_video, fps=30)

        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95)
        glb.export(os.path.join(output_dir, f"{prefix}_generated_mesh.glb"))
        print(f"[SUCCESS] Exported {prefix} asset bundle.")
    except Exception as e:
        print(f"[ERROR] Failed exporting asset frame bundles for {prefix}: {e}")

# =====================================================================
# SYSTEM MAIN ENTRY
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Multi-Run TRELLIS Diagnostic")
    parser.add_argument("--sketch", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./diagnostic_outputs", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    sketch_image = Image.open(args.sketch).convert("RGB")
    
    # -----------------------------------------------------------------
    # RUN 1: THE TRUE STOCK BASELINE TEST
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 1: LOADING PURE STOCK VANILLA PIPELINE")
    print("====================================================")
    # FIX: Load first, do NOT chain .to(device) on the from_pretrained call!
    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRAINING_CONFIG["model_backbone"])
    pipeline.to(device)
    
    # Safely set all sub-models inside the pipeline to eval mode
    for name, model in pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()
    
    print("\n--- Running Stock pipeline.run() ---")
    vanilla_outputs = pipeline.run(sketch_image, seed=42)
    export_assets(vanilla_outputs, prefix="1_pure_vanilla", output_dir=args.output_dir)
    
    # Extract tokens from the fresh vanilla graph *before* we delete it
    print("\nExtracting custom conditioning profile from pristine DINOv2 weights...")
    custom_cond = get_custom_sketch_tokens(
        pipeline=pipeline, 
        sketch_image=sketch_image, 
        device=device
    )
    
    # -----------------------------------------------------------------
    # RUN 2: THE REAL SANITY CHECK (Vanilla Model + Your Custom Tokens)
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 2: STOCK VANILLA MODEL + CUSTOM SKETCH TOKENS")
    print("====================================================")
    # Using the same vanilla pipeline instance before we inject LoRA anywhere near memory
    print("Feeding custom conditioning dict directly into the clean, stock base model layers...")
    san_check_outputs = run_manual_inference_flow(pipeline, custom_cond)
    export_assets(san_check_outputs, prefix="2_vanilla_plus_custom_tokens", output_dir=args.output_dir)
    
    # -----------------------------------------------------------------
    # RUN 3: THE FINETUNED SKETCH LORA TEST
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 3: INJECTING LORA INTO THE SYSTEM GRAPH")
    print("====================================================")
    print("Wrapping base model structures with PEFT configurations...")
    base_dit = pipeline.models['sparse_structure_flow_model']
    lora_model = PeftModel.from_pretrained(base_dit, args.checkpoint)
    pipeline.models['sparse_structure_flow_model'] = lora_model.merge_and_unload()
    
    # Keep everything pinned to eval mode
    for name, model in pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()
    
    print("\n--- Running Generation on Fine-Tuned System Stack ---")
    lora_outputs = run_manual_inference_flow(pipeline, custom_cond)
    export_assets(lora_outputs, prefix="3_fine_tuned_lora", output_dir=args.output_dir)
    
    print(f"\n[DIAGNOSTICS COMPLETE] Compare results inside: {args.output_dir}")