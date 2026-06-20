import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from PIL import Image
import imageio
from peft import PeftModel
import torchvision.transforms as T
import copy

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
from config import TRAINING_CONFIG

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# EXTRACTION LAYER (Matches training loop perfectly)
# =====================================================================
def get_custom_sketch_tokens(cache_path, device):
    assert os.path.exists(cache_path), f"Missing cache: {cache_path}"

    cond_dict = torch.load(cache_path, map_location=device)

    print("Loaded cond:", cond_dict["cond"].shape)
    print("Loaded neg_cond:", cond_dict["neg_cond"].shape)

    return {
        "cond": cond_dict["cond"].to(device),
        "neg_cond": cond_dict["neg_cond"].to(device)
    }

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

def export_assets(outputs, finename, output_dir):
    print(f"Rendering diagnostics for {finename}...")
    try:
        if 'gaussian' in outputs:
            gs_video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(output_dir, f"gs_{finename}.mp4"), gs_video, fps=30)
        if 'mesh' in outputs:
            mesh_video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(os.path.join(output_dir, f"mesh_{finename}.mp4"), mesh_video, fps=30)

        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95)
        glb.export(os.path.join(output_dir, f"mesh_{finename}.glb"))
        print(f"[SUCCESS] Exported {finename} asset bundle.")
    except Exception as e:
        print(f"[ERROR] Failed exporting asset frame bundles for {finename}: {e}")

# =====================================================================
# SYSTEM MAIN ENTRY
# =====================================================================
print("Loading base pipeline once...")

base_pipeline = TrellisImageTo3DPipeline.from_pretrained(TRAINING_CONFIG["model_backbone"])
base_pipeline.to(device)

assert base_pipeline is not None, "Pipeline failed to load (None returned)"
assert hasattr(base_pipeline, "models"), "Invalid pipeline object"


for m in base_pipeline.models.values():
    if hasattr(m, "eval"):
        m.eval()

base_state = {
    k: v.detach().cpu().clone()
    for k, v in base_pipeline.models["sparse_structure_flow_model"].state_dict().items()
}

def reset_model_state(pipe):
    """
    Restores the original backbone weights before each experiment.
    Prevents LoRA / mutation leakage between runs.
    """
    pipe.models["sparse_structure_flow_model"].load_state_dict(
        base_state,
        strict=True
    )

    # ensure eval mode stays consistent
    for m in pipe.models.values():
        if hasattr(m, "eval"):
            m.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Multi-Run TRELLIS Diagnostic")
    parser.add_argument("--sketch", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./diagnostic_outputs", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    sketch_image = Image.open(args.sketch).convert("RGB")
    filename = os.path.splitext(os.path.basename(args.sketch))[0]
    epoch = os.path.basename(args.checkpoint).split('_')[-1]

    # Pre-calculate expected output paths to check for existence
    run1_mesh_path = os.path.join(args.output_dir, f"mesh_pure_vanilla_{filename}.glb")
    run2_mesh_path = os.path.join(args.output_dir, f"mesh_vanilla_plus_custom_tokens_{filename}.glb")

    # -----------------------------------------------------------------
    # RUN 1: THE TRUE STOCK BASELINE TEST
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 1: LOADING PURE STOCK VANILLA PIPELINE")
    print("====================================================")
    # FIX: Load first, do NOT chain .to(device) on the from_pretrained call!
    pipe = base_pipeline
    
    if os.path.exists(run1_mesh_path):
        print(f"\n[⏭️ SKIP] Found existing RUN 1 output at: {run1_mesh_path}")
    else:
        print("\n--- Running Stock pipeline.run() ---")
        vanilla_outputs = pipe.run(sketch_image, seed=123)
        export_assets(vanilla_outputs, finename=f"pure_vanilla_{filename}", output_dir=args.output_dir)
    
    # Extract tokens from the fresh vanilla graph *before* we delete it
    print("\nExtracting custom conditioning profile from pristine DINOv2 weights...")
    uid = os.path.splitext(os.path.basename(args.sketch))[0]

    cache_path = os.path.join(
        TRAINING_CONFIG["data_root"],
        TRAINING_CONFIG["category"],
        "sketch_cache",
        f"{uid}.pt"
    )

    custom_cond = get_custom_sketch_tokens(cache_path, device)
    print("\n=== CONDITIONING SANITY CHECK ===")
    print("COND SHAPES:")
    print(custom_cond["cond"].shape)
    print(custom_cond["neg_cond"].shape)
    print(custom_cond["cond"].mean().item(), custom_cond["cond"].std().item())
    print(custom_cond["neg_cond"].mean().item(), custom_cond["neg_cond"].std().item())
    print("===============================\n")

    # -----------------------------------------------------------------
    # RUN 2: THE REAL SANITY CHECK (Vanilla Model + Your Custom Tokens)
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 2: STOCK VANILLA MODEL + CUSTOM SKETCH TOKENS")
    print("====================================================")
    if os.path.exists(run2_mesh_path):
        print(f"\n[⏭️ SKIP] Found existing RUN 2 output at: {run2_mesh_path}")
    else:
        # Using the same vanilla pipeline instance before we inject LoRA anywhere near memory
        print("Feeding custom conditioning dict directly into the clean, stock base model layers...")
        reset_model_state(pipe)
        torch.cuda.empty_cache()
        san_check_outputs = run_manual_inference_flow(pipe, custom_cond)
        export_assets(san_check_outputs, finename=f"vanilla_plus_custom_tokens_{filename}", output_dir=args.output_dir)
    
    # -----------------------------------------------------------------
    # RUN 3: THE FINETUNED SKETCH LORA TEST
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("RUN 3: INJECTING LORA INTO THE SYSTEM GRAPH")
    print("====================================================")
    print("Wrapping base model structures with PEFT configurations...")
    reset_model_state(pipe)
    torch.cuda.empty_cache()
    base_dit = pipe.models['sparse_structure_flow_model']
    lora_model = PeftModel.from_pretrained(base_dit, args.checkpoint)
    pipe.models['sparse_structure_flow_model'] = lora_model.merge_and_unload()
    
    # Keep everything pinned to eval mode
    for name, model in pipe.models.items():
        if hasattr(model, "eval"):
            model.eval()
    
    print("\n--- Running Generation on Fine-Tuned System Stack ---")
    lora_outputs = run_manual_inference_flow(pipe, custom_cond)
    export_assets(lora_outputs, finename=f"fine_tuned_slat_{filename}_e{epoch}", output_dir=args.output_dir)
    
    print(f"\n[DIAGNOSTICS COMPLETE] Compare results inside: {args.output_dir}")

#python sanity_check2.py --sketch "./dataloader/data/chair/sketches/8a4a3a90bc104f11b82cedd9b4e5ab6b_5.png" --checkpoint "./checkpoints/trellis_lora_epoch_10" --output_dir "./diagnostic_outputs"