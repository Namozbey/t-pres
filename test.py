import os
# Enables this your env has 'xformers' instead of 'flash-attn'
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from PIL import Image
from torch.utils.data import DataLoader
from peft import PeftModel

# Import configurations and dataloader modules
from config import TRAINING_CONFIG
from dataloader.dataset import SketchMeshDataset

# Import the pre-loaded pipeline and execution logic from sanity_check2
from diagnosis.sanity_check import (
    base_pipeline,
    run_manual_inference_flow,
    export_assets,
    reset_model_state,
    device
)

def main():
    parser = argparse.ArgumentParser(description="Batch Testing Script for TRELLIS LoRA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for test meshes")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    epoch = os.path.basename(args.checkpoint).split('_')[-1]

    print("\n====================================================")
    print(f"INITIALIZING TEST SET DATALOADER ({args.split.upper()} SPLIT)")
    print("====================================================")
    
    dataset = SketchMeshDataset(
        root_dir=TRAINING_CONFIG["data_root"], 
        category=TRAINING_CONFIG["category"], 
        image_size=TRAINING_CONFIG["image_size"], 
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=TRAINING_CONFIG["batch_size"], 
        shuffle=False,       # Keep order deterministic for testing
        num_workers=TRAINING_CONFIG["num_workers"],
        collate_fn=dataset.sparse_collate_fn,
        drop_last=False      # Ensure we test every single image, don't drop remainders
    )
    
    print(f"Total test items: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")

    # -----------------------------------------------------------------
    # PASS 1: VANILLA & VANILLA + CUSTOM TOKENS
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("PASS 1: STOCK VANILLA & CUSTOM TOKENS BASELINE")
    print("====================================================")
    reset_model_state(base_pipeline)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n--- Processing Vanilla Batch [{batch_idx + 1}/{len(dataloader)}] ---")
            current_batch_size = batch['cond_tokens'].shape[0]
            
            for i in range(current_batch_size):
                uid = batch["uid"][i]
                sketch_path = batch["sketch_path"][i]
                print(f"  -> Generating baseline meshes for UID: {uid}")
                
                run1_mesh_path = os.path.join(args.output_dir, f"mesh_pure_vanilla_{uid}.glb")
                run2_mesh_path = os.path.join(args.output_dir, f"mesh_vanilla_plus_custom_tokens_{uid}.glb")
                
                # Extract the conditioning for THIS specific item and add the batch dimension [1, seq_len, dim]
                cond_dict = {
                    "cond": batch["cond_tokens"][i].unsqueeze(0).to(device),
                    "neg_cond": batch["neg_cond_tokens"][i].unsqueeze(0).to(device)
                }

                # RUN 1: Pure Vanilla
                if os.path.exists(run1_mesh_path):
                    print(f"     [⏭️ SKIP] Found existing RUN 1 output at: {run1_mesh_path}")
                else:
                    print("     [1/3] Running Pure Stock Vanilla Pipeline...")
                    sketch_image = Image.open(sketch_path).convert("RGB")
                    vanilla_outputs = base_pipeline.run(sketch_image, seed=123)
                    export_assets(vanilla_outputs, finename=f"pure_vanilla_{uid}", output_dir=args.output_dir)

                # RUN 2: Vanilla + Custom Tokens
                if os.path.exists(run2_mesh_path):
                    print(f"     [⏭️ SKIP] Found existing RUN 2 output at: {run2_mesh_path}")
                else:
                    print("     [2/3] Running Stock Vanilla Model + Custom Tokens...")
                    san_check_outputs = run_manual_inference_flow(base_pipeline, cond_dict)
                    export_assets(san_check_outputs, finename=f"vanilla_plus_custom_tokens_{uid}", output_dir=args.output_dir)

    # -----------------------------------------------------------------
    # PASS 2: LORA INJECTION & FINE-TUNED GENERATION
    # -----------------------------------------------------------------
    print("\n====================================================")
    print("PASS 2: INJECTING LORA INTO THE SYSTEM GRAPH")
    print("====================================================")
    reset_model_state(base_pipeline)
    torch.cuda.empty_cache()
    
    base_dit = base_pipeline.models['sparse_structure_flow_model']
    lora_model = PeftModel.from_pretrained(base_dit, args.checkpoint)
    base_pipeline.models['sparse_structure_flow_model'] = lora_model.merge_and_unload()
    
    # Pin all sub-models to evaluation mode
    for name, model in base_pipeline.models.items():
        if hasattr(model, "eval"):
            model.eval()

    print("\n====================================================")
    print("STARTING FINE-TUNED BATCH INFERENCE LOOP")
    print("====================================================")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n--- Processing LoRA Batch [{batch_idx + 1}/{len(dataloader)}] ---")
            current_batch_size = batch['cond_tokens'].shape[0]
            
            for i in range(current_batch_size):
                uid = batch["uid"][i]
                print(f"  -> Generating fine-tuned mesh for UID: {uid}")
                
                expected_mesh_path = os.path.join(args.output_dir, f"mesh_fine_tuned_slat_{uid}_e{epoch}.glb")
                
                if os.path.exists(expected_mesh_path):
                    print(f"     [⏭️ SKIP] Found existing RUN 3 output at: {expected_mesh_path}")
                    continue
                
                cond_dict = {
                    "cond": batch["cond_tokens"][i].unsqueeze(0).to(device),
                    "neg_cond": batch["neg_cond_tokens"][i].unsqueeze(0).to(device)
                }
                
                print(f"     [3/3] Running Generation on Fine-Tuned System Stack (Epoch {epoch})...")
                outputs = run_manual_inference_flow(base_pipeline, cond_dict)
                export_assets(outputs, finename=f"fine_tuned_slat_{uid}_e{epoch}", output_dir=args.output_dir)

    print(f"\n[TESTING COMPLETE] All assets exported to: {args.output_dir}")

if __name__ == "__main__":
    main()