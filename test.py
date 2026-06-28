import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from torch.utils.data import DataLoader
from peft import PeftModel

# Import configurations and dataloader modules
from config import TRAINING_CONFIG
from dataloader.utils import sparse_collate_fn
from dataloader.dataset import SketchMeshDataset

# Import the pre-loaded pipeline and execution logic from sanity_check2
from sanity_check2 import (
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
        collate_fn=sparse_collate_fn,
        drop_last=False      # Ensure we test every single image, don't drop remainders
    )
    
    print(f"Total test items: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")

    print("\n====================================================")
    print("INJECTING LORA INTO THE SYSTEM GRAPH")
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
    print("STARTING BATCH INFERENCE LOOP")
    print("====================================================")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n--- Processing Batch [{batch_idx + 1}/{len(dataloader)}] ---")
            
            # The dataloader returns batches of N size. We iterate through them individually
            # so `run_manual_inference_flow` safely runs 1 sample at a time without OOMing.
            current_batch_size = batch['cond_tokens'].shape[0]
            
            for i in range(current_batch_size):
                uid = batch["uid"][i]
                print(f"  -> Generating mesh for UID: {uid}")
                
                # Pre-calculate the expected output path so we can skip it if it already exists
                expected_mesh_path = os.path.join(args.output_dir, f"mesh_test_{uid}_e{epoch}.glb")
                if os.path.exists(expected_mesh_path):
                    print(f"     [⏭️ SKIP] Found existing output at: {expected_mesh_path}")
                    continue
                
                # Extract the conditioning for THIS specific item and add the batch dimension [1, seq_len, dim]
                cond_dict = {
                    "cond": batch["cond_tokens"][i].unsqueeze(0).to(device),
                    "neg_cond": batch["neg_cond_tokens"][i].unsqueeze(0).to(device)
                }
                
                # Run the customized inference flow
                outputs = run_manual_inference_flow(base_pipeline, cond_dict)
                
                # Export the assets using the UID so you know exactly which sketch it came from
                export_assets(outputs, finename=f"test_{uid}_e{epoch}", output_dir=args.output_dir)

    print(f"\n[TESTING COMPLETE] All assets exported to: {args.output_dir}")

if __name__ == "__main__":
    main()