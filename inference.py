import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import torch
from PIL import Image
import imageio
from peft import PeftModel

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
from config import TRAINING_CONFIG

def load_sketch_pipeline(base_model_path: str, lora_checkpoint_path: str, device: str):
    """
    Loads TRELLIS and permanently bakes your sketch-trained LoRA weights 
    directly into the structural engine.
    """
    print(f"Loading Master Pipeline: {base_model_path}...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(base_model_path)
    pipeline.to(device)
    
    print(f"Merging Sketch-LoRA Adapters from {lora_checkpoint_path}...")
    base_dit = pipeline.models['sparse_structure_flow_model']
    
    # Wrap with PEFT and load your checkpointed matrices
    lora_model = PeftModel.from_pretrained(base_dit, lora_checkpoint_path)
    
    # Permanently fuse the weights back into the pipeline architecture
    pipeline.models['sparse_structure_flow_model'] = lora_model.merge_and_unload()
    
    # Lock the pipeline down for evaluation
    for model in pipeline.models.values():
        if hasattr(model, "eval"): model.eval()
        for p in model.parameters(): p.requires_grad = False
        
    print("LoRA successfully integrated into pipeline.")
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean TRELLIS Inference")
    parser.add_argument("--sketch", type=str, required=True, help="Path to input sketch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load pipeline with your custom merged weights
    pipeline = load_sketch_pipeline(
        base_model_path=TRAINING_CONFIG["model_backbone"],
        lora_checkpoint_path=args.checkpoint,
        device=device
    )
    
    # 2. Open raw user sketch (Let TRELLIS handle internal sizing/transformations)
    sketch_image = Image.open(args.sketch).convert("RGB")
    
    print("\n--- Running Generation via Native pipeline.run() ---")
    # 3. Direct execution exactly like the example files!
    outputs = pipeline.run(
        sketch_image,
        seed=42,
        # You can tune these samplers directly right here if needed!
        sparse_structure_sampler_params={
            "steps": 25,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3.0,
        }
    )
    
    # 4. Utilize the built-in Post-Processing & Rendering Suite
    print("\nProcessing outputs and rendering validation videos...")
    
    # Save standard diagnostic preview videos
    if 'gaussian' in outputs:
        gs_video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(args.output_dir, "sample_gs.mp4"), gs_video, fps=30)
        
    if 'mesh' in outputs:
        mesh_video = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(args.output_dir, "sample_mesh.mp4"), mesh_video, fps=30)

    # 5. Compile and export high-fidelity GLB
    print("Compiling finalized asset using postprocessing_utils...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    
    output_path = os.path.join(args.output_dir, "generated_mesh.glb")
    glb.export(output_path)
    print(f"[SUCCESS] Asset cleanly written to: {output_path}")