import os
import glob
import argparse
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

def main():
    parser = argparse.ArgumentParser(description="Standalone FLUX Mediator Script")
    parser.add_argument("--data_root", type=str, default="./dataloader/data", help="Root data directory")
    parser.add_argument("--category", type=str, default="chair", help="Category name (e.g., chair)")
    args = parser.parse_args()

    sketches_dir = os.path.join(args.data_root, args.category, "sketches")
    output_dir = os.path.join(args.data_root, args.category, "flux_image")
    
    # Ensure the output directory exists right next to the sketches folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG sketches in the directory
    sketch_files = glob.glob(os.path.join(sketches_dir, "*.png"))
    
    if not sketch_files:
        print(f"[ERROR] No sketches found in {sketches_dir}")
        return
        
    print(f"Found {len(sketch_files)} sketches to process.")

    print("\n====================================================")
    print("INITIALIZING FLUX PIPELINE")
    print("====================================================")
    
    MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    # PROMPT = "convert it to real colorful image without changing the structure, flat clean white background, no background effects, no shadow effects, no lighting effect, preserving the exact geometric structure and contours of the sketch."
    PROMPT = "convert it to real colorful image without changing the structure, no background effects, no shadow effects no lighting effect, keep the background clean and white."
    
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    print("\n====================================================")
    print("STARTING BATCH IMAGE GENERATION")
    print("====================================================")

    for i, sketch_path in enumerate(sketch_files):
        uid = os.path.splitext(os.path.basename(sketch_path))[0]
        final_image_path = os.path.join(output_dir, f"{uid}.png")
        
        print(f"[{i+1}/{len(sketch_files)}] Processing {uid}...")
        
        # Skip if already generated
        if os.path.exists(final_image_path):
            print(f"   -> [SKIP] Already exists: {final_image_path}")
            continue
            
        try:
            sketch_image = Image.open(sketch_path).convert("RGB")
            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(123)

            print("image:", sketch_image.size)
            print("prompt:", PROMPT)

            # Generate the new image
            mediated_result = pipe(
                prompt=PROMPT,
                image=[sketch_image],
                num_inference_steps=4,
                guidance_scale=4.0,
                num_images_per_prompt=1,
                generator=generator
            ).images[0]
            
            mediated_result.save(final_image_path)
            print(f"   -> [SUCCESS] Saved to {final_image_path}")
            
        except Exception as e:
            print(f"   -> [ERROR] Failed to process {uid}: {e}")

    print(f"\n[COMPLETE] All Flux images saved to: {output_dir}")

if __name__ == "__main__":
    main()