import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import glob
import argparse
from PIL import Image
from editing.flux import Flux2Wrapper, Flux2DevRequest
from config import TRAINING_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Standalone FLUX Mediator Script")
    parser.add_argument("--data_root", type=str, default=TRAINING_CONFIG["data_root"], help="Root data directory")
    parser.add_argument("--category", type=str, default=TRAINING_CONFIG["category"], help="Category name (e.g., chair)")
    args = parser.parse_args()

    sketches_dir = os.path.join(args.data_root, args.category, "sketches")
    output_dir = os.path.join(args.data_root, args.category, "flux_images")
    
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
    
    # PROMPT = "convert it to real colorful image without changing the structure, flat clean white background, no background effects, no shadow effects, no lighting effect, preserving the exact geometric structure and contours of the sketch."
    PROMPT = """
    Convert this hand-drawn sketch into a realistic 3D object.
    
    The input sketch defines ONLY the silhouette and viewpoint.
    Preserve the exact outline, proportions, composition, camera angle, and perspective.
    
    Infer realistic three-dimensional geometry from the sketch.
    Give every object natural thickness, rounded surfaces, smooth transitions, and physically plausible volume.
    Interpret the drawing as a fully modeled 3D object rather than black outlines.
    
    Apply high-quality PBR materials with realistic texture, subtle surface imperfections, and natural specular reflections.
    Use professional product-render lighting to emphasize the 3D form while keeping the lighting soft and even.
    
    The final image should look like a high-end CGI render or CAD visualization rather than a line drawing.
    
    Background: pure white (#FFFFFF).
    No cast shadows.
    No floor.
    No ambient occlusion on the background.
    Object centered and isolated.
    
    Do not change the object's silhouette, viewpoint, proportions, or composition.
    Only infer realistic geometry, materials, depth, and surface details from the sketch.
    """

    flux = Flux2Wrapper()
    flux.load()

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

            flux_request = Flux2DevRequest()
            flux_request.prompt = PROMPT
            flux_request.image = sketch_image
            flux_request.generator_seed = 123

            generated_images = flux.generate_images(request=flux_request)
            if generated_images:
                generated_images[0].save(final_image_path)
                print(f"   -> [SUCCESS] Saved to {final_image_path}")
            
        except Exception as e:
            print(f"   -> [ERROR] Failed to process {uid}: {e}")

    print(f"\n[COMPLETE] All Flux images saved to: {output_dir}")

if __name__ == "__main__":
    main()