import os
# This single line fixes BOTH the dense and sparse attention modules!
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import gc
import json
import torch
import argparse  # <-- NEW: Added for terminal flags
from PIL import Image

# Import your wrappers and tools
from editing.da3 import DA3
from editing.flux import Flux2Wrapper, Flux2DevRequest
from editing.trellis_editor import TrellisEditor
from editing.reconstruction import (
    extract_changes, 
    transform_bounding_box, 
    get_trellis_latent_mask
)
from editing.test import run_alignment

STATE_DIR = "editing/state"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_META_FILE = os.path.join(STATE_DIR, "pipeline_state.json")

def load_meta_state():
    """Loads the pointers to the previous run's files."""
    if os.path.exists(STATE_META_FILE):
        with open(STATE_META_FILE, 'r') as f:
            return json.load(f)
    return None

def save_meta_state(sketch_path, img_path, mesh_path, kv_state_path):
    """Saves the current run's files so they become the 'previous' files next time."""
    state = {
        "prev_sketch": sketch_path,
        "prev_img": img_path,
        "prev_mesh": mesh_path,
        "kv_state": kv_state_path
    }
    with open(STATE_META_FILE, 'w') as f:
        json.dump(state, f)

def run_pipeline(new_sketch_path, prompt, seed=123):
    print("=========================================")
    print("      STARTING SKETCH-TO-MESH RUN        ")
    print("=========================================")
    
    meta_state = load_meta_state()
    is_edit_mode = meta_state is not None
    
    # Generate unique filenames for this run
    run_idx = 1 if not is_edit_mode else 2 
    current_img_path = os.path.join(STATE_DIR, f"gen_img_{run_idx}.png")
    current_mesh_path = os.path.join(STATE_DIR, f"mesh_{run_idx}.glb")
    current_kv_path = os.path.join(STATE_DIR, "current_kv_state.pt")

    # ==========================================
    # PHASE 1: Generate 2D Image (Flux)
    # ==========================================
    print("\n--- PHASE 1: Generating Image with Flux ---")
    flux = Flux2Wrapper()
    flux.load()
    
    # 1. Prepare the exact request object your wrapper expects
    flux_request = Flux2DevRequest()
    flux_request.prompt = prompt
    flux_request.image = Image.open(new_sketch_path).convert("RGB")
    flux_request.generator_seed = seed
    
    # 2. Generate the images (returns a list of PIL Images)
    generated_images = flux.generate_images(request=flux_request)

    # 3. Grab the first image from the list and save it to your path
    if generated_images:
        generated_images[0].save(current_img_path)
    
    # UNLOAD FLUX TO FREE VRAM!
    flux.unload()

    # ==========================================
    # PHASE 2: Spatial Math & Masking (DA3 & OpenCV)
    # ==========================================
    trellis_mask = None
    
    if is_edit_mode:
        print("\n--- PHASE 2: Calculating Spatial Edit Mask ---")
        prev_sketch = meta_state["prev_sketch"]
        prev_img = meta_state["prev_img"]
        prev_mesh = meta_state["prev_mesh"]
        
        # 1. Load DA3 to get depth of the PREVIOUS image
        da3 = DA3()
        da3.load()
        depth, focal_length, fx, fy, cx, cy = da3.forward(prev_img)
        
        # UNLOAD DA3 TO FREE VRAM!
        da3.unload()

        # 2. Extract 3D changes
        print("Extracting point cloud changes...")
        _, changed_pcd = extract_changes(
            prev_sketch, new_sketch_path, prev_img, 
            depth, fx, fy, cx, cy
        )
        
        if changed_pcd is None or len(changed_pcd.points) == 0:
            print("WARNING: No changes detected between sketches.")
        else:
            # 3. Get ICP Matrix
            print("Running ICP Alignment...")
            M = run_alignment(prev_mesh, "editing/output/bg_free_full_pc.ply")
            
            # 4. Transform to Trellis space
            print("Transforming physical bounds to Trellis Space...")
            transformed_bb = transform_bounding_box(changed_pcd, M, padding=0.01)
            
            # 5. Convert to normalized Latent Mask
            trellis_mask = get_trellis_latent_mask(transformed_bb)
            print(f"Computed KV Cache Mask: {trellis_mask}")

    # ==========================================
    # PHASE 3: Generate 3D Mesh (Trellis)
    # ==========================================
    print("\n--- PHASE 3: Generating 3D Mesh with Trellis ---")
    editor = TrellisEditor(device="cuda")
    
    if not is_edit_mode:
        print("Mode: INITIAL GENERATION")
        editor.process(
            image_path=current_img_path,
            out_glb_path=current_mesh_path,
            state_out_path=current_kv_path,
            state_in_path=None,
            mask_bb=None,
            seed=seed
        )
    else:
        print("Mode: EDIT BLENDING")
        editor.process(
            image_path=current_img_path,
            out_glb_path=current_mesh_path,
            state_out_path=current_kv_path,     
            state_in_path=meta_state["kv_state"], 
            mask_bb=trellis_mask,               
            seed=seed
        )

    del editor
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # PHASE 4: Update State for Next Run
    # ==========================================
    save_meta_state(
        sketch_path=new_sketch_path,
        img_path=current_img_path,
        mesh_path=current_mesh_path,
        kv_state_path=current_kv_path
    )
    
    print("\nPipeline execution complete! Ready for next iteration.")

# ==========================================
# CLI Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Sketch-to-Mesh 3D Editing Pipeline.")
    
    # Required sketch path
    parser.add_argument(
        "-s", "--sketch", 
        type=str, 
        required=True, 
        help="Path to the new sketch image file."
    )
    
    # Optional prompt (defaults to your cherry prompt, but easily changeable)
    parser.add_argument(
        "-p", "--prompt", 
        type=str, 
        default="turn it to real image, keeping background white", 
        help="Text prompt for the 2D generation model."
    )
    
    # Optional seed
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Random seed for generation to ensure consistency."
    )

    args = parser.parse_args()

    # Verify the sketch file actually exists before starting the heavy models
    if not os.path.exists(args.sketch):
        print(f"Error: Could not find sketch file at '{args.sketch}'")
        exit(1)

    run_pipeline(
        new_sketch_path=args.sketch, 
        prompt=args.prompt, 
        seed=args.seed
    )