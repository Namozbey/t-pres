import os
# Enables this your env has 'xformers' instead of 'flash-attn'
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import gc
import cv2
import sys
import json
import torch
import argparse
from PIL import Image
import open3d as o3d
import numpy as np

# =====================================================================
# PYTORCH 2.4.0 / DIFFUSERS 0.39.0 COMPATIBILITY PATCH
# Automates the FlashAttention-3 bypass so users don't have to edit files.
# =====================================================================
if hasattr(torch, "library"):
    # 1. Patch custom_op
    if hasattr(torch.library, "custom_op"):
        _orig_custom_op = torch.library.custom_op
        def _safe_custom_op(*args, **kwargs):
            if args and isinstance(args[0], str) and "_diffusers_flash_attn_3" in args[0]:
                return lambda fn: fn
            return _orig_custom_op(*args, **kwargs)
        torch.library.custom_op = _safe_custom_op
        
    # 2. Patch register_fake
    if hasattr(torch.library, "register_fake"):
        _orig_register_fake = torch.library.register_fake
        def _safe_register_fake(*args, **kwargs):
            if args and isinstance(args[0], str) and "_diffusers_flash_attn_3" in args[0]:
                return lambda fn: fn
            return _orig_register_fake(*args, **kwargs)
        torch.library.register_fake = _safe_register_fake

# 3. Mock flex_attention (PyTorch 2.5 feature)
if "torch.nn.attention.flex_attention" not in sys.modules:
    mock_flex = MagicMock()
    sys.modules["torch.nn.attention.flex_attention"] = mock_flex
# =====================================================================

# Import your wrappers and tools
from editing.da3 import DA3
from editing.flux import Flux2Wrapper, Flux2DevRequest
from editing.trellis_editor import TrellisEditor
from editing.bbox.render import get_cam_to_mesh_matrix
from editing.bbox.warp import warp_difference, save_debug
from editing.bbox.view_search import find_best_angles
# from editing.mesh2pc import sample_mesh_surface
from editing.reconstruction import (
    transform_bounding_box, 
    get_trellis_bb,
    process_2d_changes,
    generate_pcd,
    # visualize_bounding_box
)

STATE_DIR = "state"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_META_FILE = os.path.join(STATE_DIR, "pipeline_state.json")

def load_meta_state():
    """Loads the pointers to the previous run's files."""
    if os.path.exists(STATE_META_FILE):
        with open(STATE_META_FILE, 'r') as f:
            return json.load(f)
    return None

def save_meta_state(sketch_path, img_path, mesh_path, kv_state_path, run_idx=1):
    state = {
        "prev_sketch": sketch_path,
        "prev_img": img_path,
        "prev_mesh": mesh_path,
        "kv_state": kv_state_path,
        "run_idx": run_idx
    }
    with open(STATE_META_FILE, 'w') as f:
        json.dump(state, f)

def run_pipeline(new_sketch_path, prompt, seed=123, not_from_cache=False):
    print("=========================================")
    print("      STARTING SKETCH-TO-MESH RUN        ")
    print("=========================================")
    
    meta_state = load_meta_state()
    
    # Update is_edit_mode to consider the not_from_cache flag
    is_edit_mode = (meta_state is not None) and not not_from_cache
    
    # Generate unique filenames for this run
    run_idx = meta_state.get("run_idx", 1) + 1 if is_edit_mode else 1
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
        _, _, fx, fy, cx, cy = da3.forward(prev_img)
        
        # UNLOAD DA3 TO FREE VRAM!
        da3.unload()

        process_2d_changes(prev_sketch, new_sketch_path, prev_img, save_dir=STATE_DIR)
        H, camera_pose, depth_map, center,scale = find_best_angles(prev_mesh, prev_img, fx, fy, cx, cy)

        M = get_cam_to_mesh_matrix(camera_pose, scale, center)

        rendered_img = cv2.imread(os.path.join(STATE_DIR, "render_from_mesh.png"))
        rendered_img = cv2.cvtColor(rendered_img,cv2.COLOR_BGR2RGB )

        diff_img = cv2.imread(os.path.join(STATE_DIR, "changed_part.png"))
        diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)


        aligned = warp_difference(
            diff_img,
            H,
            (rendered_img.shape[1], rendered_img.shape[0])
        )
        save_debug(aligned, os.path.join(STATE_DIR, "aligned_difference.png"))

        rgb_image = Image.open(os.path.join(STATE_DIR, "aligned_difference.png"))
        rgb_array = np.array(rgb_image)
        print(rgb_array.shape)

        base_mask = np.any(rgb_array > 0, axis=-1)

        changed_pcd = generate_pcd(rgb_array, depth_map, base_mask, fx, fy, cx, cy)
        o3d.io.write_point_cloud(os.path.join(STATE_DIR, "changed_part_pc.ply"), changed_pcd, write_ascii=False)

        if changed_pcd is None or len(changed_pcd.points) == 0:
            print("WARNING: No changes detected between sketches.")
        else:
            # 4. Transform to Trellis space
            print("Transforming physical bounds to Trellis Space...")
            transformed_bb = transform_bounding_box(changed_pcd, M, padding=0.02)

            # Uncomment if you wanna see the bounding-box for sanity-check
            # sample_mesh_surface(prev_mesh)
            # visualize_bounding_box("state/generated.ply", transformed_bb)
            
            # 5. Convert to normalized Latent Mask
            trellis_mask = get_trellis_bb(prev_mesh, transformed_bb)
            print(f"Computed KV Cache Mask: {trellis_mask}")
            with open(os.path.join(STATE_DIR, "bb.json"), "w") as fp:
                json.dump({
                    "transformed_bb": transformed_bb,
                    "trellis_mask": trellis_mask
                }, fp)

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
            seed=seed,
        )

    del editor
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # PHASE 4: Update State for Next Run
    # ==========================================
    # if is_edit_mode:
    save_meta_state(
        sketch_path=new_sketch_path,
        img_path=current_img_path,
        mesh_path=current_mesh_path,
        kv_state_path=current_kv_path,
        run_idx=run_idx
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

    default_prompt = """
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
    
    # Optional prompt
    parser.add_argument(
        "-p", "--prompt", 
        type=str, 
        default=default_prompt,
        help="Text prompt for the 2D generation model."
    )
    
    # Optional seed
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123,
        help="Random seed for generation to ensure consistency."
    )
    
    # New empty flag for clearing cache
    parser.add_argument(
        "-nc", "--not-from-cache",
        action="store_true",
        help="If provided, ignores the previous state and forces an initial generation."
    )

    args = parser.parse_args()

    # Verify the sketch file actually exists before starting the heavy models
    if not os.path.exists(args.sketch):
        print(f"Error: Could not find sketch file at '{args.sketch}'")
        exit(1)

    run_pipeline(
        new_sketch_path=args.sketch, 
        prompt=args.prompt, 
        seed=args.seed,
        not_from_cache=args.not_from_cache
    )