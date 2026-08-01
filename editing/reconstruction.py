import os
import cv2
import numpy as np
import open3d as o3d
# from rembg import remove, new_session
from PIL import Image
# import itertools
import copy

# ==========================================================
# Initialize the AI session once globally to force GPU usage
# ==========================================================
# gpu_session = new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def get_trellis_bb(trellis_mesh_path, transformed_bb_dict):
    """
    Takes a bounding box (already transformed into Trellis physical space) and 
    normalizes it relative to the Trellis generated mesh to map perfectly onto 
    the [0, 1] KV cache latent grid.
    """
    # 1. Load the Trellis generated mesh
    # (Using read_triangle_mesh since Trellis outputs a full surface, not just points)
    trellis_mesh = o3d.io.read_triangle_mesh(trellis_mesh_path)
    
    if len(trellis_mesh.vertices) == 0:
        raise ValueError(f"Could not load vertices from mesh at {trellis_mesh_path}")

    # 2. Get the absolute bounds of the Trellis Mesh
    mesh_min = np.array(trellis_mesh.get_min_bound())
    mesh_max = np.array(trellis_mesh.get_max_bound())
    
    # 3. Get the absolute bounds of your aligned BB
    part_min = np.array(transformed_bb_dict["min"])
    part_max = np.array(transformed_bb_dict["max"])
    
    # 4. Normalize the Part BB relative to the Trellis Mesh [0.0 to 1.0]
    range_xyz = mesh_max - mesh_min
    range_xyz[range_xyz == 0] = 1e-6  # Prevent division by zero
    
    norm_min = (part_min - mesh_min) / range_xyz
    norm_max = (part_max - mesh_min) / range_xyz

    # NOTE: deliberately NOT clipped to [0, 1]. Values outside that range mean
    # the edited region extends beyond the original mesh AABB (e.g. a new
    # object placed on top of a table). The voxel containment test handles
    # out-of-range bounds correctly; clipping here would delete exactly the
    # region where new geometry has to be born.

    # norm_min = np.clip(norm_min, 0.0, 1.0)
    # norm_max = np.clip(norm_max, 0.0, 1.0)
    
    # 5. Map physical space -> Trellis Latent Space [z, y, x]
    # Assuming Trellis physical is standard [X: Right, Y: Up, Z: Forward]
    # And Trellis latent is [0]: Z(Up), [1]: Y(Depth), [2]: X(Right)
    trellis_min = [
        norm_min[1],  # Latent Z (Up)
        1.0 - norm_min[2],  # Latent Y (Depth)
        norm_min[0]   # Latent X (Right)
    ]
    
    trellis_max = [
        norm_max[1],  
        1.0 - norm_max[2],  
        norm_max[0]   
    ]
    
    # 6. Safety check: ensure min is always less than max
    final_min = [float(min(a, b)) for a, b in zip(trellis_min, trellis_max)]
    final_max = [float(max(a, b)) for a, b in zip(trellis_min, trellis_max)]
    
    return {
        "min": final_min,
        "max": final_max
    }

def get_trellis_latent_mask(transformed_bb_dict, trellis_bounds=(-0.5, 0.5)):
    """
    Takes a bounding box (already transformed into Trellis physical space) and 
    normalizes it relative to the Trellis 1:1:1 global canonical workspace,
    accounting for the inverted Depth axis in Trellis.
    """
    global_min = trellis_bounds[0]
    global_max = trellis_bounds[1]
    global_range = global_max - global_min
    
    part_min = np.array(transformed_bb_dict["min"])
    part_max = np.array(transformed_bb_dict["max"])
    
    norm_min = (part_min - global_min) / global_range
    norm_max = (part_max - global_min) / global_range
    
    norm_min = np.clip(norm_min, 0.0, 1.0)
    norm_max = np.clip(norm_max, 0.0, 1.0)
    
    # 4. Map physical space -> Trellis Latent Space [z, y, x]
    # =======================================================
    # INVERT THE DEPTH: Trellis Y goes 0 (Front) -> 1 (Back)
    # Our Open3D Z goes 0 (Back) -> 1 (Front)
    # We apply `1.0 - value` to align them!
    
    trellis_min = [
        norm_min[1],            # Latent Z (Up)    <- Open3D Y
        1.0 - norm_min[2],      # Latent Y (Depth) <- INVERTED Open3D Z
        norm_min[0]             # Latent X (Right) <- Open3D X
    ]
    
    trellis_max = [
        norm_max[1],  
        1.0 - norm_max[2],      # Latent Y (Depth) <- INVERTED Open3D Z 
        norm_max[0]   
    ]
    
    # 5. Safety check (This automatically fixes the inverted min/max order)
    final_min = [float(min(a, b)) for a, b in zip(trellis_min, trellis_max)]
    final_max = [float(max(a, b)) for a, b in zip(trellis_min, trellis_max)]
    
    return {
        "min": final_min,
        "max": final_max
    }

def transform_bounding_box(changed_pcd, M, padding=0.02):
    """
    Transforms raw 3D points into Trellis space, calculates the tightest AABB, 
    and applies an optional padding to account for ICP alignment slop.
    
    Parameters:
    - padding: Absolute value to expand the bounding box in all directions 
               (e.g., 0.02 means adding 2% if the global space is 1.0)
    """
    if changed_pcd is None or len(changed_pcd.points) == 0:
        return {"min": [], "max": []}

    # 1. Deep copy the point cloud to protect the original data
    pcd_transformed = copy.deepcopy(changed_pcd)

    # 2. Apply the 4x4 Transformation Matrix to physically move the points
    pcd_transformed.transform(M)

    # 3. Get the absolute minimum and maximum XYZ bounds of the points
    new_min = pcd_transformed.get_min_bound()
    new_max = pcd_transformed.get_max_bound()

    # 4. Apply the padding (expand outward in all 6 directions)
    # Subtracting from min pushes the floor/left/back outward
    # Adding to max pushes the ceiling/right/front outward
    new_min -= padding
    new_max += padding

    return {
        "min": new_min.tolist(),
        "max": new_max.tolist()
    }

def visualize_bounding_box(ply_path, bb_dict, save_image_path=None):
    """
    Loads a point cloud and draws either an Oriented or Axis-Aligned bounding box.
    Automatically detects the box type based on the dictionary keys.
    """
    # 1. Load the Point Cloud
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Could not find point cloud at: {ply_path}")
        
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) == 0:
        print("Warning: The point cloud is empty!")
        return

    # 2. Parse the Bounding Box Dictionary
    bbox = None
    
    # CASE A: It's an Oriented Bounding Box (from extract_changes)
    if "corners" in bb_dict and len(bb_dict["corners"]) == 8:
        corners = np.array(bb_dict["corners"])
        
        # create_from_points finds the bounding geometry. Since we feed it exactly 
        # the 8 corners of a box, it perfectly reconstructs the original OBB.
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
        
        # Color it GREEN to signify it is a tight OBB
        bbox.color = (0, 1, 0) 
        print("Visualizing: Oriented Bounding Box (Green)")

    # CASE B: It's an Axis-Aligned Bounding Box (from transform_bounding_box)
    elif "min" in bb_dict and "max" in bb_dict and len(bb_dict["min"]) == 3:
        min_bound = np.array(bb_dict["min"])
        max_bound = np.array(bb_dict["max"])
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        # Color it RED to signify it is a Trellis-space AABB
        bbox.color = (1, 0, 0)
        print("Visualizing: Axis-Aligned Bounding Box (Red)")
        
    else:
        print("Warning: Invalid bb_dict format. Expected 'corners' or 'min'/'max'.")
        return

    # 3. Handle Rendering (Interactive vs File Save)
    if save_image_path:
        print(f"Rendering scene to {save_image_path}...")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False) 
        
        vis.add_geometry(pcd)
        vis.add_geometry(bbox)
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_image_path)
        vis.destroy_window()
        
        print("Visualization saved successfully.")
        
    else:
        print("Opening interactive viewer. Close the window to continue script execution.")
        o3d.visualization.draw_geometries([pcd, bbox])

def generate_pcd(rgb_image, depth_map, mask, fx, fy, cx, cy):
    """
    Helper function to project a specific 2D mask into an oriented 3D Point Cloud.
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Apply mask
    u_filtered = u[mask]
    v_filtered = v[mask]
    z_filtered = depth_map[mask]
    colors_filtered = rgb_image[mask] / 255.0

    # Back-project to 3D
    x = (u_filtered - cx) * z_filtered / fx
    y = (v_filtered - cy) * z_filtered / fy
    z = z_filtered
    points = np.stack((x, y, z), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    # Clean and orient
    if len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.0)
        pcd.transform([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]])
    return pcd

def process_2d_changes(sk1_path, sk2_path, rgb_image, save_dir, padding=3):
    """
    Compares two sketches to find the changed region, extracts the base object via AI,
    saves the 2D assets, and returns the exact and base masks.
    """
    if isinstance(rgb_image, str):
        rgb_image = cv2.imread(rgb_image)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
    h, w = rgb_image.shape[:2]
    
    # ==========================================
    # 1. Robustly Find Sketch Differences
    # ==========================================
    sk1 = cv2.imread(sk1_path, cv2.IMREAD_GRAYSCALE)
    sk2 = cv2.imread(sk2_path, cv2.IMREAD_GRAYSCALE)
    
    sk1 = cv2.resize(sk1, (w, h))
    sk2 = cv2.resize(sk2, (w, h))

    sk1_blur = cv2.GaussianBlur(sk1, (5, 5), 0)
    sk2_blur = cv2.GaussianBlur(sk2, (5, 5), 0)

    diff = cv2.absdiff(sk1_blur, sk2_blur)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    
    if np.sum(diff_clean) == 0:
        print("No significant differences found between the sketches!")
        return None, None

    # ==========================================
    # 2. Create Precise Filled Contour Mask
    # ==========================================
    bridge_radius = 20
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_radius * 2 + 1, bridge_radius * 2 + 1))
    dilated_edges = cv2.dilate(diff_clean, bridge_kernel)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_bloated_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_bloated_mask, contours, -1, 255, thickness=cv2.FILLED)

    exact_mask_uint8 = cv2.erode(filled_bloated_mask, bridge_kernel)

    if padding > 0:
        pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        exact_mask_uint8 = cv2.dilate(exact_mask_uint8, pad_kernel, iterations=1)

    exact_mask = exact_mask_uint8 > 0

    # ==========================================
    # 3. Extract Base Object via AI
    # ==========================================
    # img_pil = Image.fromarray(rgb_image)
    # # Note: Assumes `remove` and `gpu_session` are available in the global scope
    # rgba_output = remove(img_pil, session=gpu_session)
    # rgba_array = np.array(rgba_output)
    rgba_array = rgb_image
    
    base_mask = rgba_array[:, :, 3] > 128

    # ==========================================
    # 4. Save the 2D Images using Precise Mask
    # ==========================================
    # rgba_output.save(os.path.join(save_dir, "bg_free_full.png"))

    changed_rgba = rgba_array.copy()
    changed_rgba[~exact_mask, :] = 0  # Apply the precise blob mask!
    Image.fromarray(changed_rgba).save(os.path.join(save_dir, "changed_part.png"))

    return exact_mask, base_mask

def extract_changes(sk1_path, sk2_path, rgb_path, depth_map, fx, fy, cx, cy, padding=3):
    """
    Orchestrates the pipeline to find the precise contour of the changed region, 
    extracts that exact blob from both the 2D image and the 3D point cloud, 
    saves the assets, and returns the 3D Bounding Box of the changed part.
    """
    # ==========================================
    # 0. Load Main RGB Image & Setup Directories
    # ==========================================
    original_image = cv2.imread(rgb_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image at path: {rgb_path}")
        
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    save_dir = "editing/state"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 1. Process 2D Changes (Using Helper)
    # ==========================================
    exact_mask, base_mask = process_2d_changes(sk1_path, sk2_path, rgb_image, save_dir, padding)
    
    # Check if differences were actually found
    if exact_mask is None:
        return {"min": [], "max": []}, None

    # ==========================================
    # 2. Generate & Save Point Clouds
    # ==========================================
    full_pcd = generate_pcd(rgb_image, depth_map, base_mask, fx, fy, cx, cy)
    o3d.io.write_point_cloud(os.path.join(save_dir, "bg_free_full_pc.ply"), full_pcd, write_ascii=False)

    combined_mask = base_mask & exact_mask
    changed_pcd = generate_pcd(rgb_image, depth_map, combined_mask, fx, fy, cx, cy)
    o3d.io.write_point_cloud(os.path.join(save_dir, "changed_part_pc.ply"), changed_pcd, write_ascii=False)

    # ==========================================
    # 3. Calculate and Return Oriented Bounding Box
    # ==========================================
    if len(changed_pcd.points) == 0:
        print("Warning: The bounding box contained no 3D points.")
        return {"corners": []}, full_pcd

    # Get the tightest possible oriented box around the points
    obb = changed_pcd.get_oriented_bounding_box()
    
    # Extract the 8 vertices of this oriented box
    corners = np.asarray(obb.get_box_points())

    bb_dict = {
        "corners": corners.tolist()
    }
    
    print(f"Pipeline complete. Files saved to '{save_dir}'.")
    return bb_dict, changed_pcd