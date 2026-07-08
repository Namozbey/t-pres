import os
import cv2
import numpy as np
import open3d as o3d
from rembg import remove, new_session
from PIL import Image

# ==========================================================
# Initialize the AI session once globally to force GPU usage
# ==========================================================
gpu_session = new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

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
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        pcd.transform([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]])
    return pcd


def extract_changes(sk1_path, sk2_path, rgb_path, depth_map, fx, fy, cx, cy, padding=5):
    """
    Compares two sketches to find the changed region, extracts that region 
    from both the 2D image and the 3D point cloud, saves the assets, 
    and returns the 3D Bounding Box of the changed part.
    """
    # ==========================================
    # 0. Load Main RGB Image & Setup Directories
    # ==========================================
    original_image = cv2.imread(rgb_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image at path: {rgb_path}")
        
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Ensure the output directory exists
    save_dir = "editing/output"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 1. Robustly Find Sketch Differences
    # ==========================================
    sk1 = cv2.imread(sk1_path, cv2.IMREAD_GRAYSCALE)
    sk2 = cv2.imread(sk2_path, cv2.IMREAD_GRAYSCALE)
    
    h, w = rgb_image.shape[:2]
    sk1 = cv2.resize(sk1, (w, h))
    sk2 = cv2.resize(sk2, (w, h))

    # Blur to forgive minor pixel shifts
    sk1_blur = cv2.GaussianBlur(sk1, (5, 5), 0)
    sk2_blur = cv2.GaussianBlur(sk2, (5, 5), 0)

    diff = cv2.absdiff(sk1_blur, sk2_blur)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Clean up thin lines with morphology
    kernel = np.ones((5, 5), np.uint8)
    diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    diff_clean = cv2.dilate(diff_clean, kernel, iterations=2)

    # Get bounding box coordinates
    y_coords, x_coords = np.where(diff_clean > 0)
    
    if len(y_coords) == 0:
        print("No significant differences found between the sketches!")
        return {"min": [], "max": []}

    # Apply padding securely within image bounds
    y_min = max(0, y_coords.min() - padding)
    y_max = min(h, y_coords.max() + padding)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(w, x_coords.max() + padding)

    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[y_min:y_max, x_min:x_max] = True

    # ==========================================
    # 2. Extract Base Object via AI
    # ==========================================
    img_pil = Image.fromarray(rgb_image)
    rgba_output = remove(img_pil, session=gpu_session)
    rgba_array = np.array(rgba_output)
    
    base_mask = rgba_array[:, :, 3] > 128

    # ==========================================
    # 3. Save the 2D Images
    # ==========================================
    rgba_output.save(os.path.join(save_dir, "bg_free_full.png"))

    changed_rgba = rgba_array.copy()
    changed_rgba[~bbox_mask, 3] = 0 
    Image.fromarray(changed_rgba).save(os.path.join(save_dir, "changed_part.png"))

    # ==========================================
    # 4. Generate & Save Point Clouds
    # ==========================================
    full_pcd = generate_pcd(rgb_image, depth_map, base_mask, fx, fy, cx, cy)
    o3d.io.write_point_cloud(os.path.join(save_dir, "bg_free_full_pc.ply"), full_pcd, write_ascii=False)

    combined_mask = base_mask & bbox_mask
    changed_pcd = generate_pcd(rgb_image, depth_map, combined_mask, fx, fy, cx, cy)
    o3d.io.write_point_cloud(os.path.join(save_dir, "changed_part_pc.ply"), changed_pcd, write_ascii=False)

    # ==========================================
    # 5. Calculate and Return 3D Bounding Box
    # ==========================================
    if len(changed_pcd.points) == 0:
        print("Warning: The bounding box contained no 3D points.")
        return {"min": [], "max": []}

    # Open3D bounds return as numpy arrays, .tolist() converts them for standard dict format
    bb_dict = {
        "min": changed_pcd.get_min_bound().tolist(),
        "max": changed_pcd.get_max_bound().tolist()
    }
    
    print(f"Pipeline complete. Files saved to '{save_dir}'.")
    return bb_dict

def reconstruct_with_rembg(rgb_image, depth_map, fx, fy, cx, cy):
    """
    Uses AI to perfectly mask the object, ignoring shadows, 
    saves the transparent 2D image, and reconstructs the oriented 3D point cloud.
    """
    img_pil = Image.fromarray(rgb_image)
    
    # Pass the pre-loaded GPU session to the remove function
    rgba_output = remove(img_pil, session=gpu_session) 
    
    rgba_output.save("filtered_image_debug.png")
    rgba_array = np.array(rgba_output)
    
    alpha_channel = rgba_array[:, :, 3]
    object_mask = alpha_channel > 128

    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    u_filtered = u[object_mask]
    v_filtered = v[object_mask]
    z_filtered = depth_map[object_mask]
    colors_filtered = rgb_image[object_mask] / 255.0

    x = (u_filtered - cx) * z_filtered / fx
    y = (v_filtered - cy) * z_filtered / fy
    z = z_filtered
    points = np.stack((x, y, z), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.transform([[1,  0,  0, 0],
                   [0, -1,  0, 0],
                   [0,  0, -1, 0],
                   [0,  0,  0, 1]])

    o3d.io.write_point_cloud("bg_free_pc.ply", pcd, write_ascii=False)

    return pcd

def save_pointcloud_diff(
    rgb: np.ndarray,            
    depth: np.ndarray,          
    diff_mask: np.ndarray,      
    fx: float, fy: float, cx: float, cy: float,
    out_path: str = "cloud_dense_S.ply",
    depth_min: float = 0.1,
    depth_max: float = 50.0,
    stride: int = 4,
    apply_open3d_flip: bool = True,
    z_margin: float = 0.05  # Use this to shave off the background wall if it bleeds in!
) -> None:
    
    # --- STEP 1: Strict 2D Isolation (Find only the 'S') ---
    if len(diff_mask.shape) > 2:
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
        
    diff_mask = cv2.resize(diff_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # High threshold to kill faint noise/letters
    _, thresh = cv2.threshold(diff_mask, 0, 255, cv2.THRESH_BINARY)

    # Find the largest clump of differences (The 'S')
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No differences found. Aborting.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(thresh)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Setup the strided grid
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    z = depth[vv, uu].astype(np.float32)

    # --- STEP 2: Calculate 3DBB strictly from the 'S' mask ---
    inside_s = clean_mask[vv, uu] > 0
    valid_s = (z > depth_min) & (z < depth_max) & np.isfinite(z) & inside_s

    x_s = (uu[valid_s] - cx) * z[valid_s] / fx
    y_s = (vv[valid_s] - cy) * z[valid_s] / fy
    z_s = z[valid_s]

    if len(x_s) == 0:
        print("No valid depth points inside the mask. Aborting.")
        return

    # Define the 3DBB, applying the margin to the back (max_z)
    min_x, max_x = x_s.min(), x_s.max()
    min_y, max_y = y_s.min(), y_s.max()
    min_z = z_s.min()
    max_z = z_s.max() - z_margin  

    print(f"Computed Strict 3DBB:")
    print(f"X: {min_x:.3f} -> {max_x:.3f}")
    print(f"Y: {min_y:.3f} -> {max_y:.3f}")
    print(f"Z: {min_z:.3f} -> {max_z:.3f} (Margin: {z_margin})")

    # --- STEP 3: Reconstruct the FULL scene and apply 3DBB ---
    valid_full = (z > depth_min) & (z < depth_max) & np.isfinite(z)

    x_full = (uu - cx) * z / fx
    y_full = (vv - cy) * z / fy

    in_3dbb = (
        (x_full >= min_x) & (x_full <= max_x) &
        (y_full >= min_y) & (y_full <= max_y) &
        (z >= min_z) & (z <= max_z)
    )

    final_mask = valid_full & in_3dbb

    # Extract final points and colors
    points = np.stack((x_full, y_full, z), axis=-1)[final_mask]
    
    rgb_resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    colors = (rgb_resized[vv, uu][final_mask].astype(np.float32) / 255.0)

    if len(points) == 0:
        print("No points passed the 3DBB filter. Aborting.")
        return

    # --- STEP 4: Save to Open3D ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if apply_open3d_flip:
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    print(f"Successfully isolated the dense volume! Saved {len(points):,} points to {out_path}")


def get_3dbb(
    depth: np.ndarray,          
    diff_mask: np.ndarray,      
    fx: float, fy: float, cx: float, cy: float,
    depth_min: float = 0.1,
    depth_max: float = 50.0,
    stride: int = 4
) -> dict:
    
    # 1. Clean and Isolate the Mask (Threshold at 30)
    if len(diff_mask.shape) > 2:
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
        
    diff_mask = cv2.resize(diff_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, thresh = cv2.threshold(diff_mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No differences found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(thresh)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 2. Setup the grid
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    z = depth[vv, uu].astype(np.float32)

    # 3. Calculate Global Bounds (The Whole Scene)
    valid_depth = (z > depth_min) & (z < depth_max) & np.isfinite(z)
    
    x_full = (uu[valid_depth] - cx) * z[valid_depth] / fx
    y_full = (vv[valid_depth] - cy) * z[valid_depth] / fy
    z_full = z[valid_depth]
    
    if len(x_full) == 0:
        return None
        
    global_min_x, global_max_x = x_full.min(), x_full.max()
    global_min_y, global_max_y = y_full.min(), y_full.max()
    global_min_z, global_max_z = z_full.min(), z_full.max()

    # 4. Calculate Local Bounds (Strictly the 'S')
    inside_s = clean_mask[vv, uu] > 0
    valid_s = valid_depth & inside_s

    x_s = (uu[valid_s] - cx) * z[valid_s] / fx
    y_s = (vv[valid_s] - cy) * z[valid_s] / fy
    z_s = z[valid_s]

    if len(x_s) == 0:
        return None

    min_x, max_x = x_s.min(), x_s.max()
    min_y, max_y = y_s.min(), y_s.max()
    min_z, max_z = z_s.min(), z_s.max()
    
    # 5. Normalize against the global scene [0 to 1]
    # (Adding 1e-6 prevents a division-by-zero crash if a dimension is perfectly flat)
    norm_min_x = (min_x - global_min_x) / (global_max_x - global_min_x + 1e-6)
    norm_max_x = (max_x - global_min_x) / (global_max_x - global_min_x + 1e-6)
    
    norm_min_y = (min_y - global_min_y) / (global_max_y - global_min_y + 1e-6)
    norm_max_y = (max_y - global_min_y) / (global_max_y - global_min_y + 1e-6)
    
    norm_min_z = (min_z - global_min_z) / (global_max_z - global_min_z + 1e-6)
    norm_max_z = (max_z - global_min_z) / (global_max_z - global_min_z + 1e-6)

    # Convert np.float32 back to standard Python floats for clean dictionary output
    return {
        "min": [float(norm_min_x), float(norm_min_y), float(norm_min_z)],
        "max": [float(norm_max_x), float(norm_max_y), float(norm_max_z)]
    }

def save_pointcloud(
    rgb: np.ndarray,            # (Hc, Wc, 3)
    depth: np.ndarray,          # (Hd, Wd)
    fx: float, fy: float, cx: float, cy: float,
    out_path: str = "cloud.ply",
    depth_min: float = 0.1,
    depth_max: float = 50.0,
    stride: int = 4,
    resize_depth_to_rgb: bool = True,
    apply_open3d_flip: bool = True,
) -> None:
    Hc, Wc = rgb.shape[:2]
    Hd, Wd = depth.shape[:2]

    if resize_depth_to_rgb and (Hd != Hc or Wd != Wc):
        # Nearest keeps depth edges sharper; linear is smoother. Try nearest first.
        depth = cv2.resize(depth, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

    # Now proceed (memory-safe subsample)
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)

    z = depth[vv, uu].astype(np.float32)
    valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)[valid]
    colors = (rgb[vv, uu][valid].astype(np.float32) / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if apply_open3d_flip:
      pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    print(f"Saved {len(points):,} points to {out_path}")
