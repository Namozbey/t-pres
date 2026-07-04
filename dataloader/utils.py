import os
import cv2
import glob
import torch
import shutil
import trimesh
import pyrender
import objaverse
import numpy as np
import trimesh.transformations as tf

def setup_dataset(download_limit=5, category="chair"):
    save_dir = os.path.join("dataloader/data", category, "meshes")

    print("Fetching Objaverse LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    
    # Get chair UIDs and slice the limit
    chair_uids = lvis_annotations.get(category, [])
    test_uids = chair_uids[:download_limit]
    
    print(f"Downloading {download_limit} test meshes to cache...")
    # This will be instant since you already downloaded them!
    objects = objaverse.load_objects(uids=test_uids, download_processes=4)
    
    # Create the local directory right next to your .ipynb file
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nCopying files to local directory: ./{save_dir}/")
    
    local_paths = {}
    for uid, cached_path in objects.items():
        # Create a clean filename: uid.glb
        local_path = os.path.join(save_dir, f"{uid}.glb")
        
        # Copy the file from the root cache to your local folder
        shutil.copy(cached_path, local_path)
        local_paths[uid] = local_path
        
        print(f"Saved: {local_path}")
        
    return local_paths

def generate_sketch_pairs(glb_path, num_views=3):
    # Force delete the EGL variable if it exists in memory
    if "PYOPENGL_PLATFORM" in os.environ:
        del os.environ["PYOPENGL_PLATFORM"]
        print("Successfully deleted EGL ghost variable!")
    else:
        print("No ghost variable found.")

    print(f"Loading and normalizing: {os.path.basename(glb_path)}")
    
    # 1. Load the mesh (force='mesh' collapses complex scenes into one single object)
    scene_mesh = trimesh.load(glb_path, force='mesh')
    
    # 2. Normalize geometry: Center it at (0,0,0) and scale it to fit a unit box
    vertices = scene_mesh.vertices
    center = vertices.mean(axis=0)
    scene_mesh.vertices -= center
    scale = np.max(np.linalg.norm(scene_mesh.vertices, axis=1))
    scene_mesh.vertices /= (scale / 0.8) # Keep it slightly inside the view boundaries
    
    # 3. Setup Pyrender Scene
    mesh = pyrender.Mesh.from_trimesh(scene_mesh)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255]) # Pure white background
    scene.add(mesh)
    
    # Add a Perspective Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # Add a strong Directional Light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    
    # Create the offscreen renderer (518x518 resolution)
    renderer = pyrender.OffscreenRenderer(518, 518)
    
    # Orbit parameters
    azimuths = np.linspace(np.pi / 9, 2 * np.pi + np.pi / 9, num_views, endpoint=False)
    elevations = np.linspace(-np.pi / 5, -np.pi / 18, num_views, endpoint=False) # -np.pi / 8  # Camera looking slightly down
    radius = 2.0           # Distance from the chair

    rgb_images = []
    sketch_images = []
    
    for i, az in enumerate(azimuths):
        # Calculate camera orbit pose mathematically
        camera_pose = np.eye(4)
        camera_pose[2, 3] = radius
        camera_pose = tf.rotation_matrix(elevations[i], [1, 0, 0]) @ camera_pose
        camera_pose = tf.rotation_matrix(az, [0, 1, 0]) @ camera_pose
        
        # Attach camera and light to this specific angle
        cam_node = scene.add(camera, pose=camera_pose)
        light_node = scene.add(light, pose=camera_pose)
        
        # Render the RGB image
        color, _ = renderer.render(scene)
        
        # --- OPENCV CANNY EDGE DETECTION ---
        # Convert RGB to Grayscale
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny Edge Detection (You can tune the 50, 150 thresholds later!)
        edges = cv2.Canny(gray, 50, 150)
        
        # Invert colors: Make background white and edges black (looks more like a human sketch)
        sketch = cv2.bitwise_not(edges)
        
        rgb_images.append(color)
        sketch_images.append(sketch)
        
        # Clean up the scene for the next angle
        scene.remove_node(cam_node)
        scene.remove_node(light_node)
        
    renderer.delete()
    return rgb_images, sketch_images

def save_sketches (num_views=6, category="chair"):
  # 1. Define your folders
  mesh_folder = os.path.join("dataloader/data", category, "meshes")
  img_folder = os.path.join("dataloader/data", category, "images")
  sketch_folder = os.path.join("dataloader/data", category, "sketches")

  # Create output folders if they don't exist
  os.makedirs(img_folder, exist_ok=True)
  os.makedirs(sketch_folder, exist_ok=True)

  # 2. Get a list of all .glb files in the mesh folder
  mesh_files = glob.glob(os.path.join(mesh_folder, "*.glb"))
  total_files = len(mesh_files)
  print(f"Found {total_files} meshes. Starting batch generation...")

  # 3. Loop through every mesh and save the pairs
  for index, glb_path in enumerate(mesh_files):
      # Extract the unique ID (filename without the .glb extension)
      uid = os.path.splitext(os.path.basename(glb_path))[0]
      print(f"[{index + 1}/{total_files}] Processing: {uid}")
      
      try:
          # Generate 4 views (Front, Right, Back, Left)
          rgbs, sketches = generate_sketch_pairs(glb_path, num_views=num_views)
          
          for view_idx in range(len(rgbs)):
              # Create a matching filename for both (e.g., 304253851_view0.png)
              filename = f"{uid}_{view_idx}.png"
              
              # --- SAVE RGB IMAGE ---
              # IMPORTANT: OpenCV saves in BGR, so we must swap RGB -> BGR first!
              bgr_image = cv2.cvtColor(rgbs[view_idx], cv2.COLOR_RGB2BGR)
              img_save_path = os.path.join(img_folder, filename)
              cv2.imwrite(img_save_path, bgr_image)
              
              # --- SAVE SKETCH IMAGE ---
              sketch_save_path = os.path.join(sketch_folder, filename)
              cv2.imwrite(sketch_save_path, sketches[view_idx])
              
      except Exception as e:
          print(f"  -> ERROR processing {uid}: {e}")
          continue # If one mesh is broken, skip it and keep going!

  print("\nBatch processing complete! Check your 'images' and 'sketches' folders.")

def sparse_collate_fn(batch):
    """
    Custom collate function to handle variable-length sparse 3D latents,
    while preserving all 2D images and string paths.
    """
    # 1. Stack the standard fixed-size tensors (Images and Sketches)
    batched_data = {
        "uid": [item["uid"] for item in batch],
        "view_id": [item["view_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "sketch_path": [item["sketch_path"] for item in batch],
        "mesh_path": [item["mesh_path"] for item in batch],
        # Stack 2D tensors into [B, C, H, W]
        "image": torch.stack([item["image"] for item in batch]),
        "sketch": torch.stack([item["sketch"] for item in batch]),
        "ss_latent": torch.stack([item["ss_latent"] for item in batch]),
        "cond_tokens": torch.stack([item["cond_tokens"] for item in batch]),
        "neg_cond_tokens": torch.stack([item["neg_cond_tokens"] for item in batch])
    }
    
    batched_feats = []
    batched_coords = []
    
    # 2. Process the variable-length 3D sparse tensors
    for batch_idx, item in enumerate(batch):
        feats = item["latent_feats"]
        coords = item["latent_coords"]
        
        # Create a column of the current batch index: [N, 1]
        batch_idx_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
        
        # Append the batch index to the coordinates: [N, 4] -> (batch_idx, x, y, z)
        coords_with_batch = torch.cat([batch_idx_col, coords], dim=1)
        
        batched_feats.append(feats)
        batched_coords.append(coords_with_batch)
        
    # Concatenate all sparse points into single massive lists
    batched_data["latent_feats"] = torch.cat(batched_feats, dim=0)
    batched_data["latent_coords"] = torch.cat(batched_coords, dim=0)
    
    return batched_data