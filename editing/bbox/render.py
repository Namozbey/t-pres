import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import numpy as np
import trimesh.transformations as tf
import cv2

def render_single_view_for_bbox(glb_path, azimuth=np.pi/4, elevation=-np.pi/4, radius=1.5):
    """
    Renders the generated mesh from exactly ONE specified camera angle.
    Returns normalized render, depth map, and the parameters needed to un-normalize later.
    """
    scene_mesh = trimesh.load(glb_path, force='mesh')
    
    # 1. Keep track of original center and scale
    vertices = scene_mesh.vertices
    center = vertices.mean(axis=0)
    scene_mesh.vertices -= center
    scale = np.max(np.linalg.norm(scene_mesh.vertices, axis=1))
    
    # Normalize mesh geometry so it fits perfectly in the camera frame
    scene_mesh.vertices /= (scale / 0.8)
    
    mesh = pyrender.Mesh.from_trimesh(scene_mesh)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    scene.add(mesh)
    
    # 2. Define Camera Intrinsics (K)
    width, height = 518, 518
    yfov = np.pi / 3.0
    aspect_ratio = 1.0
    
    fy = (height / 2.0) / np.tan(yfov / 2.0)
    fx = fy * aspect_ratio
    cx, cy = width / 2.0, height / 2.0
    
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ])
    
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    renderer = pyrender.OffscreenRenderer(width, height)
    
    # 3. Position camera based on chosen azimuth and elevation
    camera_pose = np.eye(4)
    camera_pose[2, 3] = radius
    camera_pose = tf.rotation_matrix(elevation, [1, 0, 0]) @ camera_pose
    camera_pose = tf.rotation_matrix(azimuth, [0, 1, 0]) @ camera_pose
    
    R_c2w = camera_pose[:3, :3]
    t_c2w = camera_pose[:3, 3]
    
    # Attach camera and render
    cam_node = scene.add(camera, pose=camera_pose)
    light_node = scene.add(light, pose=camera_pose)
    
    r_img, depth_map = renderer.render(scene)
    
    renderer.delete()
    return r_img, depth_map, K, R_c2w, t_c2w, scale, center

if __name__ == "__main__":
    mesh_path = "mesh_pass.glb"

    # Validate file presence before running
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at: '{mesh_path}'")
        exit()

    # Highly-aligned Camera Angles (keep these to align render view to your photo perspective)
    target_az = -np.pi / 2.7       
    target_el = -np.pi / 6.5       
    
    img1, _, _, _, _, _, _ = render_single_view_for_bbox(mesh_path, azimuth=target_az, elevation=target_el)

    cv2.imwrite(
        "render_from_mesh.png",
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    )