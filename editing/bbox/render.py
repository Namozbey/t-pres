import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import numpy as np
import trimesh.transformations as tf
import cv2
import matplotlib.pyplot as plt

def render_single_view_for_bbox(
    glb_path, 
    fx, fy, cx, cy,  # Explicit intrinsics in pixels
    azimuth=np.pi/4, 
    elevation=-np.pi/4, 
    radius=3.5
):
    scene_mesh = trimesh.load(glb_path, force='mesh')
    
    # 1. Normalize mesh center and scale
    vertices = scene_mesh.vertices
    center = vertices.mean(axis=0)
    scene_mesh.vertices -= center
    scale = np.max(np.linalg.norm(scene_mesh.vertices, axis=1))
    scene_mesh.vertices /= (scale / 0.8)
    
    mesh = pyrender.Mesh.from_trimesh(scene_mesh)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    scene.add(mesh)
    
    # 2. Define Intrinsics Matrix (K) for downstream unprojection
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ])
    
    # 3. Create Camera directly from custom intrinsics
    # (Optional: set near/far clipping planes if working with non-standard scales)
    camera = pyrender.IntrinsicsCamera(
        fx=fx, 
        fy=fy, 
        cx=cx, 
        cy=cy, 
        znear=0.05, 
        zfar=100.0
    )
    
    # Image resolution is derived from principal points / bounds (or explicit width/height)
    width = int(cx * 2)
    height = int(cy * 2)
    renderer = pyrender.OffscreenRenderer(width, height)
    
    # 4. Position camera via azimuth and elevation
    camera_pose = np.eye(4)
    camera_pose[2, 3] = radius
    camera_pose = tf.rotation_matrix(elevation, [1, 0, 0]) @ camera_pose
    camera_pose = tf.rotation_matrix(azimuth, [0, 1, 0]) @ camera_pose
    
    R_c2w = camera_pose[:3, :3]
    t_c2w = camera_pose[:3, 3]
    
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    
    r_img, depth_map = renderer.render(scene)
    renderer.delete()
    
    return r_img, depth_map, camera_pose, scale, center

import numpy as np

import numpy as np

def get_cam_to_mesh_matrix(camera_pose, scale, center):
    """
    Constructs a single 4x4 homogeneous transformation matrix that maps 
    points from OpenGL Camera Space (output of generate_pcd) 
    directly to Original Mesh Space.
    """
    # 1. Camera to Normalized World is exactly the PyRender camera_pose
    # (No need for CV2GL flip since generate_pcd already did it)
    M_c2w = camera_pose
    
    # 2. Normalized World back to Original Mesh Space
    s = scale / 0.8
    M_denorm = np.eye(4)
    M_denorm[0, 0] = s
    M_denorm[1, 1] = s
    M_denorm[2, 2] = s
    M_denorm[0:3, 3] = center  # Translate back to original center
    
    # 3. Chain them together (Right-to-Left multiplication)
    # M_final = Denormalize * CameraPose
    M_final = M_denorm @ M_c2w
    
    return M_final

if __name__ == "__main__":
    mesh_path = "editing/state/mesh_1.glb"

    # Validate file presence before running
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at: '{mesh_path}'")
        exit()

    fx, fy, cx, cy = (1209.865588803089, 1195.9805743243244, 256.0, 256.0)

    # Highly-aligned Camera Angles (keep these to align render view to your photo perspective)
    target_az = -np.pi / 3.0       
    target_el = -np.pi / 10.5       
    
    img1, depth_map, camera_pose, scale, center = render_single_view_for_bbox(mesh_path, fx, fy, cx, cy, target_az, target_el)
    print(depth_map[200, 200])
    plt.imsave('editing/state/depth_image.png', depth_map, cmap='plasma')

    cv2.imwrite(
        "editing/state/render_from_mesh.png",
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    )