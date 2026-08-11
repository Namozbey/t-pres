import os
import trimesh
import open3d as o3d

def sample_mesh_surface(
    mesh_path: str,
    out_path: str = "state/generated.ply", 
    num_points: int = 100000):
    """
    Sample points uniformly from mesh surface and save as .ply.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mesh = trimesh.load(mesh_path, force='mesh')

    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # optional face colors
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
        colors = mesh.visual.face_colors[face_idx, :3] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(out_path, pcd)
    return pcd