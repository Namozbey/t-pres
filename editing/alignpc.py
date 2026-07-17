import os
import open3d as o3d
import numpy as np


def compute_scale_and_center(source, target):

    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    src_min = src_pts.min(axis=0)
    src_max = src_pts.max(axis=0)

    tgt_min = tgt_pts.min(axis=0)
    tgt_max = tgt_pts.max(axis=0)

    src_center = (src_min + src_max) / 2
    tgt_center = (tgt_min + tgt_max) / 2


    # object sizes
    src_extent = src_max - src_min
    tgt_extent = tgt_max - tgt_min


    # use largest dimension
    src_size = np.max(src_extent)
    tgt_size = np.max(tgt_extent)


    scale = tgt_size / (src_size + 1e-8)

    return scale, src_center, tgt_center



def align_pcs_sim3(
        source_ply: str,
        target_ply: str,
        output_path="editing/output/aligned.ply"
):

    source = o3d.io.read_point_cloud(source_ply)
    target = o3d.io.read_point_cloud(target_ply)


    # ----------------------------
    # 1. Scale source
    # ----------------------------

    scale, src_center, tgt_center = compute_scale_and_center(
    source, 
    target
)

    src_pts = np.asarray(source.points)

    # scale around source center and move to target center
    src_pts = (src_pts - src_center) * scale + tgt_center


    source_scaled = o3d.geometry.PointCloud()
    source_scaled.points = o3d.utility.Vector3dVector(src_pts)


    # ----------------------------
    # 2. ICP estimates only R + t
    # ----------------------------

    icp = o3d.pipelines.registration.registration_icp(
        source_scaled,
        target,
        0.05,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T_icp = icp.transformation


    # ----------------------------
    # 3. Export aligned cloud
    # ----------------------------

    source_aligned = o3d.geometry.PointCloud()

    pts = np.asarray(source.points)

    # IMPORTANT: same transform used for ICP
    pts = (pts - src_center) * scale + tgt_center


    pts_h = np.concatenate(
        [
            pts,
            np.ones((pts.shape[0],1))
        ],
        axis=1
    )

    pts_aligned = (T_icp @ pts_h.T).T[:,:3]


    source_aligned.points = o3d.utility.Vector3dVector(
        pts_aligned
    )


    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    o3d.io.write_point_cloud(
        output_path,
        source_aligned
    )

    # ----------------------------
    # 4. Create complete SIM(3) matrix
    # ----------------------------

    R = T_icp[:3, :3]
    t = T_icp[:3, 3]


    T_sim3 = np.eye(4)

    # rotation + scale
    T_sim3[:3, :3] = R * scale


    # translation
    T_sim3[:3, 3] = (
        -scale * R @ src_center
        + R @ tgt_center
        + t
    )

    # Only rotation + translation
    return T_sim3