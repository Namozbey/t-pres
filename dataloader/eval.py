import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import torch
import trimesh
import trimesh.registration
from PIL import Image
import shutil
import clip

from pytorch_fid import fid_score

from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    TexturesVertex,
)

# ============================================================
# CONFIG
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

FIXED_ELEVS = [-20, 0, 20]
FIXED_AZIMS = np.linspace(0, 360, 8, endpoint=False)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_vertices(vertices):
    vertices = vertices - vertices.mean(axis=0)
    scale = np.linalg.norm(vertices, axis=1).max()
    return vertices / (scale + 1e-8)

# ============================================================
# ROTATION CANDIDATES
# ============================================================

def get_rotation_candidates():
    rotations = []
    for angle in [0, 90, 180, 270]:
        rad = np.deg2rad(angle)
        R = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        rotations.append(R)
    return rotations

ROTATIONS = get_rotation_candidates()

# ============================================================
# MESH LOADING
# ============================================================

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    verts = normalize_vertices(mesh.vertices.copy())

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

    return Meshes([verts], [faces])

# ============================================================
# ICP + ROTATION SEARCH
# ============================================================

def align_meshes(pred_mesh, gt_mesh, n_points=10000):

    pred_pts = sample_points_from_meshes(pred_mesh, n_points)
    gt_pts = sample_points_from_meshes(gt_mesh, n_points)

    pred_np = pred_pts.squeeze(0).detach().cpu().numpy()
    gt_np = gt_pts.squeeze(0).detach().cpu().numpy()

    best_cd = float("inf")
    best_R, best_T = None, None

    for R in ROTATIONS:

        # ----------------------------------------------------
        # APPLY GLOBAL ROTATION
        # ----------------------------------------------------

        pred_rot = pred_np @ R.T

        # ----------------------------------------------------
        # ICP REFINE
        # ----------------------------------------------------

        T, _, _ = trimesh.registration.icp(
            pred_rot,
            gt_np,
            max_iterations=100
        )

        pred_h = np.concatenate([pred_rot, np.ones((len(pred_rot), 1))], axis=1)
        aligned = (T @ pred_h.T).T[:, :3]

        # ----------------------------------------------------
        # CHAMFER FOR SELECTION
        # ----------------------------------------------------

        aligned_t = torch.tensor(aligned, dtype=torch.float32, device=device).unsqueeze(0)
        gt_t = torch.tensor(gt_np, dtype=torch.float32, device=device).unsqueeze(0)

        cd, _ = pytorch3d_chamfer(aligned_t, gt_t)
        cd = cd.item()

        if cd < best_cd:
            best_cd = cd
            best_R = R
            best_T = T

    return best_R, best_T, best_cd

# ============================================================
# APPLY ALIGNMENT TO FULL MESH
# ============================================================

def apply_alignment(mesh, R, T):

    verts = mesh.verts_packed().detach().cpu().numpy()

    # global rotation
    verts = verts @ R.T

    # ICP transform
    verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    verts = (T @ verts_h.T).T[:, :3]

    verts = torch.tensor(verts, dtype=torch.float32, device=device)

    return Meshes([verts], mesh.faces_list())

# ============================================================
# CHAMFER
# ============================================================

def chamfer(mesh1, mesh2, n=10000):
    p1 = sample_points_from_meshes(mesh1, n)
    p2 = sample_points_from_meshes(mesh2, n)

    cd, _ = pytorch3d_chamfer(p1, p2)
    return cd.item()

# ============================================================
# RENDERING
# ============================================================

def get_renderer(image_size=518):

    cameras = FoVPerspectiveCameras(device=device)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    lights = PointLights(device=device)

    return MeshRenderer(
        MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        SoftPhongShader(device=device, lights=lights)
    )

renderer = get_renderer()

# ============================================================
# RENDERING
# ============================================================

def render_mesh_views(mesh):

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])

    mesh_r = Meshes([verts], [faces], textures=textures)

    images = []

    for elev in FIXED_ELEVS:
        for azim in FIXED_AZIMS:

            R, T = look_at_view_transform(2.5, elev, float(azim))

            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

            img = renderer(mesh_r, cameras=cameras)[0, ..., :3]
            img = (img.detach().cpu().numpy() * 255).astype(np.uint8)

            images.append(Image.fromarray(img))

    return images

# ============================================================
# CLIP
# ============================================================

def embed(img):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = clip_model.encode_image(x)
    return f / (f.norm(dim=-1, keepdim=True) + 1e-8)

def clip_similarity(sketch, imgs):

    sk = embed(sketch)

    sims = []
    for img in imgs:
        sims.append((sk @ embed(img).T).item())

    sims = sorted(sims, reverse=True)

    return float(np.mean(sims[:5]))

# ============================================================
# SAVE + FID SAFETY
# ============================================================

def save_images(imgs, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        img.save(out_dir / f"{prefix}_{i}.png")

def compute_fid(real_dir, fake_dir):

    real_files = len(list(Path(real_dir).glob("*.png")))
    fake_files = len(list(Path(fake_dir).glob("*.png")))

    if real_files == 0 or fake_files == 0:
        print(f"[WARN] Empty FID folder: real={real_files}, fake={fake_files}")
        return float("inf")

    return fid_score.calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=min(16, real_files, fake_files),
        device=device,
        dims=2048
    )

# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(data_root,
    real_dir="./renders/real",
    fake_dir="./renders/fake"):

    data_root = Path(data_root)

    shutil.rmtree(real_dir, ignore_errors=True)
    shutil.rmtree(fake_dir, ignore_errors=True)

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)

    (real_dir / "sketch").mkdir(parents=True, exist_ok=True)
    (fake_dir / "sketch").mkdir(parents=True, exist_ok=True)

    split_info = {}
    split_path = "data/eval/split.json"
    with open(split_path, "r") as f:
        split_info = json.load(f)

    gt_cache = {}

    cd_sketch = []
    clip_sketch = []

    for category_dir in sorted(data_root.iterdir()):

        if not category_dir.is_dir() or category_dir.name != "eval":
            continue

        mesh_dir = category_dir / "meshes"
        sketch_dir = category_dir / "sketches"
        gen_sketch_dir = category_dir / "gen_meshes"

        for gt_path in tqdm(list(mesh_dir.glob("*.glb")), desc=category_dir.name):
            
            obj_id = gt_path.stem

            if obj_id not in split_info.keys():
                # print(obj_id)
                continue
            else:
                mesh_view_id = split_info[obj_id]["test"][0]

            if obj_id not in gt_cache:
                gt_mesh = load_mesh(str(gt_path))
                gt_cache[obj_id] = render_mesh_views(gt_mesh)
            else:
                gt_mesh = load_mesh(str(gt_path))

            gt_render = gt_cache[obj_id]

            # ================= SKETCH =================
            for pred_path in gen_sketch_dir.glob(f"{obj_id}_{mesh_view_id}.glb"):
                pred_mesh = load_mesh(str(pred_path))
                R, T, _ = align_meshes(pred_mesh, gt_mesh)
                pred_mesh = apply_alignment(pred_mesh, R, T)

                cd_sketch.append(chamfer(gt_mesh, pred_mesh))

                pred_render = render_mesh_views(pred_mesh)

                view_id = pred_path.stem.split("_")[-1]
                sketch = list(sketch_dir.glob(f"{obj_id}_{view_id}.png"))

                if sketch:
                    clip_sketch.append(clip_similarity(Image.open(sketch[0]), pred_render))

                save_images(gt_render, real_dir / "sketch", f"{obj_id}_{view_id}")
                save_images(pred_render, fake_dir / "sketch", f"{obj_id}_{view_id}")

    fid_sketch = compute_fid(real_dir / "sketch", fake_dir / "sketch")

    print("\n===== RESULTS =====")
    print("SKETCH CD:", np.mean(cd_sketch) if cd_sketch else 0)
    print("SKETCH CLIP:", np.mean(clip_sketch) if clip_sketch else 0)
    print("SKETCH FID:", fid_sketch)

    return {
        "sketch_CD": float(np.mean(cd_sketch)) if cd_sketch else 0,
        "sketch_CLIP": float(np.mean(clip_sketch)) if clip_sketch else 0,
        "sketch_FID": fid_sketch,
    }

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(evaluate("./data"))