# 🖼️ Sketch-to-3D Dataset Generation Pipeline

This repository contains an automated, end-to-end pipeline for generating paired `{sketch, image, 3D mesh, latents}` datasets. It leverages the **Objaverse** dataset for high-quality ground-truth 3D topologies, utilizes **Open3D** for headless multi-view rendering, and extracts stylized wireframes via **OpenCV**.

It includes a fully integrated **PyTorch DataLoader** optimized for training generative models (e.g., Diffusion models, Autoregressive Transformers).

---

## 🚀 1. Data Generation Pipeline

The data generation pipeline is broken down into four distinct steps. **Note: All commands must be run from the main directory (not inside the `dataloader/` folder).**

**Step 1: Download Data and Render Images & Sketches**
This command downloads categorized 3D `.glb` files, normalizes the geometry, and renders multi-view RGB images and Canny edge sketches.

```bash
python -m dataloader.render --category chair --download_limit 10 --num_views 6

```

**Step 2: Data Preparation for Training (Latent Generation)**
This extracts features and generates SLATs and sparse structured latents. It reads the target directory and category from the `config.py`, or it can be overridden via a flag (e.g., `--data_dir dataloader/data/chair`).

```bash
python -m dataloader.generate_slats

```

**Step 3: Precompute DINO Cache**
This precomputes the DINO values for the input sketches to speed up training.

```bash
python -m dataloader.precompute_dino_cache

```

**Step 4: Flux Sketch-to-Image Generation (Optional Dataset Prep)**
This prepares a dataset of generated images directly from the sketches using the Flux model.

```bash
python -m dataloader.flux_img2img

```

**Step 5: Create Train/Test Splits (Required)**
This creates the data split files required by the DataLoader to properly assign data to training, validation, or test sets. It automatically gets the directory and category from `config.py`.

```bash
python -m dataloader.create_split

```

---

## 📂 2. Output Directory Structure

Running the complete pipeline will automatically organize your data into a target folder based on `config.py` parameters. For example, using a `data_root` of `./dataloader/data` and a `category` of `"chair"`, the structure will look like this:

```text
data/
└── chair/
    ├── meshes/        # Raw .glb 3D files (e.g., 8a4a3a90.glb)
    ├── images/        # Rendered RGB views (e.g., 8a4a3a90_0.png)
    ├── sketches/      # Extracted edge sketches (e.g., 8a4a3a90_0.png)
    ├── features/      # Extracted DINO features
    ├── latents/       # SLATs (Structured Latent Representations)
    ├── ss_latents/    # Sparse structured latents
    ├── renders/       # 150 diverse renderings of meshes used to extract voxels
    ├── voxels/        # Extracted voxels
    ├── sketch_cache/  # Precomputed DINO values for input sketches
    └── flux_images/   # Generated images from sketches via Flux

```

---

## 🧠 3. PyTorch Integration (DataLoader)

Once your data is generated, you can seamlessly stream it into your training loops using the included `SketchMeshDataset` class. It features lazy-loading, caching, and a custom collate function (`sparse_collate_fn`) designed specifically to handle variable-length token arrays.

By default, the training configuration sets the `image_size` to `518` and `batch_size` to `2`.

### Example: Testing the DataLoader (`demo.ipynb`)

```python
from dataset import SketchMeshDataset
from torch.utils.data import DataLoader

# 1. Initialize the Dataset
dataset = SketchMeshDataset(
    root_dir="data",
    category="chair",
    image_size=518,
    split="test"
)

# 2. Initialize the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,                           # How many images to process at once
    shuffle=False,                          # Randomize the order (crucial for training)
    num_workers=2,                          # Multi-processing for faster data loading
    collate_fn=dataset.sparse_collate_fn,   # Handles variable length arrays
    drop_last=True                          # Drops the last incomplete batch
)

# 3. Test the DataLoader loop
seen_uids = set()

for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    uids = batch["uid"]
    view_ids = batch["view_id"]

    # --------------------------------------------------
    # 1. SHAPE CHECKS
    # --------------------------------------------------
    print(f" - Images shape:  {batch['image'].shape}")
    print(f" - Sketches shape:{batch['sketch'].shape}")
    print(f" - latent_feats shape:{batch['latent_feats'].shape}")
    print(f" - latent_coords shape:{batch['latent_coords'].shape}")
    print(f" - ss_latents shape:{batch['ss_latent'].shape}")
    print(f" - cond_tokens shape:{batch['cond_tokens'].shape}")
    print(f" - neg_cond_tokens shape:{batch['neg_cond_tokens'].shape}")

    print(f" - mesh_paths in batch: {batch['mesh_path']}")
    print(f" - image_paths in batch: {batch['image_path']}")
    print(f" - sketch_paths in batch: {batch['sketch_path']}")

    # --------------------------------------------------
    # 2. UID CONSISTENCY CHECK
    # --------------------------------------------------
    unique_uids = set(uids)
    print("UIDs in batch:", unique_uids)
    if len(unique_uids) > 1:
        print("WARNING: mixed UIDs in same batch!")

    # --------------------------------------------------
    # 3. VIEW DIVERSITY CHECK
    # --------------------------------------------------
    print("View IDs:", view_ids)
    if len(set(view_ids)) < len(view_ids):
        print("⚠ WARNING: duplicate view_ids in batch!")

    # --------------------------------------------------
    # 4. SPLIT LEAKAGE CHECK
    # --------------------------------------------------
    for uid, v in zip(uids, view_ids):
        key = f"{uid}_{v}"
        if key in seen_uids:
            print("WARNING: duplicate sample seen:", key)
        seen_uids.add(key)

    # --------------------------------------------------
    # 5. LATENT SANITY CHECK
    # --------------------------------------------------
    feats = batch["latent_feats"]
    print("latent_feats mean/std:", feats.mean().item(), feats.std().item())

    ss = batch["ss_latent"]
    print("ss_latent mean/std:", ss.mean().item(), ss.std().item())

    break # Just run one batch to verify

```

### Expected Output Log

```text
Loaded 19 valid data pairs for category 'chair2'.
Batch 1
 - Images shape:  torch.Size([2, 3, 518, 518])
 - Sketches shape:torch.Size([2, 3, 518, 518])
 - latent_feats shape:torch.Size([17895, 8])
 - latent_coords shape:torch.Size([17895, 4])
 - ss_latents shape:torch.Size([2, 8, 16, 16, 16])
 - cond_tokens shape:torch.Size([2, 1374, 1024])
 - neg_cond_tokens shape:torch.Size([2, 1374, 1024])
 - mesh_paths in batch: ['data\chair2\meshes\09f6fe724dd1496dafd242f3022209be.glb', 'data\chair2\meshes\1e5919f8a1ef4618b4dea274e028f857.glb']
 - image_paths in batch: ['data\chair2\images\09f6fe724dd1496dafd242f3022209be_0.png', 'data\chair2\images\1e5919f8a1ef4618b4dea274e028f857_0.png']
 - sketch_paths in batch: ['data\chair2\sketches\09f6fe724dd1496dafd242f3022209be_0.png', 'data\chair2\sketches\1e5919f8a1ef4618b4dea274e028f857_0.png']
UIDs in batch: {'1e5919f8a1ef4618b4dea274e028f857', '09f6fe724dd1496dafd242f3022209be'}
WARNING: mixed UIDs in same batch!
View IDs: ['0', '0']
⚠ WARNING: duplicate view_ids in batch!
latent_feats mean/std: -0.3088565766811371 2.7937424182891846
ss_latent mean/std: 0.007412733510136604 0.37213534116744995

```
