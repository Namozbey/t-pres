# 🖼️ Sketch-to-3D Dataset Generation Pipeline

This repository contains an automated, end-to-end pipeline for generating paired `{sketch, image, 3D mesh}` datasets. It leverages the **Objaverse** dataset for high-quality ground-truth 3D topologies, utilizes **Open3D** for headless multi-view rendering, and extracts stylized wireframes via **OpenCV**.

It includes a fully integrated **PyTorch DataLoader** optimized for training generative models (e.g., Diffusion models, Autoregressive Transformers).

---

## ⚙️ 1. Installation

Clone the repository and install the required dependencies. It is recommended to use a virtual environment (e.g., `conda` or `venv`).

```bash

# Install dependencies
pip install -r requirements.txt

```

_(Note: PyTorch and Torchvision are included in the requirements, but for optimal GPU performance, install them directly via the [official PyTorch instructions](https://pytorch.org/get-started/locally/) based on your specific CUDA version)._

---

## 🚀 2. Data Generation

The data generation pipeline handles downloading categorized 3D `.glb` files, normalizing the geometry, and rendering multi-view RGB images and Canny edge sketches.

<!-- ### Option A: Command Line Interface (CLI) -->

You can run the full pipeline directly from the terminal using `render.py`.

 <!-- This is highly recommended for batch processing. -->

```bash
# Basic usage with default parameters (Chairs)
python -m dataloader.render

# Custom dataset generation (e.g., generating 6 views for 100 bottles)
python -m dataloader.render --category bottle --download_limit 100 --num_views 6

```

<!-- ### Option B: Python API

You can also import and trigger the pipeline programmatically within your own scripts or Jupyter Notebooks:

```python
from utils import setup_dataset, save_sketches

# 1. Download base 3D meshes
setup_dataset(download_limit=3, category="chair")

# 2. Render RGB images and extract sketches
save_sketches(num_views=6, category="chair")

``` -->

### 📂 Output Directory Structure

Running the generation pipeline will automatically organize your data into the following structure:

```text
data/
└── chair/
    ├── meshes/        # Raw .glb 3D files (e.g., 8a4a3a90.glb)
    ├── images/        # Rendered RGB views (e.g., 8a4a3a90_0.png)
    └── sketches/      # Extracted edge sketches (e.g., 8a4a3a90_0.png)

```

---

## 🧠 3. PyTorch Integration (DataLoader)

Once your data is generated, you can seamlessly stream it into your training loops using the included `SketchMeshDataset` class. It features lazy-loading to prevent out-of-memory errors and automatic sanity checks to ensure paired data integrity.

```python
from dataset import SketchMeshDataset
from torch.utils.data import DataLoader

# 1. Initialize the Dataset
dataset = SketchMeshDataset(
    root_dir="data",
    category="chair",
    image_size=512
)

# 2. Initialize the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,        # Number of images to process at once
    shuffle=True,        # Randomize the order (crucial for training)
    num_workers=4,       # Multi-processing for faster data loading
    drop_last=True       # Drops the last incomplete batch
)

# 3. Training Loop Example
for batch_idx, batch in enumerate(dataloader):
    # Tensors formatted as [batch_size, channels, height, width]
    images = batch['image']       # -> Shape: [8, 3, 512, 512]
    sketches = batch['sketch']    # -> Shape: [8, 3, 512, 512]
    latent_feats = batch['latent_feats']  # -> Shape: [tokens_num, 8]
    latent_coords = batch['latent_coords']  # -> Shape: [tokens_num, 4]

    # Metadata for evaluation/tracking
    uids = batch['uid']           # List of 3D object IDs
    view_ids = batch['view_id']   # List of camera view IDs (e.g., '0, 1')
    mesh_paths = batch['mesh_path'] # Paths to original .glb files
    mesh_paths = batch['image_path'] # Paths to images .png files
    mesh_paths = batch['sketch_path'] # Paths to sketches .png files

    print(f"Processing Batch {batch_idx+1} | Tensors loaded to memory.")

    # Pass to your model...
```

---

## 🛠️ Troubleshooting

- **Missing Display / Headless Rendering Errors:** This pipeline uses Open3D for rendering, which operates natively on Windows without issue. If running on a headless Linux server (like AWS or Google Colab), ensure you have a virtual framebuffer installed (e.g., `xvfb`).

# Offline latent encoding (VAE pre-encoding)

Run:

```bash
python generate_slats.py --data_dir dataloader/data/chair
```


# Sketch/Image to 3D Evaluation Pipeline

## 1. Overview

This project evaluates 3D mesh generation quality from:

- RGB images
- sketches

using:

- **Chamfer Distance (CD)**
- **CLIP Similarity**
- **FID**

The pipeline supports:

- multiple object categories
- multiple input views per object
- automatic mesh alignment
- rendered-view evaluation

---

## 2.Dataset Structure

Your dataset should follow this structure:

```text
data/
├── category/
│   ├── images/
│   │   ├── objectid_0.png
│   │   ├── objectid_1.png
│   │   └── ...
│   │
│   ├── sketches/
│   │   ├── objectid_0.png
│   │   ├── objectid_1.png
│   │   └── ...
│   │
│   ├── meshes/
│   │   ├── objectid.glb
│   │   └── ...
│   │
│   ├── gen_image/
│   │   ├── objectid_0.glb
│   │   ├── objectid_1.glb
│   │   └── ...
│   │
│   ├── gen_sketch/
│   │   ├── objectid_0.glb
│   │   ├── objectid_1.glb
│   │   └── ...
│
├── table/
│   └── ...
```

---

## 3. Evaluation Metrics

### Chamfer Distance (CD)

Measures geometric similarity between:

- predicted mesh
- ground-truth mesh

Procedure:

1. sample surface points
2. align using:
   - global rotation search
   - ICP refinement
3. compute bidirectional Chamfer distance

Lower is better.

---

### CLIP Similarity

Measures semantic consistency between:

- input image/sketch
- rendered predicted mesh views

Procedure:

1. render multiple views of predicted mesh
2. encode with CLIP
3. compare against input image/sketch embedding
4. average top-5 similarities

Higher is better.

---

### FID

Measures distribution similarity between:

- rendered GT meshes
- rendered predicted meshes

Computed separately for:

- image-conditioned generation
- sketch-conditioned generation

Lower is better.

---

## 4. Environment Setup

Before running generation or evaluation, set up the environment using:

```bash
bash setup_env.sh
```

## 5. Running Evaluation

Run:

```bash
python -m dataloader.eval.py
```

Example output:

```text
========== FINAL RESULTS ==========

--- IMAGE CONDITION ---
Chamfer: 0.0123
CLIP: 0.812
FID: 34.5

--- SKETCH CONDITION ---
Chamfer: 0.0181
CLIP: 0.744
FID: 41.2
```
