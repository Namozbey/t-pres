# Topology-Preserving Sketch-Based Generation of 3D Assets

<div align="center">
  <video src="assets/demo.mp4" width="100%" controls>
    Your browser does not support the video tag.
  </video>
</div>

## Motivation

Current 3D generation approaches often struggle to balance diversity with structural consistency. Existing models either lack diversity when the topology is well-preserved, or they lose topological stability over multiple iterations when diversity is high.

This limitation motivates our primary research question:

> _"Can we generate highly diverse and high-fidelity meshes through diffusion models, and edit parts while preserving the topology of the unedited part?"_

To address this, we leverage existing diffusion-based generation models (FLUX.2 and TRELLIS) to achieve high diversity and propose a novel **attention-caching and masking mechanism** (KV-Cache Engine and Automatic Bounding Box extraction) for the latent space to ensure topological stability during local sketch-based edits.

## Qualitative Results

Our method is capable of generating meshes that closely resemble ground-truth shapes. It accurately interprets user edits in specific regions (such as replacements or removals) while preserving the unedited parts of the original geometry.

![Qualitative Results](assets/results.png)

---

## Setup & Installation

Clone the repository

```bash
git clone --recurse-submodules https://github.com/Namozbey/t-pres.git
cd t-pres
```

Due to the complex interactions between 3D generation (Trellis), Image Generation (Flux.2), and Depth Estimation (Depth Anything 3), strict environment management is required to prevent CUDA and PyTorch conflicts.

**1. Create the environment:**

```bash
conda create -n t_pres python=3.11
conda activate t_pres
```

**2. Install PyTorch:**

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

**3. Install Flash Attention (Crucial for Trellis):**

- **Linux:** `pip install flash-attn --no-build-isolation`
- **Windows:** You must use a pre-compiled wheel matching PyTorch 2.4.0, CUDA 11.8, and Python 3.11 to avoid build errors.

```bash
pip install [https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp311-cp311-win_amd64.whl](https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp311-cp311-win_amd64.whl)

```

**4. Install Windows 3D Rendering Libraries:**

```bash
pip install kaolin==0.18.0 -f [https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html)
pip install nvdiffrast==0.4.0

```

**5. Install Project Dependencies:**

```bash
pip install -r requirements.txt

```

**6. Install Depth Anything V3:**

```bash
git clone [https://github.com/DepthAnything/Depth-Anything-V3.git](https://github.com/DepthAnything/Depth-Anything-V3.git)
cd Depth-Anything-V3
pip install -e . --no-deps
cd ..

```

## Usage

_(Coming soon)_

```bash
# Example usage for sketch-to-mesh editing
python sketch2mesh.py -s my_sketch.png -nc
```
