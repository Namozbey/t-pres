# # inspect_npz.py

# import os
# import glob
# import numpy as np

# # ============================================================
# # CONFIG
# # ============================================================

# LATENT_DIR = "./sketch2mesh_dataloader/data/chair/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"

# # ============================================================
# # FIND FILES
# # ============================================================

# npz_files = glob.glob(os.path.join(LATENT_DIR, "*.npz"))

# if len(npz_files) == 0:
#     print("No .npz files found.")
#     exit()

# print(f"Found {len(npz_files)} latent files.\n")

# # ============================================================
# # INSPECT FIRST FILE
# # ============================================================

# sample_path = npz_files[0]

# print("=" * 60)
# print(f"Inspecting file:")
# print(sample_path)
# print("=" * 60)

# data = np.load(sample_path)

# # ============================================================
# # PRINT KEYS
# # ============================================================

# print("\nAvailable keys:")
# for key in data.keys():
#     print(f"  - {key}")

# # ============================================================
# # PRINT SHAPES / DTYPES / STATS
# # ============================================================

# print("\nTensor Information:\n")

# for key in data.keys():
#     arr = data[key]

#     print(f"KEY: {key}")
#     print(f"  shape       : {arr.shape}")
#     print(f"  dtype       : {arr.dtype}")

#     # numeric stats
#     if np.issubdtype(arr.dtype, np.number):
#         print(f"  min         : {arr.min()}")
#         print(f"  max         : {arr.max()}")
#         print(f"  mean        : {arr.mean():.6f}")
#         print(f"  std         : {arr.std():.6f}")

#     # preview
#     flat = arr.reshape(-1)

#     preview_count = min(10, len(flat))
#     print(f"  first values: {flat[:preview_count]}")

#     print("-" * 50)

# # ============================================================
# # SPECIAL CHECKS
# # ============================================================

# if "coords" in data.keys():
#     coords = data["coords"]

#     print("\nCOORD ANALYSIS")
#     print("-" * 50)

#     print(f"coords shape: {coords.shape}")

#     if coords.ndim == 2:
#         print(f"coord dimension per token: {coords.shape[1]}")

#         mins = coords.min(axis=0)
#         maxs = coords.max(axis=0)

#         for i in range(coords.shape[1]):
#             print(f"dim {i}: min={mins[i]} max={maxs[i]}")

# if "feats" in data.keys():
#     feats = data["feats"]

#     print("\nFEATURE ANALYSIS")
#     print("-" * 50)

#     print(f"feats shape: {feats.shape}")

#     if feats.ndim == 2:
#         print(f"feature dimension: {feats.shape[1]}")

# print("\nInspection complete.")


# inspect_trellis_models.py

# import torch
# from trellis.pipelines import TrellisImageTo3DPipeline

# # ============================================================
# # LOAD PIPELINE
# # ============================================================

# print("Loading TRELLIS pipeline...\n")

# pipeline = TrellisImageTo3DPipeline.from_pretrained(
#     "microsoft/TRELLIS-image-large"
# )

# print("\nPipeline loaded successfully.\n")

# # ============================================================
# # INSPECT AVAILABLE MODELS
# # ============================================================

# print("=" * 80)
# print("AVAILABLE MODELS")
# print("=" * 80)

# for name, model in pipeline.models.items():

#     print(f"\nMODEL NAME: {name}")
#     print("-" * 80)

#     print(f"TYPE: {type(model)}")

#     # --------------------------------------------------------
#     # PARAMETER COUNT
#     # --------------------------------------------------------

#     try:
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(
#             p.numel() for p in model.parameters()
#             if p.requires_grad
#         )

#         print(f"TOTAL PARAMS     : {total_params:,}")
#         print(f"TRAINABLE PARAMS : {trainable_params:,}")

#     except Exception as e:
#         print(f"Could not count parameters: {e}")

#     # --------------------------------------------------------
#     # COMMON ATTRIBUTES
#     # --------------------------------------------------------

#     interesting_attrs = [
#         "resolution",
#         "in_channels",
#         "out_channels",
#         "hidden_size",
#         "num_heads",
#         "depth",
#         "voxel_resolution",
#         "latent_dim",
#         "model_channels"
#     ]

#     print("\nImportant Attributes:")

#     found_any = False

#     for attr in interesting_attrs:
#         if hasattr(model, attr):
#             found_any = True
#             try:
#                 print(f"  {attr}: {getattr(model, attr)}")
#             except:
#                 pass

#     if not found_any:
#         print("  (none found)")

#     # --------------------------------------------------------
#     # FORWARD SIGNATURE
#     # --------------------------------------------------------

#     print("\nForward Signature:")

#     try:
#         import inspect

#         sig = inspect.signature(model.forward)
#         print(f"  forward{sig}")

#     except Exception as e:
#         print(f"  Could not inspect signature: {e}")

#     # --------------------------------------------------------
#     # CHILD MODULES
#     # --------------------------------------------------------

#     print("\nTop-Level Children:")

#     try:
#         children = list(model.named_children())

#         if len(children) == 0:
#             print("  (no child modules)")
#         else:
#             for child_name, child_module in children[:20]:
#                 print(f"  - {child_name}: {type(child_module).__name__}")

#             if len(children) > 20:
#                 print(f"  ... and {len(children) - 20} more")

#     except Exception as e:
#         print(f"  Could not inspect children: {e}")

# # ============================================================
# # EXTRA: PIPELINE ATTRIBUTES
# # ============================================================

# print("\n")
# print("=" * 80)
# print("PIPELINE ATTRIBUTES")
# print("=" * 80)

# attrs = dir(pipeline)

# ignore = [
#     "__class__",
#     "__dict__",
#     "__doc__",
#     "__module__",
#     "__weakref__"
# ]

# for attr in attrs:
#     if attr.startswith("_"):
#         continue

#     if attr in ignore:
#         continue

#     try:
#         value = getattr(pipeline, attr)

#         if callable(value):
#             print(f"{attr}()")
#         else:
#             print(f"{attr}: {type(value)}")

#     except:
#         pass

# print("\nInspection complete.")



import numpy as np

path = "YOUR_FILE.npz"

data = np.load(path)

print("KEYS:")
print(data.files)

print("\n--- CONTENT INFO ---")

for k in data.files:
    arr = data[k]
    print(f"{k}:")
    print(f"  shape = {arr.shape}")
    print(f"  dtype = {arr.dtype}")

    if arr.ndim > 0:
        print(f"  min = {arr.min()}")
        print(f"  max = {arr.max()}")

    print()