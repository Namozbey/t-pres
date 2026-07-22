import os
import random
import shutil
from huggingface_hub import HfApi, hf_hub_download

# Configuration
repo_id = "ShapeNet/shapenetcore-glb"
output_dir = "./data_s"
repo_type = "dataset"
samples_per_folder = 10

# Make sampling reproducible
random.seed(42)

api = HfApi()

files = sorted(api.list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
))

subfolders = sorted(set(f.split("/")[0] for f in files if "/" in f))

print("Found subfolders:", sorted(subfolders))

for folder in subfolders:
    print(f"Processing {folder}")

    folder_files = [
        f for f in files
        if f.startswith(folder + "/")
    ]

    selected = random.sample(
        folder_files,
        min(samples_per_folder, len(folder_files))
    )

    # Create only the desired folder
    local_folder = os.path.join(output_dir, folder)
    os.makedirs(local_folder, exist_ok=True)

    for file in selected:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file
        )

        # Copy only the actual file into your folder
        shutil.copy(
            downloaded_path,
            os.path.join(local_folder, os.path.basename(file))
        )

print("Done!")