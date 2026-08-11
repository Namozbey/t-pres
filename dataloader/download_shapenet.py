import os
import random
import shutil
from huggingface_hub import HfApi, hf_hub_download

#----------------------------------------------
# Configuration
#----------------------------------------------

repo_id = "ShapeNet/shapenetcore-glb"
output_dir = "dataloader/data/eval/meshes"
repo_type = "dataset"
samples_per_folder = 10

# Reproducible sampling
random.seed(42)

api = HfApi()

# Get all files
files = sorted(api.list_repo_files(
    repo_id=repo_id,
    repo_type=repo_type
))

# Find subfolders
subfolders = sorted(
    set(f.split("/")[0] for f in files if "/" in f)
)

print("Found subfolders:", subfolders)
print(len(subfolders))

# Create one single output folder
os.makedirs(output_dir, exist_ok=True)


counter = 0

for folder in subfolders:

    print(f"Processing {folder}")

    # Files inside this subfolder
    folder_files = [
        f for f in files
        if f.startswith(folder + "/")
        and f.endswith(".glb")
    ]

    if len(folder_files) == 0:
        continue

    # Select 10 random files
    selected = random.sample(
        folder_files,
        min(samples_per_folder, len(folder_files))
    )


    for file in selected:

        print("Downloading:", file)

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file
        )

        # Keep original filename
        filename = os.path.basename(file)

        destination = os.path.join(
            output_dir,
            filename
        )

        # Avoid overwriting if different folders have same filename
        if os.path.exists(destination):
            filename = f"{folder}_{filename}"
            destination = os.path.join(
                output_dir,
                filename
            )

        shutil.copy(
            downloaded_path,
            destination
        )

        counter += 1


print(f"\nDone! Downloaded {counter} meshes.")
print(f"Saved to: {output_dir}")