import os
import glob
import json
import random
from collections import defaultdict
from config import TRAINING_CONFIG

ROOT = TRAINING_CONFIG["data_root"]
CATEGORY = TRAINING_CONFIG["category"]
SEED = 42

random.seed(SEED)

cat_dir = os.path.join(ROOT, CATEGORY)

mesh_dir = os.path.join(cat_dir, "meshes")

split_dict = {}

uid_to_views = defaultdict(list)

# ---------------------------------------
# STEP 1: derive UID from meshes
# ---------------------------------------
for mesh_path in glob.glob(os.path.join(mesh_dir, "*.glb")):

    filename = os.path.basename(mesh_path).split(".")[0]
    uid = filename  # mesh defines identity

    uid_to_views[uid] = list(range(10))  # assume 10 renders per mesh

# ---------------------------------------
# STEP 2: split per mesh
# ---------------------------------------
for uid, views in uid_to_views.items():

    views = sorted(views)
    random.shuffle(views)

    val_views = views[:2]
    test_views = views[2:4]
    train_views = views[4:]

    split_dict[uid] = {
        "train": sorted(train_views),
        "val": sorted(val_views),
        "test": sorted(test_views)
    }

# ---------------------------------------
# STEP 3: save
# ---------------------------------------
os.makedirs(ROOT, exist_ok=True)

with open(os.path.join(ROOT, "split.json"), "w") as f:
    json.dump(split_dict, f, indent=4)

print("Created mesh-based split.json")