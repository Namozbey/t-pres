import os
import glob
import json
import random
from config import TRAINING_CONFIG

ROOT = TRAINING_CONFIG["data_root"]
CATEGORY = TRAINING_CONFIG["category"]
SEED = 42

random.seed(SEED)

cat_dir = os.path.join(ROOT, CATEGORY)
mesh_dir = os.path.join(cat_dir, "meshes")

# ---------------------------------------
# STEP 1: get all mesh UIDs
# ---------------------------------------
uids = []

for mesh_path in glob.glob(os.path.join(mesh_dir, "*.glb")):
    uid = os.path.basename(mesh_path).split(".")[0]
    uids.append(uid)

uids = sorted(uids)
random.shuffle(uids)

# ---------------------------------------
# STEP 2: compute split indices
# ---------------------------------------
n = len(uids)

n_train = int(0.6 * n)
n_val = int(0.2 * n)

train_uids = uids[:n_train]
val_uids = uids[n_train:n_train + n_val]
test_uids = uids[n_train + n_val:]

# ---------------------------------------
# STEP 3: build split dictionary
# ---------------------------------------
split_dict = {}

for uid in train_uids:
    split_dict[uid] = {
        "train": [0],
        "val": [],
        "test": []
    }

for uid in val_uids:
    split_dict[uid] = {
        "train": [],
        "val": [0],
        "test": []
    }

for uid in test_uids:
    split_dict[uid] = {
        "train": [],
        "val": [],
        "test": [0]
    }

# ---------------------------------------
# STEP 4: save
# ---------------------------------------
with open(os.path.join(cat_dir, "split.json"), "w") as f:
    json.dump(split_dict, f, indent=4)

print(f"Created split.json")
print(f"Train meshes: {len(train_uids)}")
print(f"Val meshes:   {len(val_uids)}")
print(f"Test meshes:  {len(test_uids)}")