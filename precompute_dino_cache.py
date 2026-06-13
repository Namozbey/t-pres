# precompute_cond_cache.py
import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import torch
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from config import TRAINING_CONFIG

def get_custom_sketch_tokens(pipeline, sketch_image):
    processed_image = pipeline.preprocess_image(sketch_image)

    cond_encoder = pipeline.models['image_cond_model']
    cond_encoder.eval()

    with torch.no_grad():
        cond = pipeline.get_cond([processed_image])

    # sanity print
    print(cond.keys())
    print(cond["cond"].shape, cond["neg_cond"].shape)

    return {
        "cond": cond["cond"],
        "neg_cond": cond["neg_cond"]
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT: load ONLY ONCE
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        TRAINING_CONFIG["model_backbone"]
    )
    pipeline.to(device)

    # freeze everything
    for m in pipeline.models.values():
        if hasattr(m, "eval"):
            m.eval()

    root = TRAINING_CONFIG["data_root"]
    sketch_dir = os.path.join(root, TRAINING_CONFIG["category"], "sketches")
    cache_dir = os.path.join(root, TRAINING_CONFIG["category"], "sketch_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for file in os.listdir(sketch_dir):
        if not file.endswith(".png"):
            continue

        path = os.path.join(sketch_dir, file)
        img = Image.open(path).convert("RGB")

        cond_tokens = get_custom_sketch_tokens(pipeline, img)

        uid = os.path.splitext(file)[0]
        torch.save(cond_tokens, os.path.join(cache_dir, f"{uid}.pt"))

        print("cached:", uid, "cond_shape:", cond_tokens["cond"].shape)
        print("cached:", uid, "neg_cond_shape:", cond_tokens["neg_cond"].shape)


if __name__ == "__main__":
    main()