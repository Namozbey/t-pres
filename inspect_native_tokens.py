import torch
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from config import TRAINING_CONFIG

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading native pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRAINING_CONFIG["model_backbone"])
    pipeline.to(device)
    
    dummy_image = Image.new("RGB", (512, 512), color=(255, 255, 255))
    
    print("\n====================================================")
    print("INSIGHT: STEP-BY-STEP NATIVE TOKEN FLOW")
    print("====================================================")
    
    with torch.no_grad():
        # Step A: Preprocess the raw image asset
        processed_image = pipeline.preprocess_image(dummy_image)
        print(f"1. Native preprocess_image() output type: {type(processed_image)}")
        
        # Step B & C: Pass the preprocessed image list directly to get_cond
        # This mirrors exactly how pipeline.run() handles it internally!
        final_cond = pipeline.get_cond([processed_image])
        print(f"\n2. Native get_cond() structure:")
        if isinstance(final_cond, dict):
            for k, v in final_cond.items():
                if isinstance(v, torch.Tensor):
                    print(f"   -> Key '{k}': Shape {v.shape}, Range [{v.min().item():.3f}, {v.max().item():.3f}]")