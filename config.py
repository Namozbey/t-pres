# config.py
import torch

# =====================================================================
# SYSTEM HYPERPARAMETERS CONFIGURATION
# =====================================================================

TRAINING_CONFIG = {
    # Hardware & Logistics
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "checkpoints_dir": "./checkpoints",
    "save_every_n_epochs": 10,
    "hardcode_timestep": False,

    # Data Settings
    "data_root": "./dataloader/data",
    "category": "chair",
    "image_size": 518,
    "batch_size": 1,
    "num_workers": 1,

    # Optimization Hyperparameters
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,

    # Architecture Settings
    "lora_r": 128,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "target_modules": "all-linear",
    # [
    #     "to_q", "to_kv", "to_qkv", "to_out",  # Attention
    #     "fc1", "fc2", "proj", "mlp.fc1", "mlp.fc2" # Feed-Forward / MLP
    # ], 
    # ["to_q", "to_kv", "to_qkv", "to_out"],
    "model_backbone": "E:/Nov1/AI/TRELLIS/TRELLIS-image-large"
}

WANDB_CONFIG = {
    "project": "sketch2mesh",
    "entity": "namoz-tum",     # Replace with your 2-person W&B team space name
    "run_name": "lora-sparse-rectified-flow-run"
}