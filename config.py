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
    "val_fixed_timestep": False,

    # Data Settings
    "data_root": "./dataloader/data",
    "category": "chair",
    "image_size": 518,
    "batch_size": 2,
    "num_workers": 2,
    "accumulation_steps": 5,

    # Optimization Hyperparameters
    "epochs": 500,
    "learning_rate": 5e-5,
    "weight_decay": 0.0, # 1e-4

    # Architecture Settings
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    # [
    #     "to_q", "to_kv", "to_qkv", "to_out",  # Attention
    #     "fc1", "fc2", "proj", "mlp.fc1", "mlp.fc2" # Feed-Forward / MLP
    # ], 
    # ["to_q", "to_kv", "to_qkv", "to_out"],
    "model_backbone": "E:/Nov1/AI/TRELLIS/TRELLIS-image-large"
    # "model_backbone": "microsoft/TRELLIS-image-large"
}

WANDB_CONFIG = {
    "project": "sketch2mesh",
    "entity": "namoz-tum",     # Replace with your 2-person W&B team space name
    "run_name": "lora-sparse-rectified-flow-run"
}