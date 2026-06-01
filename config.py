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

    # Data Settings
    "data_root": "./dataloader/data",
    "category": "chair",
    "image_size": 518,
    "batch_size": 1,          # Kept at 2 to maximize 24GB VRAM ceiling safely
    "num_workers": 4,

    # Optimization Hyperparameters
    "epochs": 50,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,

    # Architecture Settings
    "lora_r": 16,
    "lora_alpha": 32,
    "model_backbone": "microsoft/TRELLIS-image-large"
}

WANDB_CONFIG = {
    "project": "sketch2mesh",
    "entity": "namoz-tum", 
    "run_name": "lora-sparse-rectified-flow-run"
}