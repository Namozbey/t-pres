#fine_tune.py
import os
os.environ['ATTN_BACKEND'] = 'xformers' 
os.environ['SPCONV_ALGO'] = 'native'

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from architecture import setup_trainable_structure_pipeline
from dataloader.dataset import SketchMeshDataset
from dataloader.utils import sparse_collate_fn
from config import TRAINING_CONFIG, WANDB_CONFIG

# =====================================================================
# THE UNIFIED PRODUCTION TRAINING ENGINE
# =====================================================================

def train_epoch(model_wrapper, dataloader, optimizer, device, current_epoch, accumulation_steps):
    """
    Runs a single optimization pass matching native TRELLIS math precisely.
    """
    model_wrapper.lora_dit.train()
    
    epoch_loss = 0.0
    num_batches = len(dataloader)
    sigma_min = getattr(model_wrapper, "sigma_min", 1e-5)

    # 🛑 1. Clear gradients BEFORE the loop starts
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):

        # ==========================================
        # 1. LOAD & NORMALIZE DATA
        # ==========================================
        raw_ss_latent = batch['ss_latent'].to(device)
        cond_tokens = batch["cond_tokens"].to(device)
        
        print(f"cond_token_dim:{cond_tokens.shape}")
        # TRELLIS math tracking: x_0 is clean data, noise is epsilon
        x_0 = (raw_ss_latent - model_wrapper.slat_mean) / model_wrapper.slat_std
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0)
        print("raw_ss shape:", raw_ss_latent.shape)
        print("norm shape:", model_wrapper.slat_mean.shape)

        print(
            "raw_ss mean/std:",
            raw_ss_latent.mean().item(),
            raw_ss_latent.std().item()
        )

        print(
            "x0 mean/std:",
            x_0.mean().item(),
            x_0.std().item()
        )
        # ==========================================
        # 2. ALIGNED TIMESTEP SCHEDULER
        # ==========================================
        if TRAINING_CONFIG.get("hardcode_timestep", False):
            t = torch.ones(batch_size, device=device) * 0.5
        else:
            # Native TRELLIS Logit-Normal schedule setup
            mean, std = 0.0, 1.0
            t = torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)

        t_broadcast = t.view(batch_size, 1, 1, 1, 1)

        # ==========================================
        # 3. NATIVE TRELLIS TRAJECTORY (diffuse)
        # ==========================================
        x_t = (1.0 - t_broadcast) * x_0 + (sigma_min + (1.0 - sigma_min) * t_broadcast) * noise
        
        # ==========================================
        # 4. PREDICTION
        # ==========================================

        predicted_velocity = model_wrapper(
            x_t,
            t * 1000.0,
            cond_tokens
        )

        print(
            "pred abs mean:",
            predicted_velocity.abs().mean().item()
        )
        
        # ==========================================
        # 5. ALIGNED LOSS TARGET & TIMESTEP WEIGHTING
        # ==========================================
        target_velocity = (1.0 - sigma_min) * noise - x_0
        
        # 🛑 2. Calculate raw unreduced loss
        raw_loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        
        # 🛑 3. Apply Timestep Weighting (Prioritize t=0 structural formation)
        weight = 1 #torch.exp(-3.0 * t_broadcast)
        weighted_loss = (raw_loss * weight).mean()

        print(
            "target abs mean:",
            target_velocity.abs().mean().item()
        )

        # ==========================================
        # 6. GRADIENT ACCUMULATION BACKWARD STEP
        # ==========================================
        # 🛑 4. Divide the loss by the number of accumulation steps
        loss = weighted_loss / accumulation_steps
        loss.backward()

        total_grad = 0.0
        for name, p in model_wrapper.lora_dit.named_parameters():
            if p.requires_grad and p.grad is not None:
                total_grad += p.grad.abs().mean().item()

        print("total grad:", total_grad)

        # 🛑 5. Only step the optimizer after N batches (or on the final batch)
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
            torch.nn.utils.clip_grad_norm_(model_wrapper.lora_dit.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad() # Clear gradients for the next accumulation cycle

        # ==========================================
        # 7. LOGGING
        # ==========================================
        global_step = (current_epoch - 1) * num_batches + batch_idx
        
        # 🛑 6. Multiply the printed/logged loss back by accumulation_steps so it reads normally
        display_loss = loss.item() * accumulation_steps
        
        wandb.log({
            "train/batch_loss": display_loss,
            "train/epoch": current_epoch,
            "train/global_step": global_step
        })
        
        print(f"  → Batch [{batch_idx+1}/{num_batches}] | Loss: {display_loss:.5f}")
        epoch_loss += display_loss
        
    return epoch_loss / num_batches


# =====================================================================
# SYSTEM EXECUTION PIPELINE
# =====================================================================

def main_train_pipeline():
    """
    Main execution wrapper entirely controlled by config.py
    """
    device = TRAINING_CONFIG["device"]
    # 🛑 Grab accumulation steps from config, default to 4 if not set
    accumulation_steps = TRAINING_CONFIG.get("accumulation_steps", 4)
    print(f"System Execution Backend: {device.upper()}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")

    # 1. INITIALIZE MASTER W&B RUN VIA CONFIG
    wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        name=WANDB_CONFIG["run_name"],
        config=TRAINING_CONFIG  # Passes all training metrics directly to dashboard tracking
    )
    
    # 2. Setup the architecture pipeline
    trainable_architecture, master_pipeline = setup_trainable_structure_pipeline()

    print("Instantiating AdamW Optimizer...")
    optimizer = optim.AdamW(
        trainable_architecture.lora_dit.parameters(), 
        lr=TRAINING_CONFIG["learning_rate"], 
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=TRAINING_CONFIG["epochs"], 
    #     eta_min=5e-7
    # )
    
    print("\n--- Constructing Active Production Data Layer ---")
    dataset = SketchMeshDataset(
        root_dir=TRAINING_CONFIG["data_root"],
        category=TRAINING_CONFIG["category"],
        image_size=TRAINING_CONFIG["image_size"]
    ) 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,  
        num_workers=TRAINING_CONFIG["num_workers"],
        collate_fn=sparse_collate_fn,
        drop_last=True  
    )
    
    epochs = TRAINING_CONFIG["epochs"]
    epoch_losses = []
    print(f"\nStarting training loop for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        # 🛑 Pass accumulation_steps into train_epoch
        avg_loss = train_epoch(trainable_architecture, dataloader, optimizer, device, current_epoch=epoch, accumulation_steps=accumulation_steps)
        epoch_losses.append(avg_loss)

        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]

        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch  }) # "train/learning_rate": current_lr
        
        # Save checkpoints safely based on configuration parameters
        if epoch % TRAINING_CONFIG["save_every_n_epochs"] == 0 or epoch == epochs:
            os.makedirs(TRAINING_CONFIG["checkpoints_dir"], exist_ok=True)
            checkpoint_path = f"{TRAINING_CONFIG['checkpoints_dir']}/trellis_lora_epoch_{epoch}"
            print(f"--> Checkpoint: Saving trained adapters to {checkpoint_path}...")
            trainable_architecture.lora_dit.save_pretrained(checkpoint_path)

    wandb.log({"avg_loss": torch.tensor(epoch_losses, dtype=torch.float).mean().item()})
    wandb.finish()
    print("\nTraining execution completed successfully.")

if __name__ == "__main__":
    main_train_pipeline()