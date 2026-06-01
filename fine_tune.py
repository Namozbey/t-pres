#fine_tune.py
import os

# # This single line fixes BOTH the dense and sparse attention modules!
# os.environ['ATTN_BACKEND'] = 'xformers' 
# os.environ['SPCONV_ALGO'] = 'native'

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

def train_epoch(model_wrapper, dataloader, optimizer, device, current_epoch):
    """
    Runs a single optimization pass.
    """
    # Ensure active training modules are in train mode, frozen components stay in eval
    model_wrapper.lora_dit.train()
    model_wrapper.cond_encoder.eval()
    
    epoch_loss = 0.0
    num_batches = len(dataloader)

    
    for batch_idx, batch in enumerate(dataloader):

        optimizer.zero_grad()

        # -------------------------------------------------------------
        # 1. LOAD DATA
        # -------------------------------------------------------------
        # Ground-truth structured latent
        x_0_dense = batch['ss_latent'].to(device)

        # Conditioning sketch
        sketches = batch['sketch'].to(device)

        batch_size = sketches.shape[0]

        # -------------------------------------------------------------
        # 2. SAMPLE TIMESTEPS
        # -------------------------------------------------------------
        t_per_batch_item = torch.rand(batch_size, device=device)

        # Broadcast for volumetric latent operations
        t_broadcast = t_per_batch_item.view(
            batch_size, 1, 1, 1, 1
        )

        # -------------------------------------------------------------
        # 3. SAMPLE NOISE
        # -------------------------------------------------------------
        epsilon = torch.randn_like(x_0_dense)

        # -------------------------------------------------------------
        # 4. CONDITIONAL FLOW MATCHING INTERPOLATION
        # -------------------------------------------------------------
        # x(t) = (1 - t)x0 + t*epsilon
        x_t_dense = (
            (1.0 - t_broadcast) * x_0_dense +
            t_broadcast * epsilon
        )

        # Target vector field
        # v = epsilon - x0
        target_velocity_dense = epsilon - x_0_dense

        # -------------------------------------------------------------
        # 5. MODEL PREDICTION
        # -------------------------------------------------------------
        predicted_velocity = model_wrapper(
            x_t_dense,
            t_per_batch_item,
            sketches
        )

        # -------------------------------------------------------------
        # 6. FLOW MATCHING LOSS
        # -------------------------------------------------------------
        loss = F.mse_loss(
            predicted_velocity,
            target_velocity_dense
        )

        loss.backward()
        optimizer.step()

        # -------------------------------------------------------------
        # LOG STEP-LEVEL LOSS TO W&B
        # -------------------------------------------------------------
        global_step = (current_epoch - 1) * num_batches + batch_idx
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/epoch": current_epoch,
            "train/global_step": global_step
        })
        
        print(f"  → Batch [{batch_idx+1}/{num_batches}] | Loss: {loss.item():.5f}")
        epoch_loss += loss.item()
        
    return epoch_loss / num_batches


# =====================================================================
# SYSTEM EXECUTION PIPELINE
# =====================================================================

def main_train_pipeline():
    """
    Main execution wrapper entirely controlled by config.py
    """
    device = TRAINING_CONFIG["device"]
    print(f"System Execution Backend: {device.upper()}")

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
    print(f"\nStarting training loop for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        # FIXED: Passed current_epoch into function parameters explicitly
        avg_loss = train_epoch(trainable_architecture, dataloader, optimizer, device, current_epoch=epoch)
        
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
        
        # Save checkpoints safely based on configuration parameters
        if epoch % TRAINING_CONFIG["save_every_n_epochs"] == 0 or epoch == epochs:
            os.makedirs(TRAINING_CONFIG["checkpoints_dir"], exist_ok=True)
            checkpoint_path = f"{TRAINING_CONFIG['checkpoints_dir']}/trellis_lora_epoch_{epoch}"
            print(f"--> Checkpoint: Saving trained adapters to {checkpoint_path}...")
            trainable_architecture.lora_dit.save_pretrained(checkpoint_path)

    wandb.finish()
    print("\nTraining execution completed successfully.")

if __name__ == "__main__":
    main_train_pipeline()