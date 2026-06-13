#architecture.py
import torch
import torch.nn as nn
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from peft import LoraConfig, get_peft_model
from config import TRAINING_CONFIG
import torch.nn.functional as F

# =====================================================================
# THE STRUCTURAL ENGINE WRAPPER
# =====================================================================

class TrellisSketchTrainingArchitecture(nn.Module):
    """
    A unified wrapper that manages the TRELLIS architecture for training.
    Handles the frozen conditioning extraction and the 
    LoRA-injected Sparse Structure Flow model.
    """
    def __init__(self, pipeline: TrellisImageTo3DPipeline):
        super().__init__()
        
        # -------------------------------------------------------------
        # STEP 1: EXTRACT & FREEZE BASE BACKBONE
        # -------------------------------------------------------------
        print("Extracting 'sparse_structure_flow_model'...")
        base_dit = pipeline.models['sparse_structure_flow_model']
        # Freeze 100% of Microsoft's original weights
        base_dit.requires_grad_(False)
        # Set to evaluation mode
        base_dit.eval()
        
        # -------------------------------------------------------------
        # STEP 2: INJECT LORA ADAPTERS
        # -------------------------------------------------------------
        print("Configuring and applying PEFT LoRA wrapper...")
        lora_config = LoraConfig(
            r=TRAINING_CONFIG["lora_r"],                # Low-rank dimension for 24GB VRAM target
            lora_alpha=TRAINING_CONFIG["lora_alpha"],       # Scaling factor
            target_modules=TRAINING_CONFIG["target_modules"], # Targeted attention blocks
            lora_dropout=TRAINING_CONFIG["lora_dropout"],
            bias="none"
        )
        print("Applying the PEFT wrapper to the frozen model...")
        self.lora_dit = get_peft_model(base_dit, lora_config)
        
        # -------------------------------------------------------------
        # STEP 3: EXTRACT & FREEZE IMAGE CONDITIONING ENCODER
        # -------------------------------------------------------------\
        self.register_buffer("slat_mean", torch.tensor(pipeline.slat_normalization['mean']).view(1, 8, 1, 1, 1))
        self.register_buffer("slat_std", torch.tensor(pipeline.slat_normalization['std']).view(1, 8, 1, 1, 1))

        print("Extracting and locking 'image_cond_model' for sketch features...")
        self.cond_encoder = pipeline.models['image_cond_model']
        # Freeze 100% of Microsoft's original weights
        self.cond_encoder.requires_grad_(False)
        # Set to evaluation mode
        self.cond_encoder.eval()

    def get_sketch_tokens(self, sketch_tensor):
        """
        Extracts DINOv2 visual tokens from batched training tensors 
        and pads them to exactly 1374 length to match native TRELLIS shapes.
        """
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=sketch_tensor.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=sketch_tensor.device
        ).view(1, 3, 1, 1)

        sketch_tensor = (sketch_tensor - mean) / std

        self.cond_encoder.eval()

        with torch.no_grad():
            features = self.cond_encoder(
                sketch_tensor,
                is_training=True
            )['x_prenorm']

            cond_tokens = F.layer_norm(
                features,
                features.shape[-1:]
            )

        return cond_tokens

    def forward(self, x_t, t, sketch_tensor):
        """
        Unified forward pass to be invoked inside the training loop.
        """
        # 1. Process sketches into tokens cleanly (VRAM Safe)
        cond_tokens = self.get_sketch_tokens(sketch_tensor)

        if not hasattr(self, "_printed_shape"):
            print("cond shape:", cond_tokens.shape)
            self._printed_shape = True
        
        # 2. Run the Flow Matching prediction step through LoRA layers
        predicted_velocity = self.lora_dit(x_t, t, cond=cond_tokens)
        
        return predicted_velocity

# =====================================================================
# PIPELINE CONFIGURATION BUILDER
# =====================================================================

def setup_trainable_structure_pipeline():
    """
    Initializes the master TRELLIS pipeline, extracts the structural 
    and conditioning components, and constructs the unified training wrapper.
    """

    print("Initializing TRELLIS Master Pipeline...")
    # Pulls the heavy structural and conditioning assets from weights
    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRAINING_CONFIG["model_backbone"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nConstructing Training Architecture Wrapper...")
    model_wrapper = TrellisSketchTrainingArchitecture(pipeline).to(device)

    
    # -------------------------------------------------------------
    # GRADIENT CHECKPOINTING ACTIVATION (CRITICAL FOR 24GB VRAM)
    # -------------------------------------------------------------
    print("Activating Gradient Checkpointing via PEFT helper...")
    
    # 1. First enable input gradients on the wrapper module 
    # (Crucial for frozen backbones with trainable adapters!)
    if hasattr(model_wrapper.lora_dit, "enable_input_require_grads"):
        model_wrapper.lora_dit.enable_input_require_grads()
        
    # 2. Use PEFT's native helper to enable checkpointing safely
    if hasattr(model_wrapper.lora_dit, "gradient_checkpointing_enable"):
        model_wrapper.lora_dit.gradient_checkpointing_enable()
    elif hasattr(model_wrapper.lora_dit.base_model.model, "gradient_checkpointing_enable"):
        # Fallback to the underlying model if PEFT top-level is structured differently
        model_wrapper.lora_dit.base_model.model.gradient_checkpointing_enable()

    # -------------------------------------------------------------
    # VERIFICATION AUDIT
    # -------------------------------------------------------------
    print("\n=== SYSTEM INTEGRITY AUDIT ===")
    model_wrapper.lora_dit.print_trainable_parameters()
    
    # Quick sanity check on the conditioning encoder state
    enc_trainable = any(p.requires_grad for p in model_wrapper.cond_encoder.parameters())
    print(f"Conditioning Encoder Trainable: {enc_trainable} (Expected: False)")
    
    print("================================\n")
    print("Trainable Structure Pipeline initialized")
    return model_wrapper, pipeline