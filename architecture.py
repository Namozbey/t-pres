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

        self.cond_shape_ref = None
        self.cond_dtype_ref = torch.float32

    # def _validate_cond(self, cond_tokens):
    #     if self.cond_shape_ref is None:
    #         self.cond_shape_ref = torch.tensor(cond_tokens.shape)

    #     assert cond_tokens.shape == tuple(self.cond_shape_ref.tolist()), \
    #         f"Cond shape mismatch: {cond_tokens.shape}"

    #     assert cond_tokens.dtype == torch.float32, \
    #         f"Cond dtype must be float32, got {cond_tokens.dtype}"

    #     return cond_tokens

    def forward(self, x_t, t, cond_tokens):
        # cond_tokens = self._validate_cond(cond_tokens)
        return self.lora_dit(
            x_t,
            t,
            cond=cond_tokens
        )

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
    
    print("================================\n")
    print("Trainable Structure Pipeline initialized")
    return model_wrapper, pipeline