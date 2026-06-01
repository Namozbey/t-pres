#architecture.py
import torch
import torch.nn as nn
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from peft import LoraConfig, get_peft_model

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
            r=16,                # Low-rank dimension for 24GB VRAM target
            lora_alpha=32,       # Scaling factor
            target_modules=["to_q", "to_kv", "to_qkv", "to_out"], # Targeted attention blocks
            lora_dropout=0.05,
            bias="none"
        )
        print("Applying the PEFT wrapper to the frozen model...")
        self.lora_dit = get_peft_model(base_dit, lora_config)
        
        # -------------------------------------------------------------
        # STEP 3: EXTRACT & FREEZE IMAGE CONDITIONING ENCODER
        # -------------------------------------------------------------
        print("Extracting and locking 'image_cond_model' for sketch features...")
        self.cond_encoder = pipeline.models['image_cond_model']
        # Freeze 100% of Microsoft's original weights
        self.cond_encoder.requires_grad_(False)
        # Set to evaluation mode
        self.cond_encoder.eval()

    def get_sketch_tokens(self, sketch_tensor):
        """
        Feature Extraction: Converts raw sketch pixels into 
        mathematical conditioning tokens without tracking memory gradients.
        """
        with torch.no_grad():
            # Pass preprocessed sketch pixels through the frozen encoder
            cond_output = self.cond_encoder(sketch_tensor)
            
            # Handle potential dictionary formatting vs raw tensor output
            if isinstance(cond_output, dict):
                cond_tokens = cond_output.get("cond_tokens")
            else:
                cond_tokens = cond_output

        # --- DIAGNOSTIC PRINT (Temporary: Helps us see what cond_encoder actually outputs) ---
        # print(f"[DEBUG] Raw encoder output shape: {cond_tokens.shape}")

        # Ensure we have a 3D tensor: [Batch, Sequence_Length, Feature_Channels]
        if cond_tokens.ndim == 2:
            # If [Batch, Channels], unsqueeze to [Batch, 1, Channels]
            cond_tokens = cond_tokens.unsqueeze(1)
            
        # Target sequence length that TRELLIS cross-attention is hunting for
        target_seq_len = 1024
        current_seq_len = cond_tokens.shape[1]
        
        if current_seq_len != target_seq_len:
            if current_seq_len == 1:
                # If sequence length is 1, repeat it along the sequence dimension cleanly
                cond_tokens = cond_tokens.repeat(1, target_seq_len, 1)
            else:
                # If it's another arbitrary sequence length, interpolate or pad it
                # Permute to [Batch, Channels, Seq] for 1D interpolation
                cond_tokens = cond_tokens.permute(0, 2, 1)
                cond_tokens = torch.nn.functional.interpolate(
                    cond_tokens, size=target_seq_len, mode='linear', align_corners=False
                )
                cond_tokens = cond_tokens.permute(0, 2, 1) # Permute back to [Batch, Seq, Channels]

        # --- THE INSURANCE POLICY ---
        # If the channel dimension doesn't match what the DiT cross-attention linear layer expects,
        # we need to ensure it projects correctly. If it still crashes, we will see the exact shape here:
        # print(f"[DEBUG] Final processed conditioning shape entering DiT: {cond_tokens.shape}")

        return cond_tokens

    def forward(self, x_t, t, sketch_tensor):
        """
        Unified forward pass to be invoked inside the training loop.
        """
        # 1. Process sketches into tokens cleanly (VRAM Safe)
        cond_tokens = self.get_sketch_tokens(sketch_tensor)
        
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
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nConstructing Training Architecture Wrapper...")
    model_wrapper = TrellisSketchTrainingArchitecture(pipeline).to(device)

    
    # -------------------------------------------------------------
    # GRADIENT CHECKPOINTING ACTIVATION (CRITICAL FOR 24GB VRAM)
    # -------------------------------------------------------------
    print("Activating Gradient Checkpointing on base backbone...")
    # This tells PyTorch to discard intermediate activations and recalculate them 
    # during the backward pass, saving massive amounts of VRAM.
    if hasattr(model_wrapper.lora_dit.base_model.model, "gradient_checkpointing_enable"):
        model_wrapper.lora_dit.base_model.model.gradient_checkpointing_enable()
    else:
        # Fallback if TRELLIS uses a custom flag name
        print("WARNING: Ensure gradient checkpointing is manually set in your TRELLIS DiT config!")

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