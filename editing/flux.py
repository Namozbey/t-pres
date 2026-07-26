import gc
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import Flux2KleinPipeline
from rembg import remove, new_session

gpu_session = new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# =====================================================================
# PYTORCH BACKWARD COMPATIBILITY HACK
# Diffusers tries to use `enable_gqa` which older PyTorch lacks.
# This patch intercepts the function, removes the unsupported argument,
# and mathematically replicates GQA so older PyTorch runs it perfectly.
# =====================================================================
if hasattr(F, "scaled_dot_product_attention"):
    _original_sdpa = F.scaled_dot_product_attention

    def _patched_sdpa(query, key, value, *args, **kwargs):
        # 1. Strip the unsupported argument
        kwargs.pop("enable_gqa", None)
        
        # 2. Mathematically expand Key/Value heads to match Query heads (GQA)
        if query.shape[1] != key.shape[1]:
            num_groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)
            
        # 3. Pass to native PyTorch SDPA
        return _original_sdpa(query, key, value, *args, **kwargs)

    # Overwrite the PyTorch function with our patched version
    F.scaled_dot_product_attention = _patched_sdpa
# =====================================================================

class Flux2DevRequest():
    prompt: str
    image: Image.Image
    num_inference_steps: int = 4
    guidance_scale: float = 4.0
    generator_seed: int = 123
    number_of_images: int = 1

class Flux2Wrapper():
    # def __init__(self):
    #     super().__init__(min_vram=23)

    def load(self):
        model = "black-forest-labs/FLUX.2-klein-4B"

        # Load in float16 for older GPU compatibility
        self.pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.float16)
        self.pipe.to("cuda")
        
        # EXPLICITLY lock out xformers so Diffusers stops trying to auto-route to it.
        # This will force Flux to fall back to its native attention, 
        # which will safely hit our PyTorch GQA patch at the top of the file!
        self.pipe.disable_xformers_memory_efficient_attention()

    def _cleanup_additional_resources(self):
        """
        Hook method for child classes to clean up additional resources.
        Override this method if your pipeline has additional attributes to clean up
        (e.g., compel, tokenizers, etc.).
        """
        if hasattr(self, "compel") and self.compel is not None:
            self.compel = None

    def unload(self):
        """
        Unload the pipeline from memory.
        Handles common cleanup: deletes the pipe, clears CUDA cache, and runs garbage collection.
        Child classes can override _cleanup_additional_resources() for additional cleanup.
        """
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
        self._cleanup_additional_resources()
        torch.cuda.empty_cache()
        gc.collect()

    def _create_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)
        return generator

    def generate_images(
        self,
        request: Flux2DevRequest,
        progress_callback=None,
        clear_bg=True
    ):
        image = request.image
        generator = self._create_generator(request.generator_seed)
        res_images = []
        # Run inference for each chunk in the batch plan
        result = self.pipe(
            prompt=request.prompt,
            image=image,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.number_of_images,
            generator=generator,
            callback_on_step_end=progress_callback,
        )
        res_images += result.images
        if clear_bg:
            for i, img in enumerate(res_images):
                res_images[i] = remove(img, session=gpu_session)

        return res_images