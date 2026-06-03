import torch
import numpy as np

_GEMMA_ID = "Efficient-Large-Model/gemma-2-2b-it"
_HF_REPO  = "madtune/pixeldit-diffusers"


class PixelDiTTextEncoderLoader:
    """Load Gemma text encoder — swap for any compatible model (GGUF, fine-tuned, etc.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": _GEMMA_ID}),
            }
        }

    RETURN_TYPES  = ("PIXELDIT_ENCODER",)
    RETURN_NAMES  = ("text_encoder",)
    FUNCTION      = "load"
    CATEGORY      = "PixelDiT"

    def load(self, model_id):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[PixelDiT] Loading text encoder: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            .get_decoder().eval()
        )
        return ({"tokenizer": tokenizer, "text_encoder": text_encoder},)


class PixelDiTModelLoader:
    """Load PixelDiT transformer only — no text encoder dependency."""

    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": _HF_REPO}),
            }
        }

    RETURN_TYPES  = ("PIXELDIT_MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load"
    CATEGORY      = "PixelDiT"

    def load(self, repo_id):
        if repo_id not in self._cache:
            from diffusers.pipelines.pixeldit import PixelDiTModel
            from diffusers import FlowMatchEulerDiscreteScheduler
            print(f"[PixelDiT] Loading transformer: {repo_id}")
            transformer = PixelDiTModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch.bfloat16
            ).eval()
            scheduler = FlowMatchEulerDiscreteScheduler(shift=4.0)
            self._cache[repo_id] = {"transformer": transformer, "scheduler": scheduler}
        return (self._cache[repo_id],)


class PixelDiTSampler:
    """Generate images — takes model + text encoder separately."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("PIXELDIT_MODEL",),
                "text_encoder":    ("PIXELDIT_ENCODER",),
                "prompt":          ("STRING",  {"multiline": True, "default": "a viking warrior at sunset"}),
                "negative_prompt": ("STRING",  {"multiline": True, "default": "blurry, flat, low quality, cartoon"}),
                "width":           ("INT",     {"default": 1024, "min": 512, "max": 1024, "step": 64}),
                "height":          ("INT",     {"default": 1024, "min": 512, "max": 1024, "step": 64}),
                "steps":           ("INT",     {"default": 50,   "min": 45,  "max": 150,  "step": 1}),
                "cfg":             ("FLOAT",   {"default": 7.5,  "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed":            ("INT",     {"default": 0,    "min": 0,   "max": 2**32-1}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "sample"
    CATEGORY      = "PixelDiT"

    def sample(self, model, text_encoder, prompt, negative_prompt, width, height, steps, cfg, seed):
        from diffusers.pipelines.pixeldit import PixelDiTPipeline

        pipe = PixelDiTPipeline(
            transformer=model["transformer"],
            scheduler=model["scheduler"],
            text_encoder=text_encoder["text_encoder"],
            tokenizer=text_encoder["tokenizer"],
        )
        pipe.enable_model_cpu_offload()

        out = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        img = np.array(out.images[0]).astype(np.float32) / 255.0
        return (torch.from_numpy(img).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "PixelDiTTextEncoderLoader": PixelDiTTextEncoderLoader,
    "PixelDiTModelLoader":       PixelDiTModelLoader,
    "PixelDiTSampler":           PixelDiTSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDiTTextEncoderLoader": "PixelDiT Text Encoder",
    "PixelDiTModelLoader":       "PixelDiT Model Loader",
    "PixelDiTSampler":           "PixelDiT Sampler",
}
