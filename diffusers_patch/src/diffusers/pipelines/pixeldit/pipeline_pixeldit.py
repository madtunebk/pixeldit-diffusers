# Copyright 2024 NVIDIA and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from diffusers.models import ModelMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

try:
    from .pipeline_output import PixelDiTPipelineOutput
except ImportError:
    from pipeline_output import PixelDiTPipelineOutput


logger = logging.get_logger(__name__)

# chi_prompt: the instruction prefix prepended to every user prompt during training.
# Gemma was trained to "enhance" prompts through this prefix — omitting it degrades output.
_CHI_PROMPT = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
    '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
    'Here are examples of how to transform or refine prompts:',
    '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
    '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
    'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
    'User Prompt: ',
])

_TXT_MAX_LENGTH = 300
_SELECT_IDX = [0] + list(range(-(_TXT_MAX_LENGTH - 1), 0))  # BOS + last 299 tokens

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixelDiTPipeline

        >>> pipe = PixelDiTPipeline.from_pretrained(
        ...     "madtune/pixeldit-diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "a white horse galloping through a meadow at sunset, cinematic lighting"
        >>> image = pipe(prompt).images[0]
        >>> image.save("pixeldit_out.png")
        ```
"""


class PixelDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixelDiT.

    PixelDiT is a pixel-space diffusion transformer — it generates images directly without a VAE,
    using Gemma-2-2B as the text encoder with a chi_prompt instruction prefix.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a device, etc.).

    Args:
        transformer ([`PixelDiTModel`]):
            Conditional transformer to denoise the image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler to denoise the image in combination with `transformer`.
        text_encoder ([`~transformers.AutoModelForCausalLM`]):
            Frozen Gemma-2-2B language model (decoder only). The chi_prompt prefix is applied internally.
        tokenizer ([`~transformers.AutoTokenizer`]):
            Tokenizer for the Gemma text encoder.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        transformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder,
        tokenizer,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self._num_chi_tokens = len(self.tokenizer.encode(_CHI_PROMPT))

    # ------------------------------------------------------------------
    # LoRA API
    # ------------------------------------------------------------------

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict,
        adapter_name: str = "default",
        **kwargs,
    ):
        """
        Load LoRA weights into the transformer.

        Accepts:
        - A PEFT adapter directory (must contain adapter_config.json).
        - A path to a single .safetensors / .pt / .bin file.
        - A pre-loaded state dict.

        Keys may optionally carry a ``transformer.`` prefix — it will be stripped.
        Kohya-style ``.alpha`` keys are extracted as ``network_alphas``.
        """
        print(f"[LoRA] Loading adapter '{adapter_name}'...")

        # --- PEFT adapter directory (saved by train_lora.py via model.save_pretrained) ---
        # These use adapter_model.safetensors + adapter_config.json (PEFT format).
        # diffusers' load_lora_adapter expects pytorch_lora_weights.safetensors, so
        # we use PEFT's native API here instead.
        if (
            isinstance(pretrained_model_name_or_path_or_dict, str)
            and os.path.isdir(pretrained_model_name_or_path_or_dict)
            and os.path.exists(
                os.path.join(pretrained_model_name_or_path_or_dict, "adapter_config.json")
            )
        ):
            from peft import PeftModel
            lora_dir = pretrained_model_name_or_path_or_dict
            if isinstance(self.transformer, PeftModel):
                # already wrapped — add another adapter
                self.transformer.load_adapter(lora_dir, adapter_name=adapter_name)
            else:
                # first LoRA — wrap the transformer in a PeftModel
                self.transformer = PeftModel.from_pretrained(
                    self.transformer, lora_dir, adapter_name=adapter_name, is_trainable=False
                )
            print(f"[LoRA] Loaded PEFT adapter '{adapter_name}'.")
            return

        # --- state dict path or in-memory dict ---
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = dict(pretrained_model_name_or_path_or_dict)
        else:
            path = str(pretrained_model_name_or_path_or_dict)
            if os.path.isfile(path):
                weights_file = path
            else:
                import glob
                candidates = (
                    glob.glob(os.path.join(path, "*.safetensors"))
                    + glob.glob(os.path.join(path, "*.bin"))
                    + glob.glob(os.path.join(path, "*.pt"))
                )
                if not candidates:
                    raise FileNotFoundError(f"[LoRA] No weights file found in {path}")
                weights_file = candidates[0]

            if weights_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(weights_file)
            else:
                state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

        # strip component prefix
        if any(k.startswith("transformer.") for k in state_dict):
            state_dict = {
                k[len("transformer."):]: v
                for k, v in state_dict.items()
                if k.startswith("transformer.")
            }

        # extract Kohya-style network_alphas (.alpha keys)
        network_alphas: dict = {}
        clean: dict = {}
        for k, v in state_dict.items():
            if k.endswith(".alpha"):
                network_alphas[k[: -len(".alpha")]] = float(v)
            else:
                clean[k] = v

        self.transformer.load_lora_adapter(
            clean,
            adapter_name=adapter_name,
            network_alphas=network_alphas if network_alphas else None,
            **kwargs,
        )
        print(
            f"[LoRA] Loaded adapter '{adapter_name}' "
            f"({len(clean)} keys, {len(network_alphas)} alphas)."
        )

    def save_lora_weights(
        self,
        save_directory: str,
        adapter_name: str = "default",
        safe_serialization: bool = True,
        upcast_before_saving: bool = False,
    ):
        """Save LoRA adapter weights to disk (PEFT format)."""
        self.transformer.save_lora_adapter(
            save_directory,
            adapter_name=adapter_name,
            safe_serialization=safe_serialization,
            upcast_before_saving=upcast_before_saving,
        )
        print(f"[LoRA] Saved adapter '{adapter_name}' to {save_directory}")

    def unload_lora_weights(self):
        """Remove all LoRA adapters and restore the base transformer weights."""
        from peft import PeftModel
        if isinstance(self.transformer, PeftModel):
            self.transformer = self.transformer.merge_and_unload()
            print("[LoRA] LoRA merged and unloaded.")
        elif hasattr(self.transformer, "unload_lora"):
            self.transformer.unload_lora()
            print("[LoRA] LoRA unloaded.")

    def set_adapters(self, adapter_names, adapter_weights=None):
        """Activate one or more named adapters with optional per-adapter scales."""
        self.transformer.set_adapters(
            adapter_names,
            weights=adapter_weights,
        )

    def disable_lora(self):
        self.transformer.disable_lora()

    def enable_lora(self):
        self.transformer.enable_lora()

    def fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False, adapter_names=None, **kwargs):
        """Bake LoRA weights permanently into the base transformer weights."""
        self.transformer.fuse_lora(
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
        )

    def unfuse_lora(self, **kwargs):
        """Revert a previous fuse_lora() call."""
        self.transformer.unfuse_lora()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pipeline. The transformer is loaded from a PixelDiTModel checkpoint.
        Text encoder and tokenizer are loaded from Gemma-2-2B.
        """
        import diffusers
        from .modeling_pixeldit_hf import PixelDiTModel
        # model_index.json references ["diffusers", "PixelDiTModel"] — inject at runtime
        if not hasattr(diffusers, "PixelDiTModel"):
            diffusers.PixelDiTModel = PixelDiTModel
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        lora_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt(s) using Gemma with chi_prompt prefix.
        Returns (prompt_embeds, negative_prompt_embeds), each [B, 300, 2304].

        lora_scale: if set and a LoRA is loaded on the text encoder, scales its
        contribution during encoding then restores the original scale.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # scale text-encoder LoRA if requested
        if lora_scale is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder, lora_scale)

        try:
            if hasattr(self.text_encoder, "encode"):
                prompt_embeds = self.text_encoder.encode(prompt).to(device=device, dtype=dtype)

                if do_classifier_free_guidance:
                    if negative_prompt is None:
                        negative_prompt_embeds = self.text_encoder.encode_null(batch_size)
                    else:
                        if isinstance(negative_prompt, str):
                            negative_prompt = [negative_prompt] * batch_size
                        negative_prompt_embeds = self.text_encoder.encode(negative_prompt)
                    negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                else:
                    negative_prompt_embeds = None

                return prompt_embeds, negative_prompt_embeds

            # --- positive embeds ---
            texts_full = [_CHI_PROMPT + p for p in prompt]
            max_len = self._num_chi_tokens + _TXT_MAX_LENGTH - 2
            tok = self.tokenizer(
                texts_full,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                emb = self.text_encoder(
                    input_ids=tok.input_ids,
                    attention_mask=tok.attention_mask,
                ).last_hidden_state
            prompt_embeds = emb[:, _SELECT_IDX, :].to(dtype)

            # --- negative embeds ---
            if do_classifier_free_guidance:
                if negative_prompt is None:
                    negative_prompt = [""] * batch_size
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * batch_size

                neg_tok = self.tokenizer(
                    negative_prompt,
                    max_length=_TXT_MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    neg_emb = self.text_encoder(
                        input_ids=neg_tok.input_ids,
                        attention_mask=neg_tok.attention_mask,
                    ).last_hidden_state
                negative_prompt_embeds = neg_emb.to(dtype)
            else:
                negative_prompt_embeds = None

            return prompt_embeds, negative_prompt_embeds

        finally:
            if lora_scale is not None and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

    def check_inputs(self, prompt, height, width, negative_prompt=None):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height}×{width}")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ) -> Union[PixelDiTPipelineOutput, Tuple]:
        """
        Generate images from text prompts.

        Args:
            prompt (`str` or `List[str]`): Prompt(s) to guide image generation.
            negative_prompt (`str` or `List[str]`, *optional*): Negative prompt(s).
            height (`int`, *optional*, defaults to 512): Output image height. Must be divisible by 16.
            width (`int`, *optional*, defaults to 512): Output image width. Must be divisible by 16.
            num_inference_steps (`int`, *optional*, defaults to 20): Number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 3.5): CFG guidance scale.
            generator (`torch.Generator`, *optional*): RNG for reproducibility.
            output_type (`str`, *optional*, defaults to `"pil"`): `"pil"` or `"np"`.
            return_dict (`bool`, *optional*, defaults to `True`): Return `PixelDiTPipelineOutput` or plain tuple.
            callback_on_step_end (`Callable`, *optional*): Called at end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*): Tensor names passed to callback.

        Examples:

        %s

        Returns:
            [`PixelDiTPipelineOutput`] or `tuple`.
        """
        # 0. setup
        device = self._execution_device
        dtype  = self.transformer.dtype
        self._guidance_scale = guidance_scale
        lora_scale = (cross_attention_kwargs or {}).get("scale", None)

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # 1. validate
        self.check_inputs(prompt, height, width, negative_prompt)

        # 2. encode text
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
        )

        # 3. prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 4. prepare noise (pixel-space — no VAE encoding needed)
        latents = randn_tensor(
            (batch_size, 3, height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # 5. denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand for CFG
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                latent_model_input = latents
                embeds = prompt_embeds

            # FlowMatchEulerDiscreteScheduler already returns t in [0, 1000]
            t_input = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(latent_model_input, t_input, embeds)

            # CFG
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # scheduler step
            if hasattr(self.scheduler, "scale_model_input"):
                latents = self.scheduler.step(noise_pred, t, self.scheduler.scale_model_input(latents, t), return_dict=False)[0]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                cb_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    cb_kwargs[k] = locals()[k]
                callback_on_step_end(i, t, cb_kwargs)

        # 6. decode (pixel-space — just clamp and normalize)
        image = (latents.clamp(-1, 1) + 1) / 2
        image = (image * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        if output_type == "pil":
            from PIL import Image
            image = [Image.fromarray(img) for img in image]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return PixelDiTPipelineOutput(images=image)
