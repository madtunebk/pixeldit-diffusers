# Copyright 2024 NVIDIA and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from diffusers.utils.torch_utils import randn_tensor

try:
    from .pipeline_pixeldit import PixelDiTPipeline
    from .pipeline_output import PixelDiTPipelineOutput
except ImportError:
    from pipeline_pixeldit import PixelDiTPipeline
    from pipeline_output import PixelDiTPipelineOutput


def _to_pixel_tensor(image, width, height, device, dtype):
    """Convert a PIL Image or float tensor to [B, 3, H, W] in [-1, 1]."""
    if isinstance(image, Image.Image):
        image = image.convert("RGB").resize((width, height))
        image = np.array(image, dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).div(127.5).sub(1.0)
        image = image.unsqueeze(0)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = torch.from_numpy(image.astype(np.float32)).div(127.5).sub(1.0)
        else:
            image = torch.from_numpy(image.astype(np.float32)).mul(2.0).sub(1.0)
        if image.dim() == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.is_floating_point() and image.max() <= 1.0 + 1e-4:
            image = image.mul(2.0).sub(1.0)
    return image.to(device=device, dtype=dtype)


class PixelDiTImg2ImgPipeline(PixelDiTPipeline):
    """
    Img2img pipeline for PixelDiT.

    Inherits everything from :class:`PixelDiTPipeline` — same model, same text encoder,
    same LoRA API, same schedulers.

    Pass an input image and a ``strength`` value to control how much the image is modified:
    ``strength=1.0`` equals pure text-to-image generation; ``strength=0.1`` barely changes
    the input. Because PixelDiT is a pixel-space model (no VAE), noise is injected directly
    on the pixel tensor using the flow-matching formula:
    ``x_t = (1 − σ) · image + σ · noise``

    Note: PixelDiT needs ≥ 45 total denoising steps for clean output. With low ``strength``
    the effective step count drops — keep ``num_inference_steps`` at 50+ to compensate.

    Example::

        from diffusers.pipelines.pixeldit import PixelDiTImg2ImgPipeline
        from PIL import Image
        import torch

        pipe = PixelDiTImg2ImgPipeline.from_pretrained(
            "madtune/pixeldit-diffusers", torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")

        init = Image.open("photo.jpg").convert("RGB")
        out  = pipe(
            prompt="a cinematic landscape, golden hour",
            image=init,
            strength=0.75,
            num_inference_steps=50,
        ).images[0]
        out.save("img2img_out.png")
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load from the same HF repo as :class:`PixelDiTPipeline`.
        Internally loads a T2I pipeline, then transfers its components into this class.
        """
        import diffusers
        from .modeling_pixeldit_hf import PixelDiTModel
        if not hasattr(diffusers, "PixelDiTModel"):
            diffusers.PixelDiTModel = PixelDiTModel
        t2i = PixelDiTPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            transformer=t2i.transformer,
            scheduler=t2i.scheduler,
            text_encoder=t2i.text_encoder,
            tokenizer=t2i.tokenizer,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        strength: float = 0.8,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        flow_shift: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ) -> Union[PixelDiTPipelineOutput, Tuple]:
        """
        Args:
            prompt: Text prompt(s) guiding image generation.
            image: Input image. Accepts PIL ``Image``, ``numpy.ndarray`` (H×W×3 uint8 or float),
                or ``torch.Tensor`` (3×H×W or B×3×H×W).
            strength: How much to transform the input (0 < strength ≤ 1). ``1.0`` = full noise
                (equivalent to t2i). Recommended range: 0.5–0.85.
            negative_prompt: Optional negative prompt(s).
            height: Output height in pixels (must be divisible by 16).
            width: Output width in pixels (must be divisible by 16).
            num_inference_steps: Total scheduler steps. Use ≥ 50 for best quality.
            guidance_scale: CFG scale. ~3.5–7.5 works well.
            flow_shift: Override the scheduler's flow shift at runtime (e.g. 3.0 for 512px,
                4.0 for 1024px). Leaves the scheduler config unchanged if ``None``.
            generator: Torch RNG for reproducibility.
            output_type: ``"pil"`` (default) or ``"np"`` (uint8 numpy array).
            return_dict: If ``True`` returns :class:`PixelDiTPipelineOutput`, else a tuple.
            cross_attention_kwargs: Passed to the attention processor (e.g. ``{"scale": 0.8}``
                to adjust LoRA strength at inference).
            callback_on_step_end: Optional callable invoked at the end of each denoising step.
            callback_on_step_end_tensor_inputs: Names of tensors forwarded to the callback.

        Returns:
            :class:`PixelDiTPipelineOutput` or ``tuple``.
        """
        device = self._execution_device
        dtype  = self.transformer.dtype
        self._guidance_scale = guidance_scale
        lora_scale = (cross_attention_kwargs or {}).get("scale", None)

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        self.check_inputs(prompt, height, width, negative_prompt)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
        )

        # Override flow shift if requested (reverts after this call via set_timesteps)
        if flow_shift is not None:
            self.scheduler.config.shift = flow_shift
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Skip to the start timestep determined by strength
        t_start   = max(0, int(num_inference_steps * (1.0 - strength)))
        timesteps = timesteps[t_start:]

        if len(timesteps) == 0:
            raise ValueError(
                f"strength={strength} with num_inference_steps={num_inference_steps} "
                "produces 0 denoising steps. Increase strength or num_inference_steps."
            )

        # Preprocess image and add flow-matching noise at sigma_start
        img_tensor = _to_pixel_tensor(image, width, height, device, dtype)
        if img_tensor.shape[0] == 1 and batch_size > 1:
            img_tensor = img_tensor.expand(batch_size, -1, -1, -1).contiguous()

        sigma_start = timesteps[0].float() / 1000.0
        noise   = randn_tensor(img_tensor.shape, generator=generator, device=device, dtype=dtype)
        latents = (1.0 - sigma_start) * img_tensor + sigma_start * noise

        # Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                latent_model_input = latents
                embeds = prompt_embeds

            t_input    = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(latent_model_input, t_input, embeds)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if hasattr(self.scheduler, "scale_model_input"):
                latents = self.scheduler.step(
                    noise_pred, t,
                    self.scheduler.scale_model_input(latents, t),
                    return_dict=False,
                )[0]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_on_step_end(i, t, cb_kwargs)

        # Decode (pixel-space — just clamp and normalise)
        image_out = (latents.clamp(-1, 1) + 1) / 2
        image_out = (image_out * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        if output_type == "pil":
            image_out = [Image.fromarray(img) for img in image_out]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return PixelDiTPipelineOutput(images=image_out)
