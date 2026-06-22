"""
PixelDiT Styled Pipeline — ControlNet + IP-Adapter style transfer.

Combines:
  • ControlNet scribble conditioning (HED edge map from the reference image)
  • IP-Adapter SigLIP style conditioning
  • Flow-matching img2img variation

Reference image drives both structure (ControlNet) and style (IP-Adapter).
A text prompt is optional; leave it empty for pure reference-driven generation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .pipeline_pixeldit import PixelDiTPipeline
    from .pipeline_output import PixelDiTPipelineOutput
    from .modeling_pixeldit_controlnet import (
        PixelDiTControlNet,
        load_checkpoint,
        load_ip_adapter_checkpoint,
        unwrap_transformer,
    )
    from .image_processor_hed import HEDExtractor, control_to_tensor, hed_to_scribble
except ImportError:
    import importlib, os as _os, sys as _sys
    _pkg = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg not in _sys.path:
        _sys.path.insert(0, _pkg)
    from pipeline_pixeldit import PixelDiTPipeline                          # noqa: E402
    from pipeline_output import PixelDiTPipelineOutput                      # noqa: E402
    from modeling_pixeldit_controlnet import (                               # noqa: E402
        PixelDiTControlNet,
        load_checkpoint,
        load_ip_adapter_checkpoint,
        unwrap_transformer,
    )
    from image_processor_hed import HEDExtractor, control_to_tensor, hed_to_scribble  # noqa: E402


def _pil_to_u8(image: Image.Image, width: int, height: int) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((width, height)), dtype=np.uint8).copy()


def _sigma_schedule(steps: int, flow_shift: float, device, dtype) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return (flow_shift * t / (1.0 + (flow_shift - 1.0) * t)).to(device=device, dtype=dtype)


class PixelDiTStyledPipeline(PixelDiTPipeline):
    """
    Style-transfer pipeline for PixelDiT using ControlNet scribble conditioning
    and IP-Adapter SigLIP image conditioning.

    Load via :meth:`from_pretrained_styled`:

    .. code-block:: python

        from diffusers.pipelines.pixeldit import PixelDiTStyledPipeline

        pipe = PixelDiTStyledPipeline.from_pretrained_styled(
            "madtune/pixeldit-diffusers",
            controlnet_path="/path/to/controlnet_scribble_ip_768.pt",
            ip_adapter_path="/path/to/ip_adapter_v2.pt",   # optional
            hed_ckpt_path="/path/to/ControlNetHED.pth",    # optional
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload(gpu_id=1)

        out = pipe(
            image=Image.open("style_ref.jpg"),
            prompt="gothic pale woman, dramatic rim lighting",
            variation_strength=0.85,
            ctrl_strength=0.25,
            ip_strength=0.85,
        ).images[0]

    Or from HuggingFace Hub:

    .. code-block:: python

        from huggingface_hub import hf_hub_download

        pipe = PixelDiTStyledPipeline.from_pretrained_styled(
            "madtune/pixeldit-diffusers",
            controlnet_path=hf_hub_download("madtune/pixeldit-controlnet-ip", "controlnet_scribble_ip_768.pt"),
            ip_adapter_path=hf_hub_download("madtune/pixeldit-controlnet-ip", "ip_adapter_v2.pt"),
            hed_ckpt_path=hf_hub_download("madtune/pixeldit-controlnet-ip", "ControlNetHED.pth"),
            torch_dtype=torch.bfloat16,
        )
    """

    # siglip_model and siglip_processor are optional — set as instance attrs
    # after from_pretrained_styled, not registered modules, because they use
    # a different loading path (transformers, not diffusers).
    _optional_components = ["siglip_model", "siglip_processor"]

    def __init__(self, transformer, scheduler, text_encoder, tokenizer, controlnet):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_modules(controlnet=controlnet)
        self.siglip_model     = None
        self.siglip_processor = None
        self._hed_extractor   = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_styled(
        cls,
        pretrained_model_name_or_path: str,
        controlnet_path: str,
        ip_adapter_path: Optional[str] = None,
        hed_ckpt_path:   Optional[str] = None,
        copy_blocks_num: int = 7,
        siglip_model_id: str = "google/siglip-so400m-patch14-384",
        **kwargs,
    ) -> "PixelDiTStyledPipeline":
        """
        Load the base PixelDiT model then attach ControlNet + IP-Adapter weights.

        Args:
            pretrained_model_name_or_path: HF repo or local path for the base model
                (e.g. ``"madtune/pixeldit-diffusers"``).
            controlnet_path: Local path to ``controlnet_scribble_ip_768.pt``.
                This checkpoint may also contain the IP-Adapter weights — if so,
                ``ip_adapter_path`` is optional.
            ip_adapter_path: Optional separate ``ip_adapter_v2.pt``.  Loaded on top
                of ``controlnet_path`` when provided.
            hed_ckpt_path: Optional path to ``ControlNetHED.pth``.  Required if you
                want automatic edge extraction from the reference image.  Omit when
                you will always pass an explicit ``control_image`` to ``__call__``.
            copy_blocks_num: Number of transformer blocks copied into the ControlNet
                branch.  Must match the training config (default 7).
            siglip_model_id: HF model id for the SigLIP encoder (default:
                ``google/siglip-so400m-patch14-384``).
            **kwargs: Forwarded to ``PixelDiTPipeline.from_pretrained``
                (e.g. ``torch_dtype``, ``device_map``).
        """
        import diffusers
        try:
            from .modeling_pixeldit_hf import PixelDiTModel
        except ImportError:
            from modeling_pixeldit_hf import PixelDiTModel

        if not hasattr(diffusers, "PixelDiTModel"):
            diffusers.PixelDiTModel = PixelDiTModel

        dtype = kwargs.get("torch_dtype", torch.float32)

        print("[PixelDiTStyledPipeline] Loading base model…")
        t2i = PixelDiTPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)

        print("[PixelDiTStyledPipeline] Building ControlNet…")
        inner = unwrap_transformer(t2i.transformer)
        controlnet = PixelDiTControlNet(inner, copy_blocks_num=copy_blocks_num)

        print(f"[PixelDiTStyledPipeline] Loading ControlNet checkpoint: {controlnet_path}")
        step = load_checkpoint(controlnet, controlnet_path)
        print(f"  step={step}")

        if ip_adapter_path is not None:
            print(f"[PixelDiTStyledPipeline] Loading IP-Adapter checkpoint: {ip_adapter_path}")
            ip_step = load_ip_adapter_checkpoint(controlnet, ip_adapter_path)
            print(f"  ip_step={ip_step}")

        controlnet = controlnet.to(dtype=dtype)

        pipe = cls(
            transformer=t2i.transformer,
            scheduler=t2i.scheduler,
            text_encoder=t2i.text_encoder,
            tokenizer=t2i.tokenizer,
            controlnet=controlnet,
        )

        print(f"[PixelDiTStyledPipeline] Loading SigLIP: {siglip_model_id}")
        from transformers import AutoImageProcessor, SiglipVisionModel
        pipe.siglip_processor = AutoImageProcessor.from_pretrained(siglip_model_id)
        pipe.siglip_model     = SiglipVisionModel.from_pretrained(
            siglip_model_id, torch_dtype=dtype
        ).eval()

        if hed_ckpt_path is not None:
            print(f"[PixelDiTStyledPipeline] Loading HED extractor: {hed_ckpt_path}")
            pipe._hed_extractor = HEDExtractor(hed_ckpt_path, device="cpu")

        return pipe

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_siglip(self, image: Image.Image, device, dtype) -> torch.Tensor:
        """Return IP-Adapter features for ``image`` via SigLIP + controlnet projection."""
        inputs  = self.siglip_processor(images=image, return_tensors="pt").to(device)
        patches = self.siglip_model(
            pixel_values=inputs["pixel_values"].to(dtype)
        ).last_hidden_state                                 # [1, N, 1152]
        return self.controlnet.encode_siglip(patches)       # [1, 256, 1536]

    def _extract_control(
        self,
        image_u8: np.ndarray,
        control_image: Optional[Image.Image],
        width: int,
        height: int,
        hed_thickness: int,
    ) -> np.ndarray:
        """Return HW uint8 scribble map from either a provided image or auto-HED."""
        if control_image is not None:
            ctrl = np.asarray(
                control_image.convert("L").resize((width, height), Image.NEAREST),
                dtype=np.uint8,
            ).copy()
            return np.where(ctrl > 127, 255, 0).astype(np.uint8)

        if self._hed_extractor is None:
            raise ValueError(
                "No control_image provided and no HED extractor loaded.  "
                "Pass hed_ckpt_path to from_pretrained_styled, or supply control_image."
            )
        return self._hed_extractor(image_u8, thickness=hed_thickness)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompt: Union[str, List[str]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        control_image: Optional[Image.Image] = None,
        width:  Optional[int] = None,
        height: Optional[int] = None,
        variation_strength: float = 0.85,
        ctrl_strength:      float = 0.25,
        ip_strength:        float = 0.85,
        flow_shift:         float = 8.0,
        guidance_scale:     float = 4.5,
        num_inference_steps: int  = 50,
        hed_thickness:      int   = 2,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = [],
        **kwargs,
    ) -> Union[PixelDiTPipelineOutput, Tuple]:
        """
        Run styled image generation.

        Args:
            image: Reference image — drives both ControlNet structure and IP-Adapter style.
            prompt: Optional text prompt.  Leave empty for pure reference-driven output.
            negative_prompt: Optional negative text.
            control_image: Optional pre-computed scribble map (white edges on black, PIL L or RGB).
                If ``None``, HED edges are extracted automatically from ``image``.
            width / height: Output resolution.  Defaults to reference image size (snapped to ×16).
            variation_strength: How much to vary from the reference (0 = copy, 1 = full noise).
                Recommended: 0.65–0.95.
            ctrl_strength: ControlNet skip scale.  0.25 is a good starting point.
            ip_strength: IP-Adapter style scale.  0.35 = subtle, 0.85 = strong.
            flow_shift: Flow-matching shift parameter.  Higher = more detail (7–8 for 768+ px).
            guidance_scale: CFG scale.  3.5–5.0 works well.
            num_inference_steps: Total denoising steps (≥ 50 recommended).
            hed_thickness: Scribble line thickness (0 = thin/erode, 2 = default, 4+ = thick).
            generator: Torch RNG for reproducibility.
            output_type: ``"pil"`` or ``"np"``.
            return_dict: Return :class:`PixelDiTPipelineOutput` if ``True``, else tuple.
            callback_on_step_end: Called at the end of each denoising step with
                ``(step_index, sigma, kwargs_dict)``.
        """
        device = self._execution_device
        dtype  = next(self.controlnet.parameters()).dtype

        # ── image size ────────────────────────────────────────────
        orig_w, orig_h = image.size
        if width is None:
            width  = (orig_w // 16) * 16
        if height is None:
            height = (orig_h // 16) * 16
        width  = (width  // 16) * 16
        height = (height // 16) * 16

        image_rgb = _pil_to_u8(image, width, height)

        # ── text encoding ─────────────────────────────────────────
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        has_prompt = any(p.strip() for p in prompt)
        if has_prompt:
            y_text, _ = self.encode_prompt(
                prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=False,
            )
        else:
            # null text — match original flow-matching behaviour (pure zeros)
            y_text = torch.zeros(batch_size, 300, 2304, dtype=dtype, device=device)
        y_null = torch.zeros_like(y_text)

        # ── control map ───────────────────────────────────────────
        if ctrl_strength > 0.0:
            scribble_u8 = self._extract_control(image_rgb, control_image, width, height, hed_thickness)
        else:
            scribble_u8 = np.zeros((height, width), dtype=np.uint8)
        ref_x = control_to_tensor(scribble_u8).unsqueeze(0).to(device, dtype=dtype)
        if batch_size > 1:
            ref_x = ref_x.expand(batch_size, -1, -1, -1).contiguous()

        # ── SigLIP / IP-Adapter features ──────────────────────────
        if ip_strength > 0.0 and self.siglip_model is not None:
            siglip_dev  = next(self.siglip_model.parameters()).device
            ip_features = self._encode_siglip(image, siglip_dev, dtype).to(device)
            if batch_size > 1:
                ip_features = ip_features.expand(batch_size, -1, -1).contiguous()
        else:
            ip_features = None

        # ── sigma schedule ────────────────────────────────────────
        sigmas    = _sigma_schedule(num_inference_steps, flow_shift, device, dtype)
        variation = float(np.clip(variation_strength, 0.0, 1.0))
        start     = max(0, min(num_inference_steps, round((1.0 - variation) * num_inference_steps)))

        # ── reference image tensor ────────────────────────────────
        ref_img = torch.from_numpy(image_rgb).to(device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        ref_img = ref_img / 127.5 - 1.0
        if batch_size > 1:
            ref_img = ref_img.expand(batch_size, -1, -1, -1).contiguous()

        noise = torch.randn(ref_img.shape, generator=generator, dtype=dtype).to(device)

        if start >= num_inference_steps:
            x = ref_img.clone()
        else:
            sigma_s = sigmas[start]
            x = (1.0 - sigma_s) * ref_img + sigma_s * noise

        # ── CFG scale tensors ─────────────────────────────────────
        # [0] = uncond branch, [1] = cond branch
        ctrl_scales = torch.tensor([0.0, ctrl_strength], dtype=dtype, device=device)
        ip_scales   = torch.tensor([0.0, ip_strength],   dtype=dtype, device=device)

        # ── denoising loop ────────────────────────────────────────
        total = num_inference_steps - start
        self._num_timesteps = total

        ctx = (
            torch.amp.autocast("cuda", dtype=dtype)
            if device != "cpu" and torch.cuda.is_available()
            else torch.no_grad()
        )

        with torch.inference_mode(), ctx:
            for step_i, i in enumerate(self.progress_bar(range(start, num_inference_steps))):
                sigma      = sigmas[i].item()
                sigma_next = sigmas[i + 1].item()
                t_val      = torch.full((2 * batch_size,), sigma * 1000, dtype=dtype, device=device)

                x_in   = x.repeat(2, 1, 1, 1)
                y_in   = torch.cat([y_null, y_text])
                ref_in = ref_x.repeat(2, 1, 1, 1)
                ip_in  = ip_features.repeat(2, 1, 1) if ip_features is not None else None

                v_batch = self.controlnet(
                    x_in, t_val, y_in, ref_in,
                    ctrl_scale=ctrl_scales,
                    ip_features=ip_in,
                    ip_strength=ip_scales,
                )
                v_u, v_c = v_batch.chunk(2)
                v = v_u + guidance_scale * (v_c - v_u)
                x = x + (sigma_next - sigma) * v

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals().get(k) for k in callback_on_step_end_tensor_inputs}
                    callback_on_step_end(step_i, sigma, cb_kwargs)

        # ── decode ────────────────────────────────────────────────
        image_out = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

        if output_type == "pil":
            image_out = [Image.fromarray(img) for img in image_out]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return PixelDiTPipelineOutput(images=image_out)
