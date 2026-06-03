import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, set_weights_and_activate_adapters

from ...pipelines.pixeldit.pixeldit_t2i import PixDiT_T2I


class PixelDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    PixelDiT 1.3B pixel-space diffusion transformer.
    Diffusers-native ModelMixin — supports from_pretrained, save_pretrained, and peft LoRA.
    """

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        num_groups=24,
        hidden_size=1536,
        pixel_hidden_size=16,
        pixel_attn_hidden_size=1152,
        pixel_num_groups=16,
        patch_depth=14,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=2304,
        txt_max_length=300,
        use_text_rope=True,
        text_rope_theta=10000.0,
        repa_encoder_index=-1,
        use_pixel_abs_pos=True,
    ):
        super().__init__()
        self.model = PixDiT_T2I(
            in_channels            = in_channels,
            num_groups             = num_groups,
            hidden_size            = hidden_size,
            pixel_hidden_size      = pixel_hidden_size,
            pixel_attn_hidden_size = pixel_attn_hidden_size,
            pixel_num_groups       = pixel_num_groups,
            patch_depth            = patch_depth,
            pixel_depth            = pixel_depth,
            num_text_blocks        = num_text_blocks,
            patch_size             = patch_size,
            txt_embed_dim          = txt_embed_dim,
            txt_max_length         = txt_max_length,
            use_text_rope          = use_text_rope,
            text_rope_theta        = text_rope_theta,
            repa_encoder_index     = repa_encoder_index,
            use_pixel_abs_pos      = use_pixel_abs_pos,
        )

    def forward(self, x, t, y, s=None, mask=None):
        return self.model(x, t, y, s=s, mask=mask)

    # ── Gradient checkpointing ─────────────────────────────────────────────

    def enable_input_require_grads(self):
        """
        Make the patch embedder's output require gradients so that
        gradient checkpointing can propagate through the patch blocks.
        Required when using PEFT + gradient checkpointing together.
        """
        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
        self._grad_hook = self.model.s_embedder.register_forward_hook(_hook)

    def disable_input_require_grads(self):
        if hasattr(self, "_grad_hook"):
            self._grad_hook.remove()
            del self._grad_hook

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the 14 MMDiT patch blocks."""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def is_gradient_checkpointing(self) -> bool:
        return getattr(self.model, "gradient_checkpointing", False)

    # ──────────────────────────────────────────────────────────────────────

    def set_attn_processor(self, processor) -> None:
        """Set a custom attention processor on all MMDiTJointAttention layers."""
        from ...pipelines.pixeldit.pixeldit_t2i import MMDiTJointAttention
        for module in self.modules():
            if isinstance(module, MMDiTJointAttention):
                module.set_processor(processor)

    def set_adapters(self, adapter_names, weights=None):
        """Set active adapters with optional per-adapter scale weights."""
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        if weights is None:
            weights = [1.0] * len(adapter_names)
        elif not isinstance(weights, list):
            weights = [weights] * len(adapter_names)
        set_weights_and_activate_adapters(self, adapter_names, weights)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Standard diffusers loading, with fallback to model.safetensors naming."""
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            subfolder = kwargs.pop("subfolder", "")
            dtype = kwargs.pop("torch_dtype", None)
            fname = f"{subfolder}/model.safetensors" if subfolder else "model.safetensors"
            weights_file = hf_hub_download(pretrained_model_name_or_path, fname)
            config, _, _ = cls.load_config(
                pretrained_model_name_or_path, subfolder=subfolder, return_unused_kwargs=True
            )
            model = cls(**{k: v for k, v in config.items() if not k.startswith("_")})
            sd = load_file(weights_file, device="cpu")
            model.load_state_dict(sd, strict=False)
            if dtype is not None:
                model = model.to(dtype)
            return model

    @classmethod
    def from_pth(cls, pth_path: str, **kwargs):
        """Load from original nvidia .pth checkpoint, handles core. prefix."""
        model = cls(**kwargs)
        state = torch.load(pth_path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        sd = {(k[5:] if k.startswith("core.") else k): v for k, v in sd.items()}
        sd = {"model." + k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[PixelDiTModel.from_pth] loaded — {len(missing)} missing, {len(unexpected)} unexpected")
        return model
