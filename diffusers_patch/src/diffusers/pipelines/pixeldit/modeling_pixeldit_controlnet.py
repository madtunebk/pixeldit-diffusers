"""
PixelDiT ControlNet + IP-Adapter model.

Architecture follows SANA ControlNet — copied transformer blocks inject
reference-image features into the patch pathway (s tokens) via zero-init
skip projections.  IP-Adapter adds cross-attention from SigLIP patches
into every Gemma text-token slot.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


_HIDDEN     = 1536   # PixelDiT hidden_size
_SIGLIP_DIM = 1152   # google/siglip-so400m-patch14-384 hidden_size


class ControlNetBlock(nn.Module):
    """One copied transformer block + zero-init in/out projections."""

    def __init__(self, base_block, block_index: int):
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index  = block_index

        if block_index == 0:
            self.before_proj = nn.Linear(_HIDDEN, _HIDDEN)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(_HIDDEN, _HIDDEN)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, s_main, y_emb, condition, pos, pos_txt, ctrl_s):
        dt = self.after_proj.weight.dtype
        use_ckpt = self.training and torch.is_grad_enabled()

        if self.block_index == 0:
            ctrl_s = ctrl_s + self.before_proj(ctrl_s.to(dt))
            inp    = (s_main + ctrl_s).to(dt)
        else:
            inp = ctrl_s.to(dt)

        y_in = y_emb.detach().to(dt)

        if use_ckpt:
            ctrl_s, _ = grad_ckpt(
                self.copied_block, inp, y_in, condition, pos, pos_txt, None,
                use_reentrant=False,
            )
        else:
            ctrl_s, _ = self.copied_block(inp, y_in, condition, pos, pos_txt, None)

        skip = self.after_proj(ctrl_s.to(dt))
        return ctrl_s, skip


class PixelDiTControlNet(nn.Module):
    """
    Wraps a frozen PixelDiT inner transformer with a trainable ControlNet branch
    and IP-Adapter cross-attention layers.

    The ``transformer`` argument is the raw ``.model`` attribute of
    ``PixelDiTModel`` (the inner ``SanaMS`` or equivalent nn.Module), NOT
    the ``PixelDiTModel`` wrapper itself.  Use :func:`unwrap_transformer` to
    extract it from a loaded ``PixelDiTModel``.

    Training: ref_x = same clean image as the diffusion target (self-reconstruction).
    Inference: ref_x = style / reference image to transfer from.
    """

    def __init__(self, transformer, copy_blocks_num: int = 7):
        super().__init__()
        self.transformer     = transformer
        self.copy_blocks_num = copy_blocks_num

        self.controlnet_blocks = nn.ModuleList([
            ControlNetBlock(transformer.patch_blocks[i], i)
            for i in range(copy_blocks_num)
        ])

        # SigLIP → text-space conditioning (adds to y_emb in ControlNet branch)
        self.siglip_y_proj = nn.Sequential(
            nn.LayerNorm(_SIGLIP_DIM),
            nn.Linear(_SIGLIP_DIM, _HIDDEN),
        )
        nn.init.zeros_(self.siglip_y_proj[1].weight)
        nn.init.zeros_(self.siglip_y_proj[1].bias)

        # IP-Adapter cross-attention path (keys match ip_adapter_v2.pt)
        self.ip_proj = nn.Linear(_SIGLIP_DIM, _HIDDEN, bias=True)
        nn.init.normal_(self.ip_proj.weight, std=0.02)
        nn.init.zeros_(self.ip_proj.bias)
        self.ip_k = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN, bias=False) for _ in range(transformer.patch_depth)])
        self.ip_v = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN, bias=False) for _ in range(transformer.patch_depth)])
        for layer in self.ip_v:
            nn.init.normal_(layer.weight, std=0.02)
        self.ip_scale = nn.Parameter(torch.full((transformer.patch_depth,), 0.1))

        self._freeze_transformer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _freeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def trainable_params(self):
        return (
            list(self.controlnet_blocks.parameters())
            + list(self.siglip_y_proj.parameters())
            + self.ip_params()
        )

    def ip_params(self):
        return (
            list(self.ip_proj.parameters())
            + list(self.ip_k.parameters())
            + list(self.ip_v.parameters())
            + [self.ip_scale]
        )

    def encode_siglip(self, patches: torch.Tensor, n_ip: int = 256) -> torch.Tensor:
        """Pool SigLIP patch tokens to n_ip and project to hidden space."""
        resampled = F.adaptive_avg_pool1d(patches.permute(0, 2, 1), n_ip).permute(0, 2, 1)
        return self.ip_proj(resampled.to(self.ip_proj.weight.dtype))

    def _ip_cross_attn(self, y: torch.Tensor, ip: torch.Tensor, block_idx: int) -> torch.Tensor:
        B, Ny, H = y.shape
        n_heads = self.transformer.num_groups
        hd = H // n_heads
        wdtype = self.ip_k[block_idx].weight.dtype
        q  = y.to(wdtype).reshape(B, Ny, n_heads, hd).transpose(1, 2)
        ip = ip.to(wdtype)
        k  = self.ip_k[block_idx](ip).reshape(B, ip.shape[1], n_heads, hd).transpose(1, 2)
        v  = self.ip_v[block_idx](ip).reshape(B, ip.shape[1], n_heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, Ny, H).to(y.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:          torch.Tensor,                   # noisy image  [B, 3, H, W]
        t:          torch.Tensor,                   # sigma×1000   [B]
        y:          torch.Tensor,                   # Gemma embed  [B, T, 2304]
        ref_x:      torch.Tensor,                   # reference    [B, 3, H, W]
        siglip_y:   torch.Tensor    = None,         # SigLIP tokens [B, T, 1152]
        ctrl_drop:  bool            = False,
        ctrl_scale                  = 1.0,
        siglip_scale                = 1.0,
        ip_features: torch.Tensor   = None,         # [B, 256, 1536]
        ip_strength                 = 1.0,
    ) -> torch.Tensor:
        tr     = self.transformer
        tr_dev = next(tr.parameters()).device
        cn_dev = next(self.controlnet_blocks.parameters()).device
        split  = tr_dev != cn_dev

        B, _, H, W = x.shape
        Hs, Ws = H // tr.patch_size, W // tr.patch_size
        L = Hs * Ws

        x     = x.to(tr_dev)
        t     = t.to(tr_dev)
        y     = y.to(tr_dev)
        ref_x = ref_x.to(tr_dev)

        pos = tr.fetch_pos(Hs, Ws, tr_dev)

        patches     = F.unfold(x,     kernel_size=tr.patch_size, stride=tr.patch_size).transpose(1, 2)
        ref_patches = F.unfold(ref_x, kernel_size=tr.patch_size, stride=tr.patch_size).transpose(1, 2)

        t_emb = tr.t_embedder(t.view(-1)).view(B, -1, tr.hidden_size)

        Ltxt  = min(y.shape[1], tr.txt_max_length)
        y     = y[:, :Ltxt, :]
        y_emb = tr.y_embedder(y).view(B, Ltxt, tr.hidden_size)
        y_emb = y_emb + tr.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        condition = F.silu(t_emb)
        pos_txt   = tr.fetch_pos_text(Ltxt, tr_dev) if tr.use_text_rope else None

        siglip_ctrl = None
        if siglip_y is not None:
            proj_param  = next(self.siglip_y_proj.parameters())
            sig         = siglip_y[:, :Ltxt, :].to(proj_param.device, dtype=proj_param.dtype)
            siglip_ctrl = self.siglip_y_proj(sig)
            if torch.is_tensor(siglip_scale):
                sig_scale = siglip_scale.to(siglip_ctrl.device, dtype=siglip_ctrl.dtype).view(-1, 1, 1)
            else:
                sig_scale = siglip_ctrl.new_tensor(float(siglip_scale))
            siglip_ctrl = siglip_ctrl * sig_scale

        ip_for_attn = None
        if ip_features is not None:
            if ip_features.shape[0] != B:
                raise ValueError(f"ip_features batch {ip_features.shape[0]} != latent batch {B}")
            ip_for_attn = ip_features.to(self.ip_proj.weight.device, dtype=self.ip_proj.weight.dtype)

        def apply_ip(y_tokens, block_idx):
            if ip_for_attn is None:
                return y_tokens
            ip_dev = self.ip_proj.weight.device
            y_ip   = y_tokens.to(ip_dev) if y_tokens.device != ip_dev else y_tokens
            delta  = self._ip_cross_attn(y_ip, ip_for_attn, block_idx)
            if torch.is_tensor(ip_strength):
                strength = ip_strength.to(delta.device, dtype=delta.dtype).view(-1, 1, 1)
            else:
                strength = delta.new_tensor(float(ip_strength))
            block_scale = self.ip_scale[block_idx].to(delta.device, dtype=delta.dtype)
            return y_tokens + (block_scale * strength * delta).to(y_tokens.device, dtype=y_tokens.dtype)

        s      = tr.s_embedder(patches)
        ctrl_s = tr.s_embedder(ref_patches)

        use_ckpt = self.training and torch.is_grad_enabled()

        # Block 0 — no ControlNet skip yet (matches SANA pattern)
        y_emb = apply_ip(y_emb, 0)
        if use_ckpt:
            s, y_emb = grad_ckpt(tr.patch_blocks[0], s, y_emb, condition, pos, pos_txt, None, use_reentrant=False)
        else:
            s, y_emb = tr.patch_blocks[0](s, y_emb, condition, pos, pos_txt, None)

        # ControlNet blocks inject skips into main blocks 1 … copy_blocks_num
        for i in range(self.copy_blocks_num):
            s_cn    = s.to(cn_dev)     if split else s
            ctrl_s_ = ctrl_s.to(cn_dev) if split else ctrl_s
            y_cn    = y_emb.to(cn_dev)  if split else y_emb
            if siglip_ctrl is not None:
                y_cn = y_cn + siglip_ctrl.to(cn_dev, dtype=y_cn.dtype)
            cond_cn  = condition.to(cn_dev) if split else condition
            pos_cn   = pos.to(cn_dev)       if split else pos
            pos_t_cn = pos_txt.to(cn_dev)   if (split and pos_txt is not None) else pos_txt

            ctrl_s_, skip = self.controlnet_blocks[i](s_cn, y_cn, cond_cn, pos_cn, pos_t_cn, ctrl_s_)
            ctrl_s = ctrl_s_.to(tr_dev) if split else ctrl_s_

            if ctrl_drop:
                skip_eff = skip * 0.0
            else:
                if torch.is_tensor(ctrl_scale):
                    scale = ctrl_scale.to(skip.device, dtype=skip.dtype).view(-1, 1, 1)
                else:
                    scale = skip.new_tensor(float(ctrl_scale))
                skip_eff = skip * scale
            s = s + skip_eff.to(tr_dev, dtype=s.dtype)

            y_emb = apply_ip(y_emb, i + 1)
            if use_ckpt:
                s, y_emb = grad_ckpt(tr.patch_blocks[i + 1], s, y_emb, condition, pos, pos_txt, None, use_reentrant=False)
            else:
                s, y_emb = tr.patch_blocks[i + 1](s, y_emb, condition, pos, pos_txt, None)

        # Remaining blocks
        for i in range(self.copy_blocks_num + 1, tr.patch_depth):
            y_emb = apply_ip(y_emb, i)
            if use_ckpt:
                s, y_emb = grad_ckpt(tr.patch_blocks[i], s, y_emb, condition, pos, pos_txt, None, use_reentrant=False)
            else:
                s, y_emb = tr.patch_blocks[i](s, y_emb, condition, pos, pos_txt, None)

        # Pixel pathway (unchanged from base model)
        s      = F.silu(t_emb + s)
        s_cond = s.view(B * L, tr.hidden_size)

        x_pixels = tr.pixel_embedder(x, img_height=H, img_width=W, patch_size=tr.patch_size)
        for blk in tr.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, tr.patch_size, None)

        x_pixels = tr.final_layer(x_pixels)
        P2       = tr.patch_size ** 2
        x_pixels = x_pixels.view(B, L, P2, tr.out_channels).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, tr.out_channels * P2, L)
        return F.fold(x_pixels, (H, W), kernel_size=tr.patch_size, stride=tr.patch_size)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def unwrap_transformer(pixeldit_model):
    """Extract the inner nn.Module from a PixelDiTModel wrapper."""
    return pixeldit_model.model


def _load_tensors(path: str):
    """Load a flat tensor dict from either .safetensors or .pt/.pth."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu"), {}
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # legacy nested format — extract metadata separately
    meta = {k: v for k, v in ckpt.items() if not isinstance(v, (dict, list, torch.Tensor))}
    return ckpt, meta


def _unflatten_to_state_dict(flat: dict, prefix: str) -> dict:
    """Extract a sub-state-dict from a flat safetensors dict."""
    p = prefix + "."
    return {k[len(p):]: v for k, v in flat.items() if k.startswith(p)}


def _unflatten_list_of_state_dicts(flat: dict, prefix: str, n: int):
    """Rebuild a list of state dicts from flat safetensors keys."""
    result = []
    for i in range(n):
        p = f"{prefix}.{i}."
        sd = {k[len(p):]: v for k, v in flat.items() if k.startswith(p)}
        result.append(sd)
    return result


def load_checkpoint(model: PixelDiTControlNet, path: str) -> int:
    """Load a combined ControlNet + IP-Adapter checkpoint (.safetensors or .pt)."""
    flat, meta = _load_tensors(path)

    if path.endswith(".safetensors"):
        model.controlnet_blocks.load_state_dict(_unflatten_to_state_dict(flat, "controlnet_blocks"))
        siglip_sd = _unflatten_to_state_dict(flat, "siglip_y_proj")
        if siglip_sd:
            model.siglip_y_proj.load_state_dict(siglip_sd)
        ip_proj_sd = _unflatten_to_state_dict(flat, "ip_proj")
        if ip_proj_sd:
            model.ip_proj.load_state_dict(ip_proj_sd)
            n = len(model.ip_k)
            for layer, sd in zip(model.ip_k, _unflatten_list_of_state_dicts(flat, "ip_k", n)):
                layer.load_state_dict(sd)
            for layer, sd in zip(model.ip_v, _unflatten_list_of_state_dicts(flat, "ip_v", n)):
                layer.load_state_dict(sd)
            model.ip_scale.data.copy_(flat["ip_scale"].to(model.ip_scale.device, dtype=model.ip_scale.dtype))
        step = int(meta.get("step", 0))
    else:
        model.controlnet_blocks.load_state_dict(flat["controlnet_blocks"])
        if "siglip_y_proj" in flat:
            model.siglip_y_proj.load_state_dict(flat["siglip_y_proj"])
        if "ip_proj" in flat:
            model.ip_proj.load_state_dict(flat["ip_proj"])
            for layer, state in zip(model.ip_k, flat["ip_k"]):
                layer.load_state_dict(state)
            for layer, state in zip(model.ip_v, flat["ip_v"]):
                layer.load_state_dict(state)
            model.ip_scale.data.copy_(flat["ip_scale"].to(model.ip_scale.device, dtype=model.ip_scale.dtype))
        step = int(flat.get("step", 0))

    return step


def load_ip_adapter_checkpoint(model: PixelDiTControlNet, path: str) -> int:
    """Load an IP-Adapter-only checkpoint on top of an existing ControlNet model."""
    flat, meta = _load_tensors(path)

    if path.endswith(".safetensors"):
        model.ip_proj.load_state_dict(_unflatten_to_state_dict(flat, "ip_proj"))
        n = len(model.ip_k)
        for layer, sd in zip(model.ip_k, _unflatten_list_of_state_dicts(flat, "ip_k", n)):
            layer.load_state_dict(sd)
        for layer, sd in zip(model.ip_v, _unflatten_list_of_state_dicts(flat, "ip_v", n)):
            layer.load_state_dict(sd)
        model.ip_scale.data.copy_(flat["ip_scale"].to(model.ip_scale.device, dtype=model.ip_scale.dtype))
        step = int(meta.get("step", 0))
    else:
        model.ip_proj.load_state_dict(flat["ip_proj"])
        if len(flat["ip_k"]) != len(model.ip_k) or len(flat["ip_v"]) != len(model.ip_v):
            raise ValueError("IP adapter checkpoint block count does not match PixelDiT patch depth")
        for layer, state in zip(model.ip_k, flat["ip_k"]):
            layer.load_state_dict(state)
        for layer, state in zip(model.ip_v, flat["ip_v"]):
            layer.load_state_dict(state)
        model.ip_scale.data.copy_(flat["ip_scale"].to(model.ip_scale.device, dtype=model.ip_scale.dtype))
        step = int(flat.get("step", 0))

    return step
