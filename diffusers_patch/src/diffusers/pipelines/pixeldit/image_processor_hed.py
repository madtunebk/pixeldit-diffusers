"""
HED edge detector + control-map utilities for PixelDiT ControlNet.

ControlNetHED_Apache2 is adapted from
  Sana/tools/controlnet/annotator/hed/__init__.py  (Apache-2.0).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# HED neural-network edge detector
# ------------------------------------------------------------------

class _DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in range(n_layers - 1)]
        self.convs      = nn.Sequential(*layers)
        self.projection = nn.Conv2d(out_ch, 1, 1)

    def forward(self, x, down_sampling=False):
        h = F.max_pool2d(x, 2, 2) if down_sampling else x
        for conv in self.convs:
            h = F.relu(conv(h))
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    """VGG-style HED edge detector (Apache-2.0)."""

    def __init__(self):
        super().__init__()
        self.norm   = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1 = _DoubleConvBlock(3,   64,  2)
        self.block2 = _DoubleConvBlock(64,  128, 2)
        self.block3 = _DoubleConvBlock(128, 256, 3)
        self.block4 = _DoubleConvBlock(256, 512, 3)
        self.block5 = _DoubleConvBlock(512, 512, 3)

    def forward(self, x):
        h = x - self.norm
        h, p1 = self.block1(h)
        h, p2 = self.block2(h, down_sampling=True)
        h, p3 = self.block3(h, down_sampling=True)
        h, p4 = self.block4(h, down_sampling=True)
        h, p5 = self.block5(h, down_sampling=True)
        return p1, p2, p3, p4, p5


# ------------------------------------------------------------------
# Control-map post-processing
# ------------------------------------------------------------------

def _nms(x: np.ndarray, threshold: int = 127, sigma: float = 3.0) -> np.ndarray:
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
    kernels = [
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8),
    ]
    y = np.zeros_like(x)
    for k in kernels:
        np.putmask(y, cv2.dilate(x, kernel=k) == x, x)
    out = np.zeros_like(y, dtype=np.uint8)
    out[y > threshold] = 255
    return out


def hed_to_scribble(edge_u8: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Convert soft HED output → sparse binary scribble map."""
    if edge_u8.ndim == 3:
        edge_u8 = cv2.cvtColor(edge_u8, cv2.COLOR_RGB2GRAY)
    edge = _nms(edge_u8, threshold=127, sigma=3.0)
    edge = cv2.GaussianBlur(edge, (0, 0), 3.0)
    edge[edge > 4]   = 255
    edge[edge < 255] = 0
    if thickness == 0:
        edge = cv2.erode(edge, np.ones((3, 3), np.uint8), iterations=1)
    elif thickness > 1:
        k    = max(1, thickness // 2)
        edge = cv2.dilate(edge, np.ones((k, k), np.uint8), iterations=1)
    return edge.astype(np.uint8)


def control_to_tensor(edge_u8: np.ndarray) -> torch.Tensor:
    """Convert HW uint8 scribble map → (3, H, W) float tensor in [-1, 1]."""
    if edge_u8.ndim == 2:
        edge_u8 = np.stack([edge_u8] * 3, axis=0)
    else:
        edge_u8 = edge_u8.transpose(2, 0, 1)
    return torch.from_numpy(edge_u8.astype(np.float32) / 127.5 - 1.0)


# ------------------------------------------------------------------
# High-level extractor
# ------------------------------------------------------------------

class HEDExtractor:
    """
    Loads ``ControlNetHED_Apache2`` from a checkpoint and extracts scribble
    control maps from PIL images.

    Args:
        ckpt_path: Path to ``ControlNetHED.pth``.
        device: Torch device string.
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = device
        self.net = ControlNetHED_Apache2().float().to(device).eval()
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(ckpt_path, device="cpu")
        else:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
        self.net.load_state_dict(sd)

    @torch.no_grad()
    def __call__(self, image_rgb_u8: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        Args:
            image_rgb_u8: HWC uint8 RGB numpy array.
            thickness: Scribble dilation (0 = erode, 1 = keep, 2+ = dilate).
        Returns:
            HW uint8 scribble map.
        """
        from einops import rearrange
        H, W = image_rgb_u8.shape[:2]
        t = torch.from_numpy(image_rgb_u8.astype(np.float32)).to(self.device)
        t = rearrange(t, "h w c -> 1 c h w")
        edges_out = self.net(t)
        edges_np  = [e.cpu().numpy().astype(np.float32)[0, 0] for e in edges_out]
        edges_np  = [cv2.resize(e, (W, H)) for e in edges_np]
        edge_map  = 1.0 / (1.0 + np.exp(-np.mean(np.stack(edges_np, 2), 2).astype(np.float64)))
        edge_u8   = (edge_map * 255).clip(0, 255).astype(np.uint8)
        return hed_to_scribble(edge_u8, thickness=thickness)
