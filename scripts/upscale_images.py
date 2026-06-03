"""
Upscale images 4x with RealESRGAN (512 → 2048).
Precompute will then center-crop to 1024×1024 — giving the model real
reconstructed detail instead of bicubic blur.

Usage:
    python scripts/upscale_images.py \
        --input  /home/nobus/Workbech/Yadex \
        --output /home/nobus/Raid0/lora_corry_2048 \
        --device cuda:1

Model is downloaded automatically on first run (~67 MB).
"""

# ── torchvision compatibility shim ───────────────────────────────────────────
# basicsr 1.x imports torchvision.transforms.functional_tensor which was
# removed in torchvision 0.16+. Patch it before basicsr loads.
import sys, types
import torchvision.transforms.functional as _tvf
_ft = types.ModuleType("torchvision.transforms.functional_tensor")
_ft.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _ft
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def load_upscaler(device: str) -> RealESRGANer:
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=4,
    )
    weights_url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )
    cache_dir = Path.home() / ".cache" / "realesrgan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "RealESRGAN_x4plus.pth"

    if not weights_path.exists():
        print(f"Downloading RealESRGAN weights → {weights_path}")
        import urllib.request
        urllib.request.urlretrieve(weights_url, weights_path)

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=model,
        tile=512,       # tile size to avoid OOM on large images
        tile_pad=10,
        pre_pad=0,
        half=True,      # fp16 inference
        device=device,
    )
    return upsampler


def find_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in exts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="folder with source images")
    ap.add_argument("--output", required=True, help="folder for 2048px output")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ext",    default="png",  choices=["png", "jpg"],
                    help="output format (png = lossless, recommended)")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = find_images(src)
    if not imgs:
        raise RuntimeError(f"No images found in {src}")
    print(f"Found {len(imgs)} images → upscaling 4x with RealESRGAN on {args.device}")

    upsampler = load_upscaler(args.device)

    skipped = 0
    for img_path in tqdm(imgs):
        out_path = dst / (img_path.stem + f".{args.ext}")
        if out_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  WARNING: could not read {img_path.name} — skipping")
            continue

        out, _ = upsampler.enhance(img, outscale=4)

        if args.ext == "png":
            cv2.imwrite(str(out_path), out)
        else:
            cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nDone. {len(imgs) - skipped} upscaled → {dst}/")
    if skipped:
        print(f"  ({skipped} already existed, skipped)")
    print(f"\nNext step:")
    print(f"  python scripts/precompute_lora_data.py \\")
    print(f"    --images {dst} \\")
    print(f"    --out /home/nobus/Raid0/lora_corry_cache_1024 \\")
    print(f"    --size 1024 --trigger corrychase \\")
    print(f"    --recaption --focus woman --device {args.device}")


if __name__ == "__main__":
    main()
