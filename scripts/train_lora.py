"""
LoRA fine-tuning for PixelDiT using precomputed image+caption embeddings.

Flow matching loss: predict velocity (noise - x), noisy image via shifted schedule.
Only LoRA weights update — base model is fully frozen.

Usage:
    # Precompute first:
    python scripts/precompute_lora_data.py --images /data/my_images --out /data/lora_cache

    # Train:
    python scripts/train_lora.py --data /data/lora_cache --out lora_out/ --epochs 100

    # Inference with trained LoRA:
    from peft import PeftModel
    from pixeldit.modeling_pixeldit_hf import PixelDiTModel
    model = PixelDiTModel.from_pretrained("madtune/pixeldit-diffusers", subfolder="transformer")
    model = PeftModel.from_pretrained(model, "lora_out/")
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---- Flow schedule (matching NVIDIA's training config) ----------------------

_T = 1000

def _build_flow_schedule(flow_shift: float):
    # sigmas[t] ≈ shift * (t/1000) / (1 + (shift-1) * (t/1000))
    # matches FlowMatchEulerDiscreteScheduler with the same shift value
    betas      = np.linspace(1.0, 0.001, _T, dtype=np.float64)
    sigmas_raw = 1.0 - betas
    sigmas     = flow_shift * sigmas_raw / (1 + (flow_shift - 1) * sigmas_raw)
    alphas     = 1.0 - sigmas
    return torch.from_numpy(sigmas).float(), torch.from_numpy(alphas).float()


def q_sample(x, t, noise, alphas, sigmas):
    a = alphas[t].view(-1, 1, 1, 1)
    s = sigmas[t].view(-1, 1, 1, 1)
    return a * x + s * noise


# ---- Dataset ----------------------------------------------------------------

class LoraDataset(Dataset):
    def __init__(self, data_dir):
        meta_path = os.path.join(data_dir, "meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            print(
                f"[dataset] encoder={self.meta.get('encoder', 'unknown')}  "
                f"n={self.meta.get('n_samples')}  "
                f"emb_dim={self.meta.get('emb_dim')}  "
                f"trigger={self.meta.get('trigger') or 'none'}"
            )
        self.imgs  = np.load(os.path.join(data_dir, "lora_images.npy"), mmap_mode="r")
        self.embs  = np.load(os.path.join(data_dir, "lora_embs.npy"),  mmap_mode="r")
        self.masks = np.load(os.path.join(data_dir, "lora_masks.npy"), mmap_mode="r")
        assert len(self.imgs) == len(self.embs) == len(self.masks)
        # sanity check: catch all-zeros from a failed precompute run
        sample = self.embs[:min(4, len(self.embs))]
        if np.count_nonzero(sample) == 0:
            raise RuntimeError(
                f"lora_embs.npy in {data_dir} are ALL ZEROS — "
                "re-run precompute_lora_data.py to regenerate."
            )

    def __len__(self):
        return len(self.imgs)

    @property
    def img_size(self):
        return self.meta.get("img_size", 512)

    def __getitem__(self, idx):
        img  = torch.from_numpy(self.imgs[idx].astype(np.float32))   # [3, H, H]
        emb  = torch.from_numpy(self.embs[idx].astype(np.float32))   # [300, 2304]
        mask = torch.from_numpy(self.masks[idx].astype(np.float32))  # [300]
        return img, emb, mask


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True,       help="precomputed cache dir")
    ap.add_argument("--out",     default="lora_out/",  help="output dir for LoRA weights")
    ap.add_argument("--model",   default="madtune/pixeldit-diffusers")
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--batch",   type=int,   default=2)
    ap.add_argument("--accum",   type=int,   default=4,   help="gradient accumulation steps")
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--lora_r",  type=int,   default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--cfg_drop", type=float, default=0.1, help="CFG dropout probability")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--flow_shift", type=float, default=4.0,
                    help="flow schedule shift — must match inference scheduler (default 4.0 for 1024px)")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="enable gradient checkpointing to reduce VRAM at the cost of speed")
    ap.add_argument("--timestep_logit_std", type=float, default=1.0,
                    help="std of logit-normal timestep sampling (higher = more uniform; 0 = pure midpoint)")
    ap.add_argument("--loss_weighting", default="sigma_sqrt", choices=["sigma_sqrt", "none"],
                    help="sigma_sqrt upweights low-noise steps (identity/detail); none = uniform (Flux default: sigma_sqrt)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load model + inject LoRA
    print("Loading PixelDiTModel...")
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        print("peft not installed — run: pip install peft")
        sys.exit(1)

    from diffusers.pipelines.pixeldit import PixelDiTModel

    model = PixelDiTModel.from_pretrained(args.model, subfolder="transformer")

    lora_cfg = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        target_modules = ["qkv_x", "qkv_y", "proj_x", "proj_y"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if args.grad_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("[+] Gradient checkpointing enabled")

    model = model.to(device).train()

    # null embedding for CFG dropout — zeros
    null_emb  = torch.zeros(1, 300, 2304, device=device)
    null_mask = torch.zeros(1, 300, device=device)

    # 2. Dataset + loader
    dataset = LoraDataset(args.data)
    img_size = dataset.img_size
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} samples  img_size={img_size}  batch={args.batch}  accum={args.accum}  steps/epoch={len(loader)}")

    # 3. Optimizer
    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2
    )
    total_steps = args.epochs * len(loader) // args.accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)

    sigmas_cpu, alphas_cpu = _build_flow_schedule(args.flow_shift)
    alphas = alphas_cpu.to(device)
    sigmas = sigmas_cpu.to(device)
    print(f"Flow shift: {args.flow_shift}")
    best_loss = float("inf")
    step = 0

    def save_lora(path, loss):
        model.save_pretrained(path)
        meta = {
            "data_dir": args.data,
            "model": args.model,
            "img_size": img_size,
            "epochs": args.epochs,
            "batch": args.batch,
            "accum": args.accum,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "cfg_drop": args.cfg_drop,
            "flow_shift": args.flow_shift,
            "timestep_logit_std": args.timestep_logit_std,
            "loss_weighting": args.loss_weighting,
            "loss": loss,
            "precompute": dataset.meta,
        }
        with open(os.path.join(path, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        opt.zero_grad()

        for i, (imgs, embs, _masks) in enumerate(bar):
            imgs = imgs.to(device)   # [B, 3, H, H]
            embs = embs.to(device)   # [B, 300, 2304]
            B    = imgs.shape[0]

            # CFG dropout: replace some embeddings with the null (empty) embedding
            if args.cfg_drop > 0:
                drop = torch.rand(B, device=device) < args.cfg_drop
                embs[drop] = null_emb.expand(drop.sum(), -1, -1)

            # Logit-normal timestep sampling — concentrates gradient signal
            # around mid-noise where the model does the most meaningful work.
            noise = torch.randn_like(imgs)
            u = torch.sigmoid(torch.randn(B, device=device) * args.timestep_logit_std)
            t = (u * _T).long().clamp(0, _T - 1)
            x_t    = q_sample(imgs, t, noise, alphas, sigmas)
            target = noise - imgs  # velocity: direction from data → noise

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_t.bfloat16(), t, embs.bfloat16())
                # per-sample MSE, then apply sigma weighting before reducing
                loss_per = F.mse_loss(pred.float(), target, reduction="none").mean(dim=(1, 2, 3))

            # sigma_sqrt: weight = 1/sigma² — upweights low-noise steps where
            # identity and fine detail are learned; matches Flux Dev training.
            if args.loss_weighting == "sigma_sqrt":
                sig = sigmas[t].clamp(min=1e-3)          # [B]
                w   = (1.0 / sig ** 2).to(loss_per.device)
                loss = (w * loss_per).mean()
            else:
                loss = loss_per.mean()

            (loss / args.accum).backward()
            total_loss += loss.item()
            n += 1

            if (i + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

            bar.set_postfix(loss=f"{total_loss/n:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        # apply gradients from any tail batches that didn't fill a full accum window
        if n % args.accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1

        avg = total_loss / n
        print(f"  epoch {epoch+1}  loss={avg:.4f}")

        is_last   = (epoch + 1) == args.epochs
        is_save   = (epoch + 1) % args.save_every == 0
        is_best   = avg < best_loss

        if is_best:
            best_loss = avg
            save_lora(os.path.join(args.out, "best"), avg)
            print(f"  best saved → {args.out}/best/")

        if is_save or is_best or is_last:
            save_lora(os.path.join(args.out, "latest"), avg)

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"LoRA saved to {args.out}/")
    print("\nTo use in inference:")
    print("  pipe.load_lora_weights(")
    print(f'      "{args.out}/best", weight_name="adapter_model.safetensors", adapter_name="corrychase"')
    print("  )")
    print("  pipe.set_adapters([\"corrychase\"], adapter_weights=[2.0])")
    print("  # prompt with the exact trained trigger: corrychase")


if __name__ == "__main__":
    main()
