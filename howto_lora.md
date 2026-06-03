# LoRA Training — Full Pipeline

Everything needed to train and use a character or style LoRA on PixelDiT.

---

## Quick Reference

```
images (source)
    └─ scripts/upscale_images.py            → upscaled images (2048px)
         └─ scripts/precompute_lora_data.py → cache/ (images + embeddings)
              └─ scripts/train_lora.py       → output/lora_xxx/best/
                   └─ generate.py --lora     → output/*.jpg
```

---

## Step 0 — Upscale source images (if they are under 1024px)

Training at 1024px requires source images ≥ 1024px on the short edge.
If your images are smaller, upscale first with RealESRGAN (4× — real detail, not bicubic blur).

```bash
python scripts/upscale_images.py \
    --input  /path/to/raw_images \
    --output /path/to/upscaled \
    --device cuda:0
```

- Downloads `RealESRGAN_x4plus.pth` (~67 MB) automatically on first run
- Output is ~4× the source resolution — precompute center-crops to 1024×1024
- Skips files that already exist (safe to re-run)

**Skip this step** if your images are already ≥ 1024px on the short side.

---

## Step 1 — Precompute image + caption embeddings

Runs once per dataset. Outputs go to a cache folder the training script reads directly.

### Style LoRA (e.g. yarn art, watercolor, oil painting)

```bash
python scripts/precompute_lora_data.py \
    --images  /path/to/style_images \
    --out     /path/to/cache_dir \
    --size    1024 \
    --trigger "yarn art style" \
    --recaption \
    --device  cuda:0
```

### Character LoRA (person identity)

```bash
python scripts/precompute_lora_data.py \
    --images  /path/to/person_images \
    --out     /path/to/cache_dir \
    --size    1024 \
    --trigger "mychrname" \
    --recaption \
    --focus   woman \
    --device  cuda:0
```

### Re-use existing captions (skip Qwen2.5-VL)

If `.txt` files already exist next to your images (from a previous `--recaption` run),
omit `--recaption` and precompute reads them directly — much faster.

```bash
# copy captions from source folder to upscaled folder first
cp /path/to/raw_images/*.txt /path/to/upscaled/

python scripts/precompute_lora_data.py \
    --images /path/to/upscaled \
    --out    /path/to/cache_dir \
    --size   1024 \
    --trigger "yarn art style" \
    --device cuda:0
```

### Key arguments

| Arg | Default | Notes |
|---|---|---|
| `--size` | `1024` | Must match inference resolution. Training at 512 = inference at 512. |
| `--trigger` | none | Word prepended to every caption — use it in your prompt at inference |
| `--recaption` | off | Auto-caption with Qwen2.5-VL-3B. Slow but best quality |
| `--focus` | none | Subject hint for VL captioning: `woman`, `man`, `cat`, `dog` |
| `--caption-style` | `prompt` | `prompt` = comma phrases (better for LoRA); `descriptive` = prose |
| `--encoder` | `gemma` | Must match `generate.py --encoder`. Default Gemma is correct for most cases |
| `--vl-device` | same as `--device` | Separate GPU for Qwen2.5-VL if you want to split work |
| `--batch` | `16` | Gemma encoding batch size. Reduce if OOM |

### Output files

```
cache_dir/
├── lora_images.npy   [N, 3, 1024, 1024]  float16
├── lora_embs.npy     [N, 300, 2304]      float16
├── lora_masks.npy    [N, 300]            uint8
└── meta.json
```

---

## Step 2 — Train

### Style LoRA

```bash
python scripts/train_lora.py \
    --data    /path/to/cache_dir \
    --out     output/lora_yarn \
    --epochs  80 \
    --batch   2 --accum 4 \
    --lora_r  16 --lora_alpha 16 \
    --loss_weighting sigma_sqrt \
    --device  cuda:0
```

### Character LoRA (at 1024px — needs grad checkpointing)

```bash
python scripts/train_lora.py \
    --data    /path/to/cache_dir \
    --out     output/lora_mycharacter \
    --epochs  50 \
    --batch   1 --accum 8 \
    --lora_r  32 --lora_alpha 32 \
    --loss_weighting sigma_sqrt \
    --grad_ckpt \
    --device  cuda:0
```

### Key arguments

| Arg | Default | Notes |
|---|---|---|
| `--epochs` | `100` | Style: 80–150. Character: 50–80. |
| `--batch` | `2` | At 1024px use `1`. At 512px use `2`. |
| `--accum` | `4` | Effective batch = batch × accum. Keep ≥ 8. |
| `--lora_r` | `16` | Rank. `32` for identity/character; `16` for style |
| `--lora_alpha` | `16` | Keep equal to `lora_r` |
| `--lr` | `1e-4` | Don't go above `2e-4` |
| `--flow_shift` | `4.0` | Must match inference scheduler. Do not change. |
| `--loss_weighting` | `sigma_sqrt` | `sigma_sqrt` = 1/σ² weighting (same as Flux). Upweights low-noise steps. |
| `--grad_ckpt` | off | Saves ~35% VRAM. Required at 1024px on 12GB. ~20% slower. |
| `--timestep_logit_std` | `1.0` | Logit-normal sampling concentration. Higher = more uniform. |
| `--save_every` | `10` | Save checkpoint every N epochs. `latest/` always saved on final epoch. |
| `--cfg_drop` | `0.1` | 10% CFG dropout during training |

### What gets LoRA

Targets: `qkv_x`, `qkv_y`, `proj_x`, `proj_y` in all 14 MMDiT attention blocks.

```
14 blocks × 4 modules × 2 (A/B) = 112 tensors
r=16 → ~4.1M trainable  |  r=32 → ~8.2M trainable  (out of 1.3B total)
```

### Loss curve guide

| Phase | Loss range | What's happening |
|---|---|---|
| Epoch 1–5 | drops fast | coarse structure learning |
| Epoch 5–30 | slow improvement | identity / style locking in |
| Epoch 30+ | plateau | converged — stop or continue to taste |
| Below 0.01 fast | danger zone | likely overfitting, reduce epochs |

---

## Step 3 — Generate with LoRA

### CLI

```bash
# style LoRA
python generate.py \
    --lora   output/lora_yarn/best \
    --prompt "a wolf in the forest, yarn art style" \
    --device cuda:0

# character LoRA
python generate.py \
    --lora   output/lora_mycharacter/best \
    --prompt "mychrname, portrait, cinematic lighting" \
    --device cuda:0
```

### Settings block in generate.py (no CLI args)

Edit the top of `generate.py`:

```python
PROMPT     = "a dragon perched on a mountain, yarn art style"
LORA       = "output/lora_yarn/best"
LORA_SCALE = 1.0
DEVICE     = "cuda:0"
```

### LoRA scale

`LORA_SCALE` / `--lora_scale` scales the LoRA influence on the text encoder.

- `1.0` — full (default)
- `0.7` — softer, more prompt flexibility
- `1.5` — stronger push (may reduce prompt following)

### Python API

```python
from diffusers.pipelines.pixeldit import PixelDiTPipeline

pipe = PixelDiTPipeline.from_pretrained("madtune/pixeldit-diffusers", ...)
pipe.load_lora_weights("output/lora_yarn/best", adapter_name="yarn")
pipe.enable_model_cpu_offload()

image = pipe(
    "a wolf in the forest, yarn art style",
    cross_attention_kwargs={"scale": 0.8},
).images[0]
```

### Multiple LoRAs

```python
pipe.load_lora_weights("output/lora_yarn/best",        adapter_name="style")
pipe.load_lora_weights("output/lora_mycharacter/best", adapter_name="char")

# style LoRA light, character LoRA strong
pipe.set_adapters(["char", "style"], adapter_weights=[1.8, 0.25])
```

Prompt with both triggers:

```
mychrname, portrait, yarn art style, cinematic lighting
```

If style dominates identity, lower style weight. If identity is lost, raise it.

### Save / fuse / remove

```python
pipe.save_lora_weights("my_export/")           # save before fusing
pipe.fuse_lora(lora_scale=1.0, safe_fusing=True)  # bake into weights
pipe.unload_lora_weights()                     # remove adapters
```

---

## Troubleshooting

### Identity / style doesn't stick

1. **Resolution mismatch** — check `meta.json` → `img_size`. Must match inference resolution.
2. **flow_shift mismatch** — check `training_meta.json` → `flow_shift`. Must be `4.0`.
3. **Too few epochs** — try more.
4. **Trigger word** — always include the trigger in your inference prompt.

### OOM at 1024px

- Add `--grad_ckpt`
- Use `--batch 1 --accum 8`
- Reduce `--lora_r` from 32 to 16

### LoRA loads but crashes

Check `adapter_config.json` → `auto_mapping.parent_library`.
Must be `diffusers.pipelines.pixeldit.modeling_pixeldit_hf`. If not, retrain.

### `lora_embs.npy` all zeros

Gemma OOMed silently. Reduce `--batch` in precompute (try `--batch 4`) and re-run.

---

## Scripts reference

| Script | Purpose |
|---|---|
| `upscale_images.py` | RealESRGAN 4× upscale before precompute |
| `precompute_lora_data.py` | Build image+embedding cache for LoRA training |
| `train_lora.py` | LoRA fine-tuning |
| `precompute_proj_embs.py` | Build embedding cache for Qwen projector training |
| `train_qwen_proj_fast.py` | Train Qwen→Gemma projection (analytical + optional SGD) |
| `setup_diffusers_pixeldit.py` | Install/update pipeline into active venv |
| `download_hf_dataset.py` | Download a HF image+caption dataset |
| `download_unsplash.py` | Download images from Pexels by search query |
| `convert_checkpoint.py` | Convert NVIDIA .pth checkpoint → HF safetensors format |
