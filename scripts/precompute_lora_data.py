"""
Precompute image+caption pairs for LoRA training.

Reads a folder of images, optionally captions them with Qwen2.5-VL,
injects a trigger word, then encodes with Gemma or Qwen.

Output:
    {out}/lora_images.npy   [N, 3, size, size] float16  (default size=1024)
    {out}/lora_embs.npy     [N, 300, 2304]     float16
    {out}/lora_masks.npy    [N, 300]            uint8
    {out}/meta.json         encoder info

Usage:
    # with existing .txt captions, Gemma encoder (default)
    python scripts/precompute_lora_data.py --images /path/to/images --out /path/to/cache --trigger "yarn art style"

    # auto-caption + Gemma encoder
    python scripts/precompute_lora_data.py --images /path/to/images --out /path/to/cache --trigger "yarn art style" --recaption

    # Qwen encoder (match generate.py --encoder qwen)
    python scripts/precompute_lora_data.py --images /path/to/images --out /path/to/cache --encoder qwen --proj qwen_proj.pt --trigger "yarn art style" --recaption
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_GEMMA_ID   = "Efficient-Large-Model/gemma-2-2b-it"
_QWEN_VL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_TXT_MAX    = 300

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

def make_img_transform(size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


def find_images(images_dir):
    exts = {"jpg", "jpeg", "png", "webp"}
    imgs = [
        p for p in sorted(Path(images_dir).iterdir())
        if p.is_file() and p.suffix.lstrip(".").lower() in exts
    ]
    return imgs


def clean_caption(text):
    text = " ".join(text.strip().split())
    for prefix in [
        "The image features ",
        "The image shows ",
        "The image depicts ",
        "This image features ",
        "This image shows ",
        "This image depicts ",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip(" ,.")


def recaption(img_paths, device, focus=None, trigger=None, caption_style="prompt"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("Loading Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(_QWEN_VL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _QWEN_VL_ID, torch_dtype=torch.bfloat16
    ).to(device).eval()

    if caption_style == "prompt":
        if focus:
            user_text = (
                f"Write one concise prompt-style caption for LoRA training. "
                f"Describe the main {focus}'s stable identity traits first, including hair color and style, "
                f"eye color if visible, face, skin tone, and body build. Then describe clothing, pose, "
                f"expression, and background. Use comma-separated visual phrases. "
                f"Do not mention that this is an image. Do not say 'The image features', 'Create', or 'Generate'. "
                f"Do not use any real person name or trigger token."
            )
            system_text = (
                "You write compact image-generation training captions. "
                "Return only comma-separated visual phrases, no full-sentence explanation."
            )
        else:
            user_text = (
                "Write one concise prompt-style caption for LoRA training. "
                "Use comma-separated visual phrases for subject, colors, materials, style, composition, "
                "pose, and background. Do not mention that this is an image. "
                "Do not say 'The image features', 'Create', or 'Generate'."
            )
            system_text = (
                "You write compact image-generation training captions. "
                "Return only comma-separated visual phrases, no full-sentence explanation."
            )
    elif focus:
        user_text = (
            f"Describe the {focus} in this image in detail. "
            f"Cover their appearance, face, hair color and style, eye color, skin tone, "
            f"clothing, outfit details, pose, expression, and the background/setting. "
            f"Be specific and descriptive. Do not use the word '{focus}' — describe what you see."
        )
        system_text = (
            f"You are an image description assistant specializing in character descriptions. "
            f"Always describe the {focus} as the main subject. "
            f"Never say 'Create an image of' or 'Generate'. Just describe what you see."
        )
    else:
        user_text = "Describe this image in detail. Focus on colors, textures, materials, style, and composition. Be concise and descriptive."
        system_text = "You are an image description assistant. Describe images concisely in plain English. Never say 'Create an image of' or 'Generate'. Just describe what you see."

    captions = []
    for img_path in tqdm(img_paths, desc="captioning"):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text",  "text": user_text},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        caption = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        caption = clean_caption(caption)
        if trigger:
            caption = f"{trigger}, {caption}"
        img_path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        captions.append(caption)

    del model
    torch.cuda.empty_cache()
    return captions


def encode_gemma(captions, device, batch=32):
    print("Loading Gemma...")
    tok = AutoTokenizer.from_pretrained(_GEMMA_ID)
    tok.padding_side = "right"
    model = (AutoModelForCausalLM.from_pretrained(_GEMMA_ID, torch_dtype=torch.bfloat16)
             .get_decoder().eval().to(device))
    num_chi = len(tok.encode(_CHI_PROMPT))
    max_len = num_chi + _TXT_MAX - 2
    select  = [0] + list(range(-(_TXT_MAX - 1), 0))

    all_embs, all_masks = [], []
    print("Encoding with Gemma...")
    for i in tqdm(range(0, len(captions), batch), desc="encoding"):
        batch_caps = [_CHI_PROMPT + c for c in captions[i:i+batch]]
        t = tok(batch_caps, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(t.input_ids, attention_mask=t.attention_mask).last_hidden_state
        all_embs.append(emb[:, select, :].cpu().to(torch.float16))  # float16 for storage
        all_masks.append(t.attention_mask[:, select].cpu().to(torch.uint8))

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_embs), torch.cat(all_masks)


def encode_qwen(captions, proj_path, device, batch=32):
    from transformers import AutoModel
    import torch.nn as nn

    _QWEN_ID  = "Qwen/Qwen3-2B"
    _QWEN_DIM = 2048

    print("Loading Qwen3-2B...")
    qtok = AutoTokenizer.from_pretrained(_QWEN_ID)
    qtok.padding_side = "right"
    qmodel = AutoModel.from_pretrained(_QWEN_ID, torch_dtype=torch.float16).eval().to(device)

    proj = nn.Linear(_QWEN_DIM, 2304, bias=False).to(torch.float16).to(device)
    if proj_path and os.path.exists(proj_path):
        sd = torch.load(proj_path, map_location="cpu", weights_only=True)
        proj.load_state_dict(sd)
        print(f"Loaded projection: {proj_path}")
    else:
        raise RuntimeError(f"qwen_proj.pt not found at {proj_path} — run train_qwen_proj.py first")

    num_chi = len(qtok.encode(_CHI_PROMPT))
    max_len = num_chi + _TXT_MAX - 2
    select  = [0] + list(range(-(_TXT_MAX - 1), 0))

    all_embs, all_masks = [], []
    print("Encoding with Qwen+projection...")
    for i in tqdm(range(0, len(captions), batch), desc="encoding"):
        batch_caps = [_CHI_PROMPT + c for c in captions[i:i+batch]]
        t = qtok(batch_caps, max_length=max_len, padding="max_length",
                 truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = qmodel(**t).last_hidden_state
            emb = emb[:, select, :].to(torch.float16)
            emb = proj(emb)
        all_embs.append(emb.cpu())
        all_masks.append(t.attention_mask[:, select].cpu().to(torch.uint8))

    del qmodel, proj
    torch.cuda.empty_cache()
    return torch.cat(all_embs), torch.cat(all_masks)


def verify_embeddings(embs_mm, label="embs"):
    arr = np.array(embs_mm[:min(4, len(embs_mm))])
    nz  = np.count_nonzero(arr)
    if nz == 0:
        raise RuntimeError(
            f"BUG: {label} are ALL ZEROS — encoding failed silently. "
            "Check GPU memory, model load, and that captions are non-empty."
        )
    print(f"[verify] {label}: min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  nonzero={nz}/{arr.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",    required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--size",      type=int, default=1024,
                    help="square crop size for training images (must match inference resolution, default 1024)")
    ap.add_argument("--trigger",   default=None,  help="trigger word/phrase to prepend to every caption")
    ap.add_argument("--recaption", action="store_true", help="auto-caption with Qwen2.5-VL")
    ap.add_argument("--focus",     default=None,  help="subject focus for VL captioning, e.g. 'person' or 'woman'")
    ap.add_argument("--caption-style", default="prompt", choices=["prompt", "descriptive"],
                    help="recaption format: prompt-style comma phrases or descriptive prose")
    ap.add_argument("--encoder",    default="gemma", choices=["gemma", "qwen"],
                    help="text encoder (must match generate.py --encoder)")
    ap.add_argument("--proj",       default="qwen_proj.pt", help="path to qwen_proj.pt (only for --encoder qwen)")
    ap.add_argument("--device",     default="cuda:0", help="device for text encoder")
    ap.add_argument("--vl-device",  default=None,     help="device for Qwen2.5-VL captioning (default: same as --device)")
    ap.add_argument("--batch",      type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device    = torch.device(args.device)
    vl_device = torch.device(args.vl_device if args.vl_device else args.device)
    img_transform = make_img_transform(args.size)

    img_paths = find_images(args.images)
    if not img_paths:
        raise RuntimeError(f"No images found in {args.images}")
    N = len(img_paths)
    print(f"Found {N} images  encoder={args.encoder}  size={args.size}×{args.size}")

    # memmaps
    images_mm = np.lib.format.open_memmap(f"{args.out}/lora_images.npy", mode="w+", dtype=np.float16, shape=(N, 3, args.size, args.size))
    embs_mm   = np.lib.format.open_memmap(f"{args.out}/lora_embs.npy",   mode="w+", dtype=np.float16, shape=(N, _TXT_MAX, 2304))
    masks_mm  = np.lib.format.open_memmap(f"{args.out}/lora_masks.npy",  mode="w+", dtype=np.uint8,   shape=(N, _TXT_MAX))

    # 1. process images
    print("Processing images...")
    for i, img_path in enumerate(tqdm(img_paths)):
        images_mm[i] = img_transform(Image.open(img_path)).numpy().astype(np.float16)
    images_mm.flush()

    # 2. get captions
    if args.recaption:
        captions = recaption(
            img_paths, vl_device, focus=args.focus, trigger=args.trigger, caption_style=args.caption_style
        )
    else:
        captions = []
        for img_path in img_paths:
            txt = img_path.with_suffix(".txt")
            cap = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            if not cap:
                print(f"  WARNING: no caption for {img_path.name} — using filename")
                cap = img_path.stem.replace("-", " ").replace("_", " ")
            captions.append(cap)

    # 3. inject trigger word (only for non-recaption mode; recaption already injects per-image)
    if args.trigger and not args.recaption:
        captions = [f"{args.trigger}, {c}" for c in captions]

    if args.trigger:
        print(f"Trigger: '{args.trigger}'")
        print(f"Example: {captions[0][:120]}")

    # 4. encode
    if args.encoder == "gemma":
        embs, masks = encode_gemma(captions, device, args.batch)
    else:
        embs, masks = encode_qwen(captions, args.proj, device, args.batch)

    embs_mm[:]  = embs.numpy()
    masks_mm[:] = masks.numpy()
    embs_mm.flush()
    masks_mm.flush()

    # 5. verify — catch silent zero-fill bugs immediately
    verify_embeddings(embs_mm, "lora_embs")

    # 6. save metadata
    meta = {
        "encoder": args.encoder,
        "n_samples": N,
        "emb_dim": 2304,
        "seq_len": _TXT_MAX,
        "img_size": args.size,
        "trigger": args.trigger,
        "recaption": args.recaption,
        "focus": args.focus,
        "caption_style": args.caption_style,
        "images_dir": args.images,
        "caption_example": captions[0] if captions else None,
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {N} samples → {args.out}/")
    print(f"  encoder : {args.encoder}")
    print(f"  embs    : {args.out}/lora_embs.npy  {embs.shape}")
    print(f"  masks   : {args.out}/lora_masks.npy  {masks.shape}")
    print(f"  images  : {args.out}/lora_images.npy  {images_mm.shape}")


if __name__ == "__main__":
    main()
