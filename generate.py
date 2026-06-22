import argparse
import os
import random
import torch

# ── Settings (edit these to run without CLI args) ─────────────────────────────
PROMPT        = None          # None → runs the built-in batch of fun prompts
NEGATIVE      = "blurry, low quality, deformed, duplicate buildings, bad anatomy, cropped, text, watermark, oversaturated, low detail, noisy, distorted perspective"
ENCODER       = "gemma"       # "gemma" | "qwen" | "siglip"
PROJ          = None          # path to qwen_proj.pt / siglip_proj.pt (required for qwen/siglip)
HEIGHT        = 1024
WIDTH         = 1024
STEPS         = 50            # minimum 45 — below that output is garbage
CFG           = 7.5
SEED          = None          # None → random each run
OUT           = "output/"
DEVICE        = "cuda"
LORA          = None          # path to LoRA adapter folder  e.g. "lora_yarn_out/best"
LORA_SCALE    = 1.0           # LoRA influence on the text encoder (cross_attention_kwargs scale)
SCHEDULER     = "euler"       # "euler" | "heun" | "lcm"
# ─────────────────────────────────────────────────────────────────────────────

_HF_REPO  = "madtune/pixeldit-diffusers"
_GEMMA_ID = "Efficient-Large-Model/gemma-2-2b-it"

PROMPTS = [
    "A majestic white duck wearing ornate golden papal robes stands on a floating cathedral above the clouds. Tens of thousands of faithful ducks gather below in giant flying flocks. Sunbeams pierce the heavens while giant stained-glass wings unfold behind the Duck Pope. Epic religious fantasy, absurd realism, cinematic lighting, masterpiece, highly detailed.",
    "A single banana sits on a throne in a post-apocalyptic wasteland. Thousands of survivors in ragged armor make a pilgrimage across the desert to witness the Last Banana. Massive ruined cities loom in the distance. The banana radiates divine golden light. Ultra-detailed, dramatic, photorealistic, epic scale, absurd but serious.",
    "A giant fluffy hamster emperor sits inside a colossal mechanical battle fortress shaped like a hamster wheel. Thousands of tiny engineers operate steam-powered machinery while armies march beneath banners displaying hamster symbols. Epic dieselpunk fantasy, cinematic atmosphere, highly detailed, absurd realism.",
    "Seven ancient frogs wearing wizard robes sit around a glowing crystal table deep inside an enchanted swamp. Magical fireflies illuminate the fog while giant mushrooms tower overhead. The frogs debate the fate of reality itself. High fantasy, ultra-detailed, cinematic lighting, masterpiece.",
    "A stern orange tabby cat wearing a three-piece Victorian suit arrives at a medieval village riding a giant snail. Terrified villagers hand over fish, milk, and yarn as taxes. The cat records everything in a massive leather ledger. Hyper-realistic, cinematic, absurd historical fantasy.",
    "A gigantic capybara king leads an army across a frozen mountain pass. Thousands of armored capybara knights carry banners into battle while enormous mammoth-like creatures march alongside them. Epic fantasy warfare, dramatic snowfall, photorealistic, cinematic scale.",
    "A tiny Chihuahua stands alone on a mountain of skulls beneath a blood-red sky. Behind it rises an army of terrified dragons, demons, giants, and monsters. The Chihuahua looks completely calm and mildly annoyed. Epic dark fantasy, absurd realism, cinematic masterpiece, ultra-detailed.",
]


def _adapter_dir(path, component):
    if not path:
        return None
    component_dir = os.path.join(path, component)
    if os.path.exists(os.path.join(component_dir, "adapter_config.json")):
        return component_dir
    if component == "transformer" and os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    return None

def load_pipeline(encoder, proj, device, lora, scheduler, lora_scale=1.0, lora_component="all"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from diffusers.pipelines.pixeldit import PixelDiTPipeline, PixelDiTModel

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    print("[1/2] Loading Gemma text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(_GEMMA_ID)
    tokenizer.padding_side = "right"
    gemma = (
        AutoModelForCausalLM.from_pretrained(_GEMMA_ID, torch_dtype=torch.bfloat16)
        .get_decoder().eval()
    )

    print(f"[2/2] Loading pipeline from {_HF_REPO}...")
    pipe = PixelDiTPipeline.from_pretrained(
        _HF_REPO,
        text_encoder=gemma,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )

    if lora:
        print(f"[+] Loading LoRA from {lora}...")
        transformer_lora = _adapter_dir(lora, "transformer")
        text_lora = _adapter_dir(lora, "text_encoder")

        if lora_component in ("all", "transformer"):
            if transformer_lora:
                pipe.load_lora_weights(transformer_lora)
                if lora_scale != 1.0:
                    set_lora_scale(pipe, lora_scale)
                    print(f"[+] PixelDiT LoRA scale -> {lora_scale}")
            else:
                print(f"[!] No PixelDiT transformer adapter found under {lora}")
        else:
            print("[+] Skipping PixelDiT transformer LoRA")

        if lora_component in ("all", "text_encoder") and text_lora:
            if encoder == "gemma":
                from peft import PeftModel
                pipe.text_encoder = PeftModel.from_pretrained(
                    pipe.text_encoder, text_lora, is_trainable=False
                ).eval()
                print("[+] Loaded Gemma text-encoder LoRA")
            else:
                print("[!] Gemma text-encoder LoRA present but encoder=qwen; skipping it")
        elif lora_component in ("all", "text_encoder"):
            print(f"[!] No Gemma text-encoder adapter found under {lora}")

    if encoder == "qwen":
        from diffusers.pipelines.pixeldit import QwenEncoder
        print("[+] Swapping to Qwen text encoder...")
        pipe.text_encoder = QwenEncoder(proj_path=proj, output_device=device)

    if encoder == "siglip":
        from diffusers.pipelines.pixeldit.text_encoder_siglip import SiglipEncoder
        print("[+] Swapping to SigLIP text encoder...")
        pipe.text_encoder = SiglipEncoder(proj_path=proj, output_device=device)

    if scheduler != "euler":
        from diffusers import FlowMatchHeunDiscreteScheduler, FlowMatchLCMScheduler
        schedulers = {"heun": FlowMatchHeunDiscreteScheduler, "lcm": FlowMatchLCMScheduler}
        pipe.scheduler = schedulers[scheduler].from_config(pipe.scheduler.config)
        print(f"[+] Scheduler → {scheduler}")

    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


def set_lora_scale(pipe, scale):
    """Scale transformer LoRA strength. 1.0 = full, 0.5 = subtle, 1.5 = aggressive."""
    from peft.tuners.lora.layer import LoraLayer
    for module in pipe.transformer.modules():
        if isinstance(module, LoraLayer):
            module.scale_layer(scale)


def run_pipe(pipe, prompt, negative_prompt, height, width, steps, cfg, seed):
    out = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",          default=PROMPT)
    ap.add_argument("--negative_prompt", default=NEGATIVE)
    ap.add_argument("--encoder",         default=ENCODER,    choices=["gemma", "qwen", "siglip"])
    ap.add_argument("--proj",            default=PROJ,       help="qwen_proj.pt / siglip_proj.pt path")
    ap.add_argument("--height",  type=int,   default=HEIGHT)
    ap.add_argument("--width",   type=int,   default=WIDTH)
    ap.add_argument("--steps",   type=int,   default=STEPS)
    ap.add_argument("--cfg",     type=float, default=CFG)
    ap.add_argument("--seed",    type=int,   default=SEED)
    ap.add_argument("--out",                 default=OUT)
    ap.add_argument("--save",                default=None,   help="exact output file path (overrides --out)")
    ap.add_argument("--device",              default=DEVICE)
    ap.add_argument("--lora",                default=LORA,       help="LoRA adapter folder")
    ap.add_argument("--lora_scale", type=float, default=LORA_SCALE)
    ap.add_argument("--lora_component", default="all", choices=["all", "transformer", "text_encoder"],
                    help="which adapter component to load from --lora")
    ap.add_argument("--scheduler",           default=SCHEDULER,  choices=["euler", "heun", "lcm"])
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipeline(
        args.encoder, args.proj, args.device, args.lora, args.scheduler,
        args.lora_scale, args.lora_component,
    )

    print(f"  seed: {seed}  (--seed {seed} to reproduce)")
    prompts = [args.prompt] if args.prompt else PROMPTS

    for i, prompt in enumerate(prompts):
        print(f"\n[gen] [{i+1}/{len(prompts)}] {prompt[:80]}")
        img = run_pipe(
            pipe, prompt, args.negative_prompt,
            args.height, args.width, args.steps, args.cfg, seed,
        )
        if args.save and len(prompts) == 1:
            fname = args.save
            os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
        else:
            safe = prompt[:50].replace(" ", "_").replace(",", "").replace("'", "")
            fname = os.path.join(args.out, f"{i:02d}_{safe}.jpg")
        img.save(fname)
        print(f"  saved → {fname}")


if __name__ == "__main__":
    main()
