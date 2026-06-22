import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import os
import gradio as gr
import torch
import random
try:
    from diffusers import PixelDiTImg2ImgPipeline, PixelDiTPipeline, PixelDiTStyledPipeline
except ImportError:
    import subprocess, sys, importlib
    subprocess.run([sys.executable, "scripts/setup_diffusers_pixeldit.py"], check=True)
    for _k in [k for k in sys.modules if k.startswith("diffusers")]:
        del sys.modules[_k]
    importlib.invalidate_caches()
    from diffusers import PixelDiTImg2ImgPipeline, PixelDiTPipeline, PixelDiTStyledPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

_GPU_ID = 1 if torch.cuda.device_count() > 1 else 0
_DEVICE  = f"cuda:{_GPU_ID}" if torch.cuda.is_available() else "cpu"

_pipe        = None
_styled_pipe = None


def _load_pipeline():
    global _pipe
    _pipe = PixelDiTImg2ImgPipeline.from_pretrained(
        "madtune/pixeldit-diffusers", torch_dtype=torch.bfloat16
    )
    _pipe.enable_model_cpu_offload(gpu_id=_GPU_ID)
    print(f"[pipeline] loaded, offload gpu_id={_GPU_ID}")


def _load_styled_pipeline():
    global _styled_pipe
    controlnet_path = hf_hub_download("madtune/pixeldit-controlnet", "controlnet.safetensors")
    ip_adapter_path = hf_hub_download("madtune/pixeldit-controlnet", "ip_adapter.safetensors")
    hed_ckpt_path   = hf_hub_download("madtune/pixeldit-controlnet", "hed_detector.safetensors")
    _styled_pipe = PixelDiTStyledPipeline.from_pretrained_styled(
        "madtune/pixeldit-diffusers",
        controlnet_path=controlnet_path,
        ip_adapter_path=ip_adapter_path,
        hed_ckpt_path=hed_ckpt_path,
        torch_dtype=torch.bfloat16,
    )
    _styled_pipe.enable_model_cpu_offload(gpu_id=_GPU_ID)
    print(f"[styled pipeline] loaded, offload gpu_id={_GPU_ID}")


def get_pipeline():
    if _pipe is None:
        _load_pipeline()
    return _pipe


def get_styled_pipeline():
    if _styled_pipe is None:
        _load_styled_pipeline()
    return _styled_pipe


def safeguard(image: Image.Image, max_size_w: int = 1280, max_size_h: int = 1280, multiple: int = 16):
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    orig_w, orig_h = image.size
    scale = min(1.0, max_size_w / orig_w, max_size_h / orig_h)
    new_w = (int(orig_w * scale) // multiple) * multiple
    new_h = (int(orig_h * scale) // multiple) * multiple
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    print(f"Image: {orig_w}x{orig_h} → {new_w}x{new_h}")
    return new_w, new_h, image


_DEFAULT_PROMPT   = "A fantasy landscape with mountains and a river, in the style of Studio Ghibli"
_DEFAULT_NEGATIVE = "low quality, blurry, deformed, bad anatomy, extra limbs, ugly, distorted face"


def new_seed():
    return random.randint(0, 2**32 - 1)


def generate_t2i(prompt, negative_prompt, flow_shift, seed, steps, cfg, width, height, progress=gr.Progress()):
    pipe = get_pipeline()
    pipe.scheduler.config.shift = float(flow_shift)
    gen   = torch.Generator("cpu").manual_seed(int(seed))
    total = int(steps)

    def _cb(i, t, cb_kwargs):
        progress((i + 1) / total, desc=f"Step {i + 1} / {total}")
        return cb_kwargs

    return PixelDiTPipeline.__call__(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=total,
        guidance_scale=float(cfg),
        width=int(width),
        height=int(height),
        generator=gen,
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]


def generate_i2i(image, prompt, negative_prompt, strength, flow_shift, seed, steps, cfg, width, height, progress=gr.Progress()):
    width, height, image = safeguard(image)
    pipe  = get_pipeline()
    pipe.scheduler.config.shift = float(flow_shift)
    gen   = torch.Generator("cpu").manual_seed(int(seed))
    total = int(steps)

    def _cb(i, t, cb_kwargs):
        progress((i + 1) / total, desc=f"Step {i + 1} / {total}")
        return cb_kwargs

    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        strength=float(strength),
        num_inference_steps=total,
        guidance_scale=float(cfg),
        width=int(width),
        height=int(height),
        generator=gen,
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]


def generate_styled(
    image, prompt, variation_strength, ctrl_strength, ip_strength,
    flow_shift, seed, steps, cfg, hed_thickness, progress=gr.Progress()
):
    if image is None:
        raise gr.Error("Upload a reference image first.")
    _, _, image = safeguard(image)
    pipe  = get_styled_pipeline()
    gen   = torch.Generator("cpu").manual_seed(int(seed))
    total = int(steps)

    def _cb(step_i, sigma, cb_kwargs):
        progress((step_i + 1) / total, desc=f"Step {step_i + 1} / {total}")

    return pipe(
        image=image,
        prompt=prompt.strip() or "",
        variation_strength=float(variation_strength),
        ctrl_strength=float(ctrl_strength),
        ip_strength=float(ip_strength),
        flow_shift=float(flow_shift),
        guidance_scale=float(cfg),
        num_inference_steps=total,
        hed_thickness=int(hed_thickness),
        generator=gen,
        callback_on_step_end=_cb,
    ).images[0]


def on_image_upload(img):
    if img is None:
        return gr.update(), gr.update()
    w, h, _ = safeguard(img)
    return gr.update(value=w), gr.update(value=h)


_HERE = os.path.dirname(os.path.abspath(__file__))
CSS       = open(os.path.join(_HERE, "static/style.css")).read()
HERO_HTML = open(os.path.join(_HERE, "static/hero.html")).read()

LIGHTBOX_JS = """
() => {
  window.openLightbox = function(src) {
    var lb = document.getElementById('pxl-lightbox');
    var img = document.getElementById('pxl-lightbox-img');
    if (!lb || !img) return;
    img.src = src;
    lb.classList.add('open');
  };
  window.closeLightbox = function() {
    var lb = document.getElementById('pxl-lightbox');
    var img = document.getElementById('pxl-lightbox-img');
    if (lb) lb.classList.remove('open');
    if (img) img.src = '';
  };
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') window.closeLightbox();
  });
  document.addEventListener('click', function(e) {
    if (e.target.id === 'pxl-lightbox') window.closeLightbox();
  });
  document.addEventListener('click', function(e) {
    var img = e.target.closest('.output-image img');
    if (!img || !img.src) return;
    e.preventDefault(); e.stopPropagation();
    window.openLightbox(img.src);
  }, true);
}
"""


def _prompt_block():
    prompt   = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what you want…", value=_DEFAULT_PROMPT)
    negative = gr.Textbox(label="Negative Prompt", lines=2, placeholder="What to avoid…", value=_DEFAULT_NEGATIVE)
    return prompt, negative


def _prompt_block_optional():
    prompt = gr.Textbox(
        label="Prompt (optional — leave empty for pure reference-driven output)",
        lines=3,
        placeholder="gothic pale woman, dramatic rim lighting, graphic novel illustration…",
    )
    return prompt


def _seed_block():
    with gr.Row(elem_classes="seed-row"):
        seed = gr.Number(label="Seed", value=42, precision=0, elem_classes="seed-out", scale=4)
        dice = gr.Button("🎲", elem_classes="dice-btn", scale=1, size="sm", min_width=48)
    dice.click(fn=new_seed, outputs=[seed])
    return seed


def _settings_block(with_strength=False):
    with gr.Accordion("Generation settings", open=True):
        steps = gr.Slider(label="Steps",      minimum=45,  maximum=100,  step=1,   value=50)
        cfg   = gr.Slider(label="CFG scale",  minimum=1.0, maximum=12.0, step=0.5, value=4.5)
        shift = gr.Slider(label="Flow shift", minimum=1.0, maximum=10.0, step=0.5, value=4.0)
        strength = None
        if with_strength:
            strength = gr.Slider(label="Strength", minimum=0.05, maximum=1.0, step=0.05, value=0.8)
        with gr.Row():
            w = gr.Slider(label="Width",  minimum=256, maximum=1280, step=16, value=768)
            h = gr.Slider(label="Height", minimum=256, maximum=1280, step=16, value=1024)
    return steps, cfg, shift, strength, w, h


def _styled_settings_block():
    with gr.Accordion("Style settings", open=True):
        variation = gr.Slider(label="Variation strength  (0=copy · 1=full noise)", minimum=0.5, maximum=1.0, step=0.05, value=0.80)
        ctrl      = gr.Slider(label="ControlNet strength (scribble / HED edges)", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
        ip        = gr.Slider(label="IP-Adapter strength (SigLIP style pull)",    minimum=0.0, maximum=1.0, step=0.05, value=0.85)
    with gr.Accordion("Generation settings", open=True):
        steps     = gr.Slider(label="Steps",         minimum=20,  maximum=100,  step=1,   value=50)
        cfg       = gr.Slider(label="CFG scale",     minimum=1.0, maximum=6.0,  step=0.5, value=3.0)
        shift     = gr.Slider(label="Flow shift",    minimum=1.0, maximum=10.0, step=0.5, value=8.0)
        thickness = gr.Slider(label="HED thickness  (0=thin · 2=default · 4+=thick)", minimum=0, maximum=6, step=1, value=2)
    return variation, ctrl, ip, steps, cfg, shift, thickness


with gr.Blocks(title="PixelDiT Studio") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ── Tab 1: Text to Image ───────────────────────────────────────
        with gr.Tab("✦ Text to Image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes="panel"):
                    t2i_prompt, t2i_negative = _prompt_block()
                    t2i_seed = _seed_block()
                    t2i_steps, t2i_cfg, t2i_shift, _, t2i_w, t2i_h = _settings_block(with_strength=False)
                    t2i_btn = gr.Button("Generate", elem_classes="gen-btn", variant="primary")

                with gr.Column(scale=1, elem_classes="panel"):
                    t2i_out = gr.Image(label="Output", elem_classes="image-box output-image", interactive=False, buttons=["download"])

            t2i_btn.click(
                fn=generate_t2i,
                inputs=[t2i_prompt, t2i_negative, t2i_shift, t2i_seed, t2i_steps, t2i_cfg, t2i_w, t2i_h],
                outputs=[t2i_out],
            )

        # ── Tab 2: Image to Image ──────────────────────────────────────
        with gr.Tab("↻ Image to Image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes="panel"):
                    i2i_image = gr.Image(label="Reference Image", type="pil", elem_classes="image-box")
                    i2i_prompt, i2i_negative = _prompt_block()
                    i2i_seed = _seed_block()
                    i2i_steps, i2i_cfg, i2i_shift, i2i_strength, i2i_w, i2i_h = _settings_block(with_strength=True)
                    i2i_btn = gr.Button("Generate", elem_classes="gen-btn", variant="primary")

                with gr.Column(scale=1, elem_classes="panel"):
                    i2i_out = gr.Image(label="Output", elem_classes="image-box output-image", interactive=False, buttons=["download"])

            i2i_image.change(fn=on_image_upload, inputs=[i2i_image], outputs=[i2i_w, i2i_h])

            i2i_btn.click(
                fn=generate_i2i,
                inputs=[i2i_image, i2i_prompt, i2i_negative, i2i_strength, i2i_shift, i2i_seed, i2i_steps, i2i_cfg, i2i_w, i2i_h],
                outputs=[i2i_out],
            )

        # ── Tab 3: Styled (ControlNet + IP-Adapter) ────────────────────
        with gr.Tab("✦ Styled"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes="panel"):
                    sty_image  = gr.Image(label="Reference Image — drives structure (ControlNet) and style (IP-Adapter)", type="pil", elem_classes="image-box")
                    sty_prompt = _prompt_block_optional()
                    sty_seed   = _seed_block()
                    sty_variation, sty_ctrl, sty_ip, sty_steps, sty_cfg, sty_shift, sty_thickness = _styled_settings_block()
                    sty_btn    = gr.Button("Generate", elem_classes="gen-btn", variant="primary")

                with gr.Column(scale=1, elem_classes="panel"):
                    sty_out = gr.Image(label="Output", elem_classes="image-box output-image", interactive=False, buttons=["download"])

            sty_btn.click(
                fn=generate_styled,
                inputs=[sty_image, sty_prompt, sty_variation, sty_ctrl, sty_ip, sty_shift, sty_seed, sty_steps, sty_cfg, sty_thickness],
                outputs=[sty_out],
            )

demo.launch(server_name="10.147.18.150", share=False, css=CSS, js=LIGHTBOX_JS)
