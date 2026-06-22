"""
Install PixelDiT pipeline into the active venv's diffusers package.
Source of truth: diffusers_patch/src/diffusers/pipelines/pixeldit/

Run once after installing diffusers — subsequent runs happen automatically
via sitecustomize.py installed into the venv:
    python scripts/setup_diffusers_pixeldit.py
"""

import os
import re
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PIPE = os.path.join(PROJECT_ROOT, "diffusers_patch", "src", "diffusers", "pipelines", "pixeldit")

# Full list of names exported at the diffusers.pipelines level.
# Keep in sync with diffusers_patch/.../pixeldit/__init__.py
_PIPELINE_EXPORTS = [
    "PixelDiTPipeline",
    "PixelDiTImg2ImgPipeline",
    "PixelDiTPipelineOutput",
    "PixelDiTModel",
    "PixelDiTJointAttnProcessor",
    "QwenEncoder",
]

# Names added to the top-level diffusers/__init__.py, split by kind so we
# anchor each next to an existing name of the same kind.
_TOP_LEVEL_MODELS    = ["PixelDiTModel"]                             # anchor: FluxTransformer2DModel
_TOP_LEVEL_PIPELINES = ["PixelDiTPipeline", "PixelDiTImg2ImgPipeline"]  # anchor: DiTPipeline


def get_diffusers_path():
    import diffusers
    return os.path.dirname(diffusers.__file__)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, txt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def is_patched(D):
    """Return True if diffusers/__init__.py already exports PixelDiTPipeline."""
    txt = _read(os.path.join(D, "__init__.py"))
    return "PixelDiTPipeline" in txt


def install_pipeline_folder(D):
    dst = os.path.join(D, "pipelines", "pixeldit")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC_PIPE, dst)
    print("[1] Installed pipelines/pixeldit/")


def register_in_pipelines_init(D):
    path = os.path.join(D, "pipelines", "__init__.py")
    txt = _read(path)

    exports_str = ", ".join(f'"{e}"' for e in _PIPELINE_EXPORTS)
    imports_str = ", ".join(_PIPELINE_EXPORTS)

    new_structure_line = f'    _import_structure["pixeldit"] = [{exports_str}]'
    new_import_line    = f"        from .pixeldit import {imports_str}"

    if '_import_structure["pixeldit"]' in txt:
        txt = re.sub(
            r'    _import_structure\["pixeldit"\] = \[.*?\]',
            new_structure_line,
            txt,
        )
        print("[2] Updated _import_structure[pixeldit] in pipelines/__init__.py")
    else:
        txt = txt.replace(
            '    _import_structure["stable_diffusion_3"] = [',
            new_structure_line + '\n    _import_structure["stable_diffusion_3"] = [',
        )
        print("[2] Registered _import_structure[pixeldit] in pipelines/__init__.py")

    if "from .pixeldit import" in txt:
        txt = re.sub(
            r"        from \.pixeldit import .*",
            new_import_line,
            txt,
        )
        print("[2b] Updated 'from .pixeldit import' in pipelines/__init__.py")
    else:
        txt = txt.replace(
            "        from .stable_diffusion_3 import (",
            new_import_line + "\n        from .stable_diffusion_3 import (",
        )
        print("[2b] Added 'from .pixeldit import' in pipelines/__init__.py")

    _write(path, txt)


def register_in_diffusers_init(D):
    path = os.path.join(D, "__init__.py")
    txt = _read(path)

    # Pipelines: add to _import_structure pipelines list (near DiTPipeline) and
    # to the from .pipelines import (...) eager block (near DiTPipeline,).
    for name in _TOP_LEVEL_PIPELINES + _TOP_LEVEL_MODELS:
        if f'"{name}"' in txt and f"\n    {name}," in txt:
            print(f"[3] {name} already in diffusers/__init__.py")
            continue
        # Lazy loader string list — pipelines section
        if f'"{name}"' not in txt:
            txt = txt.replace(
                '"DiTPipeline",',
                f'"DiTPipeline",\n            "{name}",',
            )
        # Eager import block — DiTPipeline, anchor
        if f"\n            {name}," not in txt:
            txt = txt.replace(
                "            DiTPipeline,",
                f"            DiTPipeline,\n            {name},",
            )
        # from .pipelines import ( block
        if f"\n    {name}," not in txt:
            txt = txt.replace(
                "from .pipelines import (",
                f"from .pipelines import (\n    {name},",
            )
        print(f"[3] Registered {name} in diffusers/__init__.py")

    _write(path, txt)


def install_sitecustomize():
    """Install sitecustomize.py into the venv so patching re-runs if diffusers is updated."""
    import site
    site_pkgs = site.getsitepackages()
    if not site_pkgs:
        print("[4] Could not locate site-packages — skipping sitecustomize install")
        return

    script_abs = os.path.abspath(__file__)
    content = f"""\
# Auto-patch diffusers with PixelDiT — managed by setup_diffusers_pixeldit.py
def _ensure_pixeldit_patched():
    import os
    try:
        import diffusers
        D = os.path.dirname(diffusers.__file__)
    except ImportError:
        return
    if "PixelDiTPipeline" in open(os.path.join(D, "__init__.py")).read():
        return  # already patched
    import subprocess, sys
    subprocess.run([sys.executable, {script_abs!r}], check=True)

_ensure_pixeldit_patched()
del _ensure_pixeldit_patched
"""
    dst = os.path.join(site_pkgs[0], "sitecustomize.py")
    _write(dst, content)
    print(f"[4] Installed sitecustomize.py → {dst}")


def main():
    if not os.path.exists(SRC_PIPE):
        print(f"ERROR: source not found: {SRC_PIPE}")
        sys.exit(1)

    D = get_diffusers_path()
    print(f"Diffusers: {D}")

    install_pipeline_folder(D)
    register_in_pipelines_init(D)
    register_in_diffusers_init(D)
    install_sitecustomize()

    print("\nDone!")
    print('Test: python -c "from diffusers import PixelDiTPipeline, PixelDiTModel; print(\'OK\')"')


if __name__ == "__main__":
    main()
