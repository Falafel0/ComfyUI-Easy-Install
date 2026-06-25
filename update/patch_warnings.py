"""Re-apply warning suppression patches after ComfyUI update."""
import sys
from pathlib import Path

comfy_root = Path(__file__).resolve().parent.parent / "ComfyUI"

patches = [
    (comfy_root / "comfy" / "quant_ops.py",
     'logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")',
     'logging.info("You need pytorch with cu130 or higher to use optimized CUDA operations.")'),

    (comfy_root / "comfy" / "supported_models_base.py",
     'logging.warning("\\nWARNING, you accessed {} from the model config object which doesn\'t exist. Please fix your code.\\n".format(name))',
     'logging.debug("\\nWARNING, you accessed {} from the model config object which doesn\'t exist. Please fix your code.\\n".format(name))'),

    (comfy_root / "main.py",
     'logging.warning(\n            "Dynamic vram disabled with argument.',
     'logging.info(\n            "Dynamic vram disabled with argument.'),
]

count = 0
for filepath, old, new in patches:
    try:
        text = filepath.read_text(encoding="utf-8")
        if old in text:
            filepath.write_text(text.replace(old, new), encoding="utf-8")
            print(f"Patched: {filepath.name}")
            count += 1
        else:
            print(f"Already patched or not found: {filepath.name}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")

print(f"Done. {count} file(s) patched.")
