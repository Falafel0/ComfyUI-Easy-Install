"""Re-apply patches after ComfyUI update: warning suppression + int8_tensorwise support."""
import sys
from pathlib import Path

comfy_root = Path(__file__).resolve().parent.parent / "ComfyUI"

# Format: (filepath, old_string, new_string)
patches = [
    # ── Warning suppression ────────────────────────────────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     'logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")',
     'logging.info("You need pytorch with cu130 or higher to use optimized CUDA operations.")'),

    (comfy_root / "comfy" / "supported_models_base.py",
     'logging.warning("\\nWARNING, you accessed {} from the model config object which doesn\'t exist. Please fix your code.\\n".format(name))',
     'logging.debug("\\nWARNING, you accessed {} from the model config object which doesn\'t exist. Please fix your code.\\n".format(name))'),

    (comfy_root / "main.py",
     'logging.warning(\n            "Dynamic vram disabled with argument.',
     'logging.info(\n            "Dynamic vram disabled with argument.'),

    # ── int8_tensorwise: quant_ops.py import ───────────────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     '        TensorCoreNVFP4Layout as _CKNvfp4Layout,\n        register_layout_op,',
     '        TensorCoreNVFP4Layout as _CKNvfp4Layout,\n        TensorWiseINT8Layout as _CKTensorWiseINT8Layout,\n        register_layout_op,'),

    # ── int8_tensorwise: quant_ops.py except stub ──────────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     '    class _CKNvfp4Layout:\n        pass\n\n    def register_layout_class(name, cls):',
     '    class _CKNvfp4Layout:\n        pass\n\n    class _CKTensorWiseINT8Layout:\n        pass\n\n    def register_layout_class(name, cls):'),

    # ── int8_tensorwise: quant_ops.py alias ────────────────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     'if not _CK_MXFP8_AVAILABLE:\n    class _CKMxfp8Layout:\n        pass\n\nimport comfy.float',
     'if not _CK_MXFP8_AVAILABLE:\n    class _CKMxfp8Layout:\n        pass\n\nTensorWiseINT8Layout = _CKTensorWiseINT8Layout\n\nimport comfy.float'),

    # ── int8_tensorwise: quant_ops.py layout registration ──────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     'if _CK_MXFP8_AVAILABLE:\n    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)\n\nQUANT_ALGOS = {',
     'if _CK_MXFP8_AVAILABLE:\n    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)\nregister_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)\n\nQUANT_ALGOS = {'),

    # ── int8_tensorwise: quant_ops.py QUANT_ALGOS entry ────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     'QUANT_ALGOS["mxfp8"] = {\n        "storage_t": torch.float8_e4m3fn,\n        "parameters": {"weight_scale", "input_scale"},\n        "comfy_tensor_layout": "TensorCoreMXFP8Layout",\n        "group_size": 32,\n    }\n\n\n# ===',
     'QUANT_ALGOS["mxfp8"] = {\n        "storage_t": torch.float8_e4m3fn,\n        "parameters": {"weight_scale", "input_scale"},\n        "comfy_tensor_layout": "TensorCoreMXFP8Layout",\n        "group_size": 32,\n    }\n\nQUANT_ALGOS["int8_tensorwise"] = {\n    "storage_t": torch.int8,\n    "parameters": {"weight_scale"},\n    "comfy_tensor_layout": "TensorWiseINT8Layout",\n    "quantize_input": False,\n}\n\n\n# ==='),

    # ── int8_tensorwise: quant_ops.py __all__ ──────────────────────────────
    (comfy_root / "comfy" / "quant_ops.py",
     '    "TensorCoreNVFP4Layout",\n    "QUANT_ALGOS",',
     '    "TensorCoreNVFP4Layout",\n    "TensorWiseINT8Layout",\n    "QUANT_ALGOS",'),

    # ── int8_tensorwise: ops.py scale handling ─────────────────────────────
    (comfy_root / "comfy" / "ops.py",
     '            scales = {"scale": ts, "block_scale": bs}\n        else:\n            raise ValueError(f"Unsupported quantization format: {module.quant_format}")',
     '            scales = {"scale": ts, "block_scale": bs}\n        elif module.quant_format == "int8_tensorwise":\n            scale = pop_scale("weight_scale")\n            if scale is None:\n                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")\n            scales = {"scale": scale}\n            params_conf = layer_conf.get("params", {})\n            if not isinstance(params_conf, dict):\n                params_conf = {}\n            if layer_conf.get("convrot", params_conf.get("convrot", False)):\n                scales["convrot"] = True\n                scales["convrot_groupsize"] = int(\n                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))\n                )\n        else:\n            raise ValueError(f"Unsupported quantization format: {module.quant_format}")'),
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
