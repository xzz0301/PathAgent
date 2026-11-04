"""Configurable loading for the original models and the explicit local fallback."""
import importlib
import json
import sys
from pathlib import Path


def model_load_kwargs(args, role):
    import torch

    kwargs = {"device_map": getattr(args, f"{role}_device_map")}
    kwargs["torch_dtype"] = ("auto" if args.model_dtype == "auto"
                             else getattr(torch, args.model_dtype))
    if args.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    raw = getattr(args, f"{role}_max_memory")
    if raw:
        memory = json.loads(raw) if isinstance(raw, str) else raw
        kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v
                                for k, v in memory.items()}
    return kwargs


def load_models(args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
                              Qwen2_5_VLForConditionalGeneration)

    for role, device in (("executor", args.executor_device_map),
                         ("perceptor", args.perceptor_device_map),
                         ("navigator", args.plip_device)):
        if device.startswith("cuda:"):
            index = int(device.split(":")[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(f"{role}: requested {device}, but it is not available")
            if role != "navigator" and args.model_dtype == "bfloat16":
                with torch.cuda.device(index):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError(f"{role}: {device} does not support bfloat16")

    library = str(Path(args.plip_lib_path).resolve())
    if library not in sys.path:
        sys.path.insert(0, library)
    PLIP = importlib.import_module("plip").PLIP
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_ckpt, **model_load_kwargs(args, "executor"))
    plip = PLIP(args.plip_ckpt)
    device = args.plip_device
    if device != "auto":
        plip.model = plip.model.to(device)
        plip.device = device
    perceptor = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.patho_r1_ckpt, **model_load_kwargs(args, "perceptor"))
    processor = AutoProcessor.from_pretrained(args.patho_r1_ckpt)
    print(f"Models loaded | profile={args.profile_name} | quantization={args.quantization}")
    return tokenizer, model, plip, perceptor, processor


def _device_map_metadata(model):
    mapping = getattr(model, "hf_device_map", None)
    if isinstance(mapping, dict):
        return {str(name): device if isinstance(device, (str, int)) else str(device)
                for name, device in mapping.items()}
    return str(getattr(model, "device", "unknown"))


def runtime_metadata(args, model, perceptor, plip):
    import torch

    return {
        "profile": args.profile_name, "seed": args.seed,
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                  "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(i)}
                 for i in range(torch.cuda.device_count())],
        "executor_device_map": _device_map_metadata(model),
        "perceptor_device_map": _device_map_metadata(perceptor),
        "navigator_device": str(plip.device),
        "weights": {"executor": args.qwen_ckpt, "perceptor": args.patho_r1_ckpt,
                    "navigator": args.plip_ckpt},
        "quantization": args.quantization, "requested_dtype": args.model_dtype,
        "executor_dtype": str(getattr(model, "dtype", "unknown")),
        "perceptor_dtype": str(getattr(perceptor, "dtype", "unknown")),
        "minimum_observation_rounds": args.min_observation_rounds,
        "allow_empty_descriptions": args.allow_empty_descriptions,
        "image_backend":getattr(args,'image_backend','patch'),
        "zoom_source": "original_wsi" if getattr(args,'image_backend','patch')=='wsi' else "crop_existing_patch_not_higher_resolution_wsi",
    }
