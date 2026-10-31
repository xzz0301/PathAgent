"""Read-only preflight; never substitutes missing models or generates inputs."""
import json
import math
from pathlib import Path


def checkpoint_report(path):
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    indexes = list(root.glob("*.index.json"))
    shards = set()
    for index in indexes:
        data = json.loads(index.read_text(encoding="utf-8"))
        shards.update(data.get("weight_map", {}).values())
    if not shards:
        shards = {p.name for pattern in ("*.safetensors", "pytorch_model*.bin")
                  for p in root.glob(pattern)}
    if not shards:
        raise ValueError(f"No model weights found in {root}")
    for name in shards:
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Missing/invalid weight shard: {target}")
    return {"path": str(root), "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "weight_files": len(shards), "weight_bytes": sum((root / n).stat().st_size for n in shards)}


def validate_inputs(args):
    for name in ("plip_lib_path", "feature_dir", "patch_root"):
        if not Path(getattr(args, name)).is_dir():
            raise FileNotFoundError(f"{name}: {getattr(args, name)}")
    if not (Path(args.plip_lib_path) / "plip.py").is_file():
        raise FileNotFoundError("PLIP library must contain plip.py")
    reports = {role: checkpoint_report(path) for role, path in (
        ("executor", args.qwen_ckpt), ("perceptor", args.patho_r1_ckpt), ("navigator", args.plip_ckpt))}
    if args.profile_name == "server_original":
        perceptor = reports["perceptor"]
        if (perceptor["model_type"], perceptor["hidden_size"], perceptor["num_hidden_layers"]) != ("qwen2_5_vl", 3584, 28):
            raise ValueError("server_original requires the Patho-R1-7B architecture; a local 3B substitute was supplied")
    for root in (Path(args.qwen_ckpt), Path(args.patho_r1_ckpt)):
        for filename in ("tokenizer_config.json", "tokenizer.json"):
            if not (root / filename).is_file():
                raise FileNotFoundError(root / filename)
    for root in (Path(args.patho_r1_ckpt), Path(args.plip_ckpt)):
        if not (root / "preprocessor_config.json").is_file():
            raise FileNotFoundError(root / "preprocessor_config.json")
    from data_processing.utils import load_all_vqa_pairs
    descriptions = json.loads(Path(args.descriptions_file).read_text(encoding="utf-8"))
    pairs = load_all_vqa_pairs(args.questions_file, dataset_name=args.dataset_name, image_dir=args.patch_root)
    if not pairs:
        raise ValueError("No questions found")
    geometry = json.loads(Path(args.geometry_file).read_text(encoding="utf-8")) if args.geometry_file else {}
    cases = []
    for pair in pairs:
        key = pair["long_id"]
        matches = [key] if key in descriptions else [name for name in descriptions if name.startswith(key)]
        if len(matches) != 1:
            raise ValueError(f"Case {key}: expected one description key, found {matches}")
        key = matches[0]
        desc = descriptions[key]
        if not desc:
            raise ValueError(f"Case {key}: no candidate patches")
        if any(not isinstance(value, str) for value in desc.values()):
            raise ValueError(f"Case {key}: generic descriptions must be strings")
        empty = sum(not value.strip() for value in desc.values())
        if empty and not args.allow_empty_descriptions:
            raise ValueError(f"Case {key}: {empty} empty generic descriptions; generate Quilt-LLaVA descriptions first, or explicitly enable --allow-empty-descriptions for a smoke test")
        if args.geometry_file:
            if key not in geometry:
                raise ValueError(f"Missing geometry for case {key}")
            for field in ("coordinate_scale", "base_magnification"):
                value = float(geometry[key][field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Case {key}: invalid {field}")
        features = 0
        for patch in desc:
            from data_processing.utils import get_patch_fullpath
            if not Path(get_patch_fullpath(args.patch_root, key, patch)).is_file():
                raise FileNotFoundError(Path(args.patch_root) / key / patch)
            features += (Path(args.feature_dir) / key / (Path(patch).stem + ".npy")).is_file()
        if not features:
            raise ValueError(f"Case {key}: no cached PLIP features")
        cases.append({"case": key, "patches": len(desc), "features": features,
                      "empty_generic_descriptions": empty})
    return {"profile": args.profile_name, "models": reports, "cases": cases,
            "warnings": ["Existing-patch cropping is used for zoom; no higher-resolution WSI read is performed."]}
