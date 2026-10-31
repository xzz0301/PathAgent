"""Validate SICAP tile stride and write an explicit coordinate convention."""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image


def prepare_geometry(archive, patient, case_id, patch_dir):
    pattern = re.compile(r"Block_Region_(\d+)_(\d+)_(\d+)_xini_(\d+)_yini_(\d+)\.jpg$")
    origins = {}
    sizes = set()
    with zipfile.ZipFile(archive) as zf:
        readme = zf.read("SICAPv2/readme.txt").decode("utf-8")
        if "10X magnification" not in readme or "overlap of 50%" not in readme:
            raise ValueError("Cannot establish the 10x/50%-overlap convention from archive readme")
        for name in zf.namelist():
            if not name.startswith(f"SICAPv2/images/{patient}_"):
                continue
            match = pattern.search(name)
            if not match:
                continue
            region, row, col, x, y = map(int, match.groups())
            with Image.open(io.BytesIO(zf.read(name))) as image:
                sizes.add(image.size)
            origins[(region, row, col)] = (x, y)
    if sizes != {(512, 512)} or not origins:
        raise ValueError(f"Unexpected tile dimensions or empty patient: {sizes}")
    scale_checks = []
    for (region, row, col), (x, y) in origins.items():
        right = origins.get((region, row, col + 1))
        below = origins.get((region, row + 1, col))
        if right is not None:
            if right[1] != y:
                raise ValueError("Horizontal adjacency changes y")
            scale_checks.append((right[0] - x) / (512 * 0.5))
        if below is not None:
            if below[0] != x:
                raise ValueError("Vertical adjacency changes x")
            scale_checks.append((below[1] - y) / (512 * 0.5))
    if not scale_checks or set(scale_checks) != {4.0}:
        raise ValueError(f"Inconsistent inferred pixel-to-origin scale: {set(scale_checks)}")
    archive_origins = set(origins.values())
    local_origins = set()
    for path in Path(patch_dir).glob("*.png"):
        x, y = map(int, path.stem.split("_"))
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise ValueError(f"Unexpected local tile size: {path}")
        local_origins.add((x, y))
    if local_origins != archive_origins:
        raise ValueError("Local patches do not match this archive patient's coordinates")
    return {case_id: {
        "patient_id": patient,
        "coordinate_scale": 4.0,
        "base_magnification": 10,
        "coordinate_space": "sicap_filename_inferred",
        "level0_verified": False,
        "tile_count": len(origins),
        "adjacent_pair_checks": len(scale_checks),
        "evidence": "Archive readme: 512px, 10X, 50% overlap. All adjacent tile origin strides: 1024 units. Scale = 1024/(512*0.5) = 4.",
        "source_archive": str(Path(archive).resolve()),
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--patient", default="16B0001851")
    parser.add_argument("--case-id", default="DX1_SICAP001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = prepare_geometry(args.archive, args.patient, args.case_id, args.patch_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    main()
