"""Read downloaded WSI pyramids and prepare verified low-resolution overviews."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camelyon-root", type=Path, default=Path(r"D:\pathologyData\CAMELYON16"))
    parser.add_argument("--pathogaze-root", type=Path, default=Path(r"D:\pathologyData\PathoGaze1.0"))
    args = parser.parse_args()
    manifest = json.loads((args.camelyon_root / "pathogaze_pilot_manifest.json").read_text(encoding="utf-8"))
    with (args.pathogaze_root / "ImageMetrics.csv").open(encoding="utf-8-sig") as handle:
        metrics = {row["experimentImageName"]: row for row in csv.DictReader(handle)}
    out = args.camelyon_root / "pathogaze_pilot"
    out.mkdir(parents=True, exist_ok=True)
    report, thumbnails = [], []
    for row in manifest["cases"]:
        path = args.camelyon_root / "images" / row["filename"]
        expected = metrics[row["experiment_id"]]
        with openslide.OpenSlide(str(path)) as slide:
            dims = (int(float(expected["width(px)"])), int(float(expected["height(px)"])))
            if slide.dimensions != dims:
                raise ValueError(f"{path.name}: WSI dimensions {slide.dimensions} differ from PathoGaze {dims}")
            thumbnail = slide.get_thumbnail((1600, 1600)).convert("RGB")
            thumb_path = out / f"{row['experiment_id']}_{row['slide_id']}_overview.png"
            thumbnail.save(thumb_path)
            pixels = np.asarray(thumbnail)
            if np.std(pixels.astype(float)) < 1:
                raise ValueError(f"Blank thumbnail for {path.name}")
            # Decode a native-resolution tissue patch as a separate TIFF codec check.
            tissue = (pixels.max(axis=2) - pixels.min(axis=2) > 20) & (pixels.mean(axis=2) < 235)
            locations = np.argwhere(tissue)
            if not len(locations):
                raise ValueError(f"No tissue detected in {path.name}")
            ty, tx = locations[len(locations) // 2]
            origin_x = min(max(0, round(tx / thumbnail.width * dims[0]) - 128), dims[0] - 256)
            origin_y = min(max(0, round(ty / thumbnail.height * dims[1]) - 128), dims[1] - 256)
            patch_path = out / f"{row['experiment_id']}_native_patch.png"
            slide.read_region((origin_x, origin_y), 0, (256, 256)).convert("RGB").save(patch_path)
            inspection = {**row, "local_path": str(path), "dimension_match": True,
                          "slide_dimensions": list(slide.dimensions),
                          "level_dimensions": [list(d) for d in slide.level_dimensions],
                          "level_downsamples": list(slide.level_downsamples),
                          "objective_power": slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER),
                          "mpp_x": slide.properties.get(openslide.PROPERTY_NAME_MPP_X),
                          "mpp_y": slide.properties.get(openslide.PROPERTY_NAME_MPP_Y),
                          "diagnosis": expected["diagnosis"], "tumor_size": expected["tumorSize"],
                          "overview": str(thumb_path), "native_patch": str(patch_path),
                          "native_patch_origin": [origin_x, origin_y]}
            report.append(inspection)
            thumbnails.append((inspection, thumbnail))
            print(f"{row['experiment_id']} {path.name}: {dims}, {slide.level_count} levels, native patch decoded", flush=True)
    (out / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    gallery = Image.new("RGB", (1530, 790), "white")
    draw = ImageDraw.Draw(gallery)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, (row, thumbnail) in enumerate(thumbnails):
        left = index * 510 + 12
        draw.text((left, 16), f"{row['experiment_id']} | {row['slide_id']} | {row['tumor_size']}", fill="#222222", font=font)
        thumbnail.thumbnail((486, 705), Image.Resampling.LANCZOS)
        gallery.paste(thumbnail, (left + (486 - thumbnail.width) // 2, 62 + (705 - thumbnail.height) // 2))
    gallery.save(out / "pilot_overviews.png")


if __name__ == "__main__":
    main()
