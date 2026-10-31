"""Export PathAgent observations as a trace table and a sparse tile overview."""
import csv
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class TraceRecorder:
    def __init__(self, output_dir, result_id):
        self.path = Path(output_dir) / f"{result_id}_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.points = []

    def event(self, event_type, **data):
        event = {"event": event_type, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **data}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def start_observation(self, **data):
        row = {"step": len(self.points) + 1, "status": "started", **data}
        self.points.append(row)
        self.event("observation_started", **row)
        return row, time.perf_counter()

    def complete_observation(self, started, description):
        row, start_time = started
        row.update(status="completed", inference_ms=round((time.perf_counter() - start_time) * 1000, 2),
                   description=description)
        self.event("observation_completed", **row)


def split_zoom_patch(patch_path, target_magnification, base_magnification, coordinate_scale):
    factor = float(target_magnification) / base_magnification
    if not math.isfinite(factor) or factor < 1 or not factor.is_integer():
        raise ValueError("Crop magnification must be an integer multiple of native tile magnification")
    factor = int(factor)
    x, y = map(int, Path(patch_path).stem.split("_"))
    with Image.open(patch_path) as source:
        image = source.convert("RGB")
    if factor > min(image.size):
        raise ValueError("Crop would have zero pixels")
    patches = []
    for row in range(factor):
        for col in range(factor):
            left, top = col * image.width // factor, row * image.height // factor
            right, bottom = (col + 1) * image.width // factor, (row + 1) * image.height // factor
            patches.append((image.crop((left, top, right, bottom)),
                            (x + left * coordinate_scale, y + top * coordinate_scale)))
    return patches


def _font(size=14):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_tiles(patch_dir):
    tiles = []
    for path in sorted(Path(patch_dir).glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            x_str, y_str = path.stem.split("_")
            with Image.open(path) as image:
                tiles.append((int(x_str), int(y_str), image.convert("RGB").copy()))
        except (ValueError, OSError):
            continue
    return tiles


def _arrow(draw, a, b):
    length = math.dist(a, b)
    if length < 2:
        return
    dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    end = (b[0] - dx * min(14, length / 3), b[1] - dy * min(14, length / 3))
    draw.line([a, end], fill="#ce3628", width=3)
    draw.polygon([end, (end[0] - dx * 10 + dy * 5, end[1] - dy * 10 - dx * 5),
                  (end[0] - dx * 10 - dy * 5, end[1] - dy * 10 + dx * 5)], fill="#ce3628")


def _panel(tiles, trajectory, bounds, size, coordinate_scale, show_steps=True):
    x0, y0, x1, y1 = bounds
    scale = min((size[0] - 36) / (x1 - x0), (size[1] - 36) / (y1 - y0))
    offset_x = (size[0] - (x1 - x0) * scale) / 2
    offset_y = (size[1] - (y1 - y0) * scale) / 2
    transform = lambda x, y: (offset_x + (x - x0) * scale, offset_y + (y - y0) * scale)
    panel = Image.new("RGB", size, "#eef0f1")
    draw = ImageDraw.Draw(panel)
    for x, y, image in tiles:
        w, h = image.width * coordinate_scale, image.height * coordinate_scale
        if x + w < x0 or x > x1 or y + h < y0 or y > y1:
            continue
        pos = transform(x, y)
        thumbnail = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        panel.paste(thumbnail, (round(pos[0]), round(pos[1])))
    points = []
    for row in trajectory:
        w, h = row["width"] * coordinate_scale, row["height"] * coordinate_scale
        left, top = transform(row["x"], row["y"])
        right, bottom = transform(row["x"] + w, row["y"] + h)
        draw.rectangle((left, top, right, bottom), outline="#ce3628", width=2)
        points.append(transform(row["x"] + w / 2, row["y"] + h / 2))
    if show_steps:
        for a, b in zip(points, points[1:]):
            _arrow(draw, a, b)
        for row, (x, y) in zip(trajectory, points):
            draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill="#ce3628", outline="white", width=2)
            draw.text((x, y), str(row["step"]), anchor="mm", fill="white", font=_font(12))
            zoom = row.get("magnification")
            zoom_label = f"{zoom:g}X" if isinstance(zoom, (int, float)) else "?X"
            tag = f"{row['step']} · {zoom_label}"
            tag_box = draw.textbbox((0, 0), tag, font=_font(13))
            tag_w = tag_box[2] - tag_box[0] + 10
            tag_h = tag_box[3] - tag_box[1] + 6
            label_x = min(x + 18, size[0] - tag_w - 3)
            label_y = max(2, y - tag_h // 2)
            draw.rounded_rectangle((label_x, label_y, label_x + tag_w, label_y + tag_h), radius=3,
                                   fill="#ffffff", outline="#ce3628", width=1)
            draw.text((label_x + 5, label_y + 3), tag, fill="#922519", font=_font(13))
    return panel


def export_trajectory_artifacts(trajectory, patch_dir, output_dir, case_id, result_id, geometry=None):
    geometry = geometry or {"coordinate_scale": 1, "coordinate_space": "patch_filename_unverified", "level0_verified": False}
    coordinate_scale = float(geometry["coordinate_scale"])
    if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("Invalid coordinate scale")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{result_id}_trajectory"
    csv_path = os.path.join(output_dir, stem + ".csv")
    json_path = os.path.join(output_dir, stem + ".json")
    image_path = os.path.join(output_dir, stem + ".png")

    rows = []
    for row in trajectory:
        center_x = row["x"] + row["width"] * coordinate_scale / 2
        center_y = row["y"] + row["height"] * coordinate_scale / 2
        rows.append({**row, "case_id": case_id, "agent_id": result_id,
                     "x_px": center_x, "y_px": center_y, "center_x": center_x, "center_y": center_y,
                     "coordinate_scale": coordinate_scale, "coordinate_space": geometry["coordinate_space"],
                     "level0_verified": geometry.get("level0_verified", False)})
    fields = ["case_id", "agent_id", "step", "round", "action", "patch_name", "status", "x", "y", "x_px", "y_px",
              "center_x", "center_y", "width", "height", "magnification", "inference_ms",
              "coordinate_scale", "coordinate_space", "level0_verified"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"case_id": case_id, "geometry": geometry, "trajectory": rows}, handle, ensure_ascii=False, indent=2)

    tiles = _load_tiles(patch_dir)
    if not tiles:
        return {"csv": csv_path, "json": json_path, "image": None}

    bounds = (min(x for x, _, _ in tiles), min(y for _, y, _ in tiles),
              max(x + img.width * coordinate_scale for x, _, img in tiles),
              max(y + img.height * coordinate_scale for _, y, img in tiles))
    canvas = Image.new("RGB", (1320, 930), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), f"{case_id} | PathAgent observation trajectory", fill="#222222", font=_font(24))
    subtitle = ("CAMELYON16 independent candidate tile reconstruction | origins in level-0 pixels"
                if geometry.get("coordinate_space") == "CAMELYON16_level0" else
                "SICAP tile reconstruction | origin coordinates inferred from filenames | no original WSI")
    draw.text((24, 51), subtitle, fill="#555555", font=_font(15))
    draw.text((24, 94), "All available tiles", fill="#222222", font=_font(18))
    draw.text((340, 94), "Observed region (detail)", fill="#222222", font=_font(18))
    canvas.paste(_panel(tiles, trajectory, bounds, (280, 750), coordinate_scale, show_steps=False), (24, 126))
    focus = bounds
    if trajectory:
        padding = max(max(row["width"], row["height"]) * coordinate_scale for row in trajectory) * 0.4
        focus = (min(row["x"] for row in trajectory) - padding,
                 min(row["y"] for row in trajectory) - padding,
                 max(row["x"] + row["width"] * coordinate_scale for row in trajectory) + padding,
                 max(row["y"] + row["height"] * coordinate_scale for row in trajectory) + padding)
    canvas.paste(_panel(tiles, trajectory, focus, (956, 750), coordinate_scale), (340, 126))
    draw.text((24, 897), f"{len(rows)} observations | arrows follow Perceptor call order | coordinate scale: {coordinate_scale:g} origin units / tile pixel",
              fill="#555555", font=_font(15))
    canvas.save(image_path, optimize=True)
    return {"csv": csv_path, "json": json_path, "image": image_path}


def main():
    parser = argparse.ArgumentParser(description="Re-render a completed run without loading models")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    paths = export_trajectory_artifacts(result["trajectory"], args.patch_dir, args.result.parent,
                                       result["long_id"], args.result.stem, result.get("geometry"))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
