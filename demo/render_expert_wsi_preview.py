"""Overlay a PathoGaze expert path on its coordinate-matched CAMELYON overview."""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def group_display_points(coords):
    groups = {}
    for index, coord in enumerate(coords, start=1):
        groups.setdefault(coord, []).append(index)
    return list(groups.items())


def render(case_id, participant, trajectories, preview_manifest, out_dir):
    slide = next(row for row in preview_manifest if row["experiment_id"] == case_id)
    if not slide["aspect_ratio_match"]:
        raise ValueError(f"Overview aspect ratio is not aligned for {case_id}")
    points = sorted((row for row in trajectories if row["case_id"] == case_id and row["participant_id"] == participant),
                    key=lambda row: int(row["step"]))
    if not points:
        raise ValueError(f"No expert trajectory for {case_id}/{participant}")
    source = Image.open(slide["overview_path"]).convert("RGB")
    width, height = slide["slide_dimensions_level0"]
    if abs(source.width / source.height - width / height) > 1e-6:
        raise ValueError("Overview and level-0 WSI aspect ratios differ")
    header = 74
    side_width = 270
    canvas = Image.new("RGB", (source.width + side_width, source.height + header), "white")
    canvas.paste(source, (0, header))
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("arial.ttf", 24)
        label = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        title = label = ImageFont.load_default()
    draw.text((18, 10), f"PathoGaze {case_id} / {slide['slide_id']} | expert {participant}", fill="#202a31", font=title)
    draw.text((18, 43), f"{slide['diagnosis']} / {slide['tumor_size']} | {len(points)} simplified viewport nodes | overview 1:{slide['overview_downsample']}", fill="#52616b", font=label)
    panel_x = source.width
    draw.line((panel_x, header, panel_x, canvas.height), fill="#c7d0d4", width=1)
    draw.text((panel_x + 12, header + 10), "Node / start time (s) / magnification", fill="#202a31", font=label)
    for index, row in enumerate(points, start=1):
        y = header + 34 + (index - 1) * 20
        draw.text((panel_x + 12, y), f"{index:02d}   {float(row['time_ms']) / 1000:.3f}s   {float(row['magnification']):.2f}x",
                  fill="#005f5f", font=label)
    coords = [(round(float(row["x_px"]) / width * (source.width - 1)),
               round(header + float(row["y_px"]) / height * (source.height - 1))) for row in points]
    for start, end in zip(coords, coords[1:]):
        draw.line((start, end), fill="#00a6a6", width=3)
        distance = math.dist(start, end)
        if distance > 5:
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            size = 10
            left = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
            right = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
            draw.polygon([end, left, right], fill="#00a6a6")
    # Repeated visits share a location; label every event without moving its coordinate.
    groups = group_display_points(coords)
    occupied = []
    for (x, y), indices in groups:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#007f7f", outline="white", width=2)
        text = ",".join(map(str, indices))
        bounds = draw.textbbox((0, 0), text, font=label)
        box_w, box_h = bounds[2] - bounds[0] + 12, 24
        candidates = [(x + 12, y - 30), (x + 12, y + 12),
                      (x - box_w - 12, y - 30), (x - box_w - 12, y + 12)]
        candidates.extend((x + 12, y + 12 + offset * 28) for offset in range(1, len(groups) + 1))
        for left, top in candidates:
            left = min(max(left, 2), source.width - box_w - 2)
            top = min(max(top, header + 2), canvas.height - box_h - 2)
            rect = (left, top, left + box_w, top + box_h)
            if not any(rect[0] < other[2] + 3 and rect[2] > other[0] - 3
                       and rect[1] < other[3] + 3 and rect[3] > other[1] - 3
                       for other in occupied):
                break
        else:
            raise ValueError(f"Cannot place event label for {case_id}: {indices}")
        occupied.append(rect)
        draw.line((x, y, left + box_w / 2, top + box_h / 2), fill="#007f7f", width=1)
        draw.rounded_rectangle(rect, radius=3, fill="#007f7f", outline="white", width=1)
        draw.text((left + box_w / 2, top + box_h / 2), text, anchor="mm", fill="white", font=label)
    note_y = header + 44 + len(points) * 20
    draw.text((panel_x + 12, note_y), f"{len(points)} nodes / {len(groups)} plotted locations", fill="#202a31", font=label)
    draw.text((panel_x + 12, note_y + 24), "Shared position may retain zoom/revisit.", fill="#52616b", font=label)
    if "source_event_count" in points[0]:
        count = sum(int(row["source_event_count"]) for row in points)
        draw.text((panel_x + 12, note_y + 48), f"{count} input fixations accounted for.", fill="#52616b", font=label)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{case_id}_{participant}_expert_on_wsi.png"
    canvas.save(target, optimize=True)
    audit = {
        "case_id": case_id,
        "participant_id": participant,
        "event_count": len(points),
        "unique_wsi_positions": len({(float(row["x_px"]), float(row["y_px"])) for row in points}),
        "display_groups": [{"overview_pixel": [x, y - header], "event_numbers": indices}
                           for (x, y), indices in groups],
        "events": [{"event_number": index, "step": int(row["step"]),
                    "time_ms": float(row["time_ms"]), "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]), "magnification": float(row["magnification"]),
                    "source_indices": json.loads(row.get("source_indices", "[]")),
                    "source_event_count": int(row.get("source_event_count", 1))}
                   for index, row in enumerate(points, start=1)],
    }
    assert sorted(index for _, indices in groups for index in indices) == list(range(1, len(points) + 1))
    target.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camelyon-root", type=Path, default=Path(r"D:\pathologyData\CAMELYON16"))
    parser.add_argument("--expert-csv", type=Path, default=Path(r"C:\Users\14910\Documents\ChatGPT\病理医学研究\trajectory_demo\runs\chakraborty_viewport\expert_trajectories.csv"))
    parser.add_argument("--participant", default="P1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    pilot = args.camelyon_root / "pathogaze_pilot"
    with args.expert_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    slides = json.loads((pilot / "preview_manifest.json").read_text(encoding="utf-8"))
    for case in ("C1", "C48", "C6"):
        print(render(case, args.participant, rows, slides, args.output_dir or pilot / "expert_overlays"), flush=True)


if __name__ == "__main__":
    main()
