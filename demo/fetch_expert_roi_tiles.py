"""Fetch native-resolution CAMELYON tiles intersecting selected expert paths."""
import argparse
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .fetch_camelyon_previews import fetch_range, remote_directories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\pathologyData"))
    parser.add_argument("--participant", default="P1")
    parser.add_argument("--cases", nargs="+", default=["C1", "C48", "C6"])
    args = parser.parse_args()
    cam_root = args.data_root / "CAMELYON16"
    pathogaze = args.data_root / "PathoGaze1.0"
    pilot = cam_root / "pathogaze_pilot"
    pilot_manifest = json.loads((cam_root / "pathogaze_pilot_manifest.json").read_text(encoding="utf-8"))
    case_manifest = {row["experiment_id"]: row for row in pilot_manifest["cases"]}
    expert_path = Path(r"C:\Users\14910\Documents\ChatGPT\病理医学研究\trajectory_demo\runs\chakraborty_viewport\expert_trajectories.csv")
    selected = {}
    with expert_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = row["case_id"]
            if case not in args.cases or row["participant_id"] != args.participant:
                continue
            meta = case_manifest[case]
            tile_w = tile_h = 512
            slide_w, slide_h = meta["slide_width"], meta["slide_height"]
            x_px, y_px = float(row["x_px"]), float(row["y_px"])
            col = min(max(int(x_px // tile_w), 0), (slide_w - 1) // tile_w)
            tile_row = min(max(int(y_px // tile_h), 0), (slide_h - 1) // tile_h)
            x0, y0 = col * tile_w, tile_row * tile_h
            key = (case, x0, y0)
            selected.setdefault(key, []).append({"step": int(row["step"]), "time_ms": float(row["time_ms"]),
                                                 "expert_center_x": x_px, "expert_center_y": y_px,
                                                 "magnification": float(row["magnification"])})
    grouped = {}
    for (case, x, y), visits in selected.items():
        grouped.setdefault(case, []).append((x, y, visits))

    def fetch_case(case):
        item = case_manifest[case]
        url = f"{pilot_manifest['source']}/{item['filename']}"
        width, height = item["slide_width"], item["slide_height"]
        tags, info = remote_directories(url, (width, height))
        offsets = tags[324] if isinstance(tags[324], tuple) else (tags[324],)
        lengths = tags[325] if isinstance(tags[325], tuple) else (tags[325],)
        jpeg_tables = bytes(tags[347]) if isinstance(tags.get(347), tuple) else tags.get(347, b"")
        if info["tile_width"] != 512 or info["tile_height"] != 512:
            raise ValueError(f"Unexpected native tile shape: {info}")
        across = (width + 511) // 512
        tile_jobs = []
        for x, y, visits in grouped[case]:
            index = (y // 512) * across + (x // 512)
            tile_jobs.append((index, x, y, visits, int(offsets[index]), int(lengths[index])))
        lo = min(job[4] for job in tile_jobs)
        hi = max(job[4] + job[5] for job in tile_jobs)
        if hi - lo <= 64 * 1024 * 1024:
            blob = fetch_range(url, lo, hi - 1)
            fetch = lambda offset, length: blob[offset - lo:offset - lo + length]
        else:
            fetch = lambda offset, length: fetch_range(url, offset, offset + length - 1)
        out_dir = pilot / "roi_tiles" / case
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, x, y, visits, offset, length in tile_jobs:
            tile_data = fetch(offset, length)
            if jpeg_tables and tile_data.startswith(b"\xff\xd8"):
                table_data = jpeg_tables[2:-2] if jpeg_tables.endswith(b"\xff\xd9") else jpeg_tables[2:]
                tile_data = tile_data[:2] + table_data + tile_data[2:]
            with Image.open(io.BytesIO(tile_data)) as image:
                image.load()
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected native WSI tile dimensions {image.size}")
                filename = f"{x}_{y}.jpg"
                (out_dir / filename).write_bytes(tile_data)
            records.append({"case_id": case, "slide_id": item["slide_id"], "patch_name": filename,
                            "x_px": x, "y_px": y, "width": 512, "height": 512,
                            "source_level": 0, "visits": visits,
                            "source_url": url, "source_offset": offset, "source_bytes": length})
        print(f"{case}: {len(records)} unique native WSI tiles ({sum(r['source_bytes'] for r in records):,} bytes)", flush=True)
        return records

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch_case, case) for case in grouped]
        for future in futures:
            results.extend(future.result())
    (pilot / "expert_roi_tiles.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (pilot / "expert_roi_tiles.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["case_id", "slide_id", "patch_name", "x_px", "y_px", "width", "height", "source_level", "source_offset", "source_bytes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fields})


if __name__ == "__main__":
    main()
