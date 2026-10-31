"""Fetch only the JPEG tiles for a small, real CAMELYON16 WSI pyramid level."""
import argparse
import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from .inspect_remote_tiff import parse_value


def fetch_range(url, start, end):
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
            raise RuntimeError(f"Range mismatch at {start}-{end}")
        return response.content


def remote_directories(url, target_dimensions=None):
    session = requests.Session()
    size = int(session.head(url, timeout=30).headers["Content-Length"])
    endian_head = fetch_range(url, 0, 15)
    endian = "<" if endian_head[:2] == b"II" else ">"
    if struct.unpack(endian + "H", endian_head[2:4])[0] != 43:
        raise ValueError("Expected BigTIFF")
    first_ifd = struct.unpack(endian + "Q", endian_head[8:16])[0]

    class Reader:
        def read(self, offset, length):
            return fetch_range(url, offset, offset + length - 1)

    reader = Reader()
    offset, seen = first_ifd, set()
    while offset and offset not in seen:
        seen.add(offset)
        count = struct.unpack(endian + "Q", reader.read(offset, 8))[0]
        entries = reader.read(offset + 8, count * 20 + 8)
        tags = {}
        for index in range(count):
            item = entries[index * 20:(index + 1) * 20]
            tag, typ = struct.unpack(endian + "HH", item[:4])
            if tag not in (256, 257, 259, 262, 322, 323, 324, 325, 347):
                continue
            n = struct.unpack(endian + "Q", item[4:12])[0]
            value, _ = parse_value(reader, endian, typ, n, item[12:20])
            tags[tag] = value
        width, height = int(tags.get(256, 0)), int(tags.get(257, 0))
        matches = ((width, height) == tuple(target_dimensions) if target_dimensions is not None
                   else 1024 <= width <= 2048)
        if matches and tags.get(324) and tags.get(325):
            return tags, {"width": width, "height": height, "ifd_offset": offset,
                          "file_bytes": size, "tile_width": tags[322], "tile_height": tags[323],
                          "compression": tags[259], "photometric": tags[262]}
        offset = struct.unpack(endian + "Q", entries[count * 20:count * 20 + 8])[0]
    raise ValueError("No 1K-2K pyramid image level found")


def compose_preview(item, output_dir, base_url):
    url = f"{base_url}/{item['filename']}"
    tags, info = remote_directories(url)
    if info["compression"] != 7 or info["photometric"] != 6:
        raise ValueError(f"Unsupported WSI tile encoding for {item['filename']}: {info}")
    offsets = tags[324] if isinstance(tags[324], tuple) else (tags[324],)
    counts = tags[325] if isinstance(tags[325], tuple) else (tags[325],)
    expected_tiles = ((info["width"] + info["tile_width"] - 1) // info["tile_width"] *
                      ((info["height"] + info["tile_height"] - 1) // info["tile_height"]))
    if len(offsets) != len(counts) or len(offsets) != expected_tiles:
        raise ValueError(f"Tile table size mismatch: offsets={len(offsets)} byte_counts={len(counts)} expected={expected_tiles}")
    starts = [int(value) for value in offsets]
    lengths = [int(value) for value in counts]
    jpeg_tables = bytes(tags[347]) if isinstance(tags.get(347), tuple) else tags.get(347, b"")
    lo, hi = min(starts), max(start + length for start, length in zip(starts, lengths))
    if hi - lo > 64 * 1024 * 1024:
        raise ValueError(f"Selected pyramid tile range unexpectedly large: {hi-lo} bytes")
    compressed = fetch_range(url, lo, hi - 1)
    image = Image.new("RGB", (info["width"], info["height"]), "white")
    tiles_across = (info["width"] + info["tile_width"] - 1) // info["tile_width"]
    for index, (start, length) in enumerate(zip(starts, lengths)):
        tile_bytes = compressed[start - lo:start - lo + length]
        if jpeg_tables and tile_bytes.startswith(b"\xff\xd8") and jpeg_tables.startswith(b"\xff\xd8"):
            table_data = jpeg_tables[2:-2] if jpeg_tables.endswith(b"\xff\xd9") else jpeg_tables[2:]
            tile_bytes = tile_bytes[:2] + table_data + tile_bytes[2:]
        with Image.open(io.BytesIO(tile_bytes)) as tile:
            decoded = tile.convert("RGB")
        x = index % tiles_across * info["tile_width"]
        y = index // tiles_across * info["tile_height"]
        image.paste(decoded, (x, y))
    base_width, base_height = item["slide_width"], item["slide_height"]
    downsample = None
    content_size = None
    for power in range(1, 13):
        candidate = 2 ** power
        content_w = (base_width + candidate - 1) // candidate
        content_h = (base_height + candidate - 1) // candidate
        if (content_w <= info["width"] and content_h <= info["height"]
                and info["width"] - content_w < info["tile_width"]
                and info["height"] - content_h < info["tile_height"]):
            downsample, content_size = candidate, (content_w, content_h)
            break
    if downsample is None:
        raise ValueError(f"Cannot infer integer pyramid downsample from base size for {item['filename']}")
    image = image.crop((0, 0, *content_size))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / f"{item['experiment_id']}_{item['slide_id']}_overview.png"
    image.save(preview, optimize=True)
    info.update({"experiment_id": item["experiment_id"], "slide_id": item["slide_id"],
                 "filename": item["filename"], "source_url": url,
                 "selected_tile_count": len(offsets), "selected_compressed_bytes": len(compressed),
                 "overview_path": str(preview), "whole_wsi_downloaded": False,
                 "coordinate_space": "CAMELYON16 level-0 WSI pixels",
                 "slide_dimensions_level0": [base_width, base_height],
                 "overview_dimensions": list(image.size), "overview_downsample": downsample,
                 "diagnosis": item["diagnosis"], "tumor_size": item["tumor_size"],
                 "aspect_ratio_match": abs(image.width / image.height - base_width / base_height) < 1e-6})
    print(f"{item['experiment_id']} overview {info['width']}x{info['height']}; "
          f"{len(offsets)} JPEG tiles, {len(compressed):,} downloaded bytes", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camelyon-root", type=Path, default=Path(r"D:\pathologyData\CAMELYON16"))
    parser.add_argument("--preview-output", type=Path, default=Path(r"D:\pathologyData\CAMELYON16\pathogaze_pilot"))
    args = parser.parse_args()
    manifest = json.loads((args.camelyon_root / "pathogaze_pilot_manifest.json").read_text(encoding="utf-8"))
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(compose_preview, item, args.preview_output, manifest["source"]) for item in manifest["cases"]]
        for future in futures:
            results.append(future.result())
    (args.preview_output / "preview_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
