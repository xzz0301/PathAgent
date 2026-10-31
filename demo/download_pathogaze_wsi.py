"""Resumable, checksum-verified download of PathoGaze-matched CAMELYON16 slides."""
import argparse
import hashlib
import json
import re
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


CASES = ("C1", "C48", "C6")
BASE_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/images"


def load_manifest(data_root, camelyon_root):
    import csv
    mapping = {}
    with (data_root / "ImageMapping.csv").open(encoding="utf-8-sig") as handle:
        import csv
        for row in csv.DictReader(handle):
            mapping[row["experimentImageName"]] = row["CAMELYONImageName"]
    wanted = {mapping[case]: case for case in CASES}
    metrics = {}
    with (data_root / "ImageMetrics.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            metrics[row["experimentImageName"]] = row
    sizes, checksums = {}, {}
    with (camelyon_root / "_images_not_downloaded.txt").open(encoding="utf-8-sig") as handle:
        for line in handle:
            match = re.match(r"CAMELYON16/images/([^\t]+\.tif)\t(\d+)", line.strip())
            if match:
                sizes[match.group(1)] = int(match.group(2))
    with (camelyon_root / "checksums.md5").open(encoding="utf-8-sig") as handle:
        for line in handle:
            match = re.search(r"([0-9a-fA-F]{32})\s+\*?images/([^\s]+\.tif)$", line.strip())
            if match:
                checksums[match.group(2)] = match.group(1).lower()
    result = []
    for filename, case in wanted.items():
        if filename + ".tif" not in sizes or filename + ".tif" not in checksums:
            raise ValueError(f"Missing size/checksum for {filename}")
        metric = metrics[case]
        result.append({"experiment_id": case, "slide_id": filename,
                       "filename": filename + ".tif", "bytes": sizes[filename + ".tif"],
                       "md5": checksums[filename + ".tif"],
                       "slide_width": int(float(metric["width(px)"])),
                       "slide_height": int(float(metric["height(px)"])),
                       "diagnosis": metric["diagnosis"], "tumor_size": metric["tumorSize"]})
    return result


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(item, offset, end):
    for attempt in range(1, 7):
        try:
            with requests.get(f"{BASE_URL}/{item['filename']}",
                              headers={"Range": f"bytes={offset}-{end}"},
                              stream=True, timeout=(20, 35)) as response:
                response.raise_for_status()
                expected_range = f"bytes {offset}-{end}/{item['bytes']}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Incorrect HTTP range response: {response.status_code} {response.headers.get('Content-Range')}")
                segment = b"".join(response.iter_content(chunk_size=1024 * 1024))
                if len(segment) != end - offset + 1:
                    raise RuntimeError("Short range response")
                return segment
        except Exception as error:
            if attempt == 6:
                raise
            print(f"Retry {attempt}/6 {item['filename']} offset={offset}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("Range retry limit reached")


def download_one(item, output_dir, chunk_size=4 * 1024 * 1024):
    target = output_dir / item["filename"]
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == item["bytes"] and md5_file(target) == item["md5"]:
        print(f"Already verified: {target}", flush=True)
        return target
    if target.exists():
        target.rename(target.with_suffix(target.suffix + ".invalid"))
    # Keep a small batch in memory, and commit only contiguous, complete ranges.
    with ThreadPoolExecutor(max_workers=24) as ranges:
        while not part.exists() or part.stat().st_size < item["bytes"]:
            offset = part.stat().st_size if part.exists() else 0
            starts = list(range(offset, min(item["bytes"], offset + 24 * chunk_size), chunk_size))
            futures = [ranges.submit(fetch_range, item, start, min(start + chunk_size, item["bytes"]) - 1)
                       for start in starts]
            for future in futures:
                segment = future.result()
                with part.open("ab") as handle:
                    handle.write(segment)
            current = part.stat().st_size
            print(f"{item['experiment_id']} {current}/{item['bytes']} bytes ({current/item['bytes']:.1%})", flush=True)
    if part.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Size mismatch: got {part.stat().st_size}, expected {item['bytes']}")
    actual = md5_file(part)
    if actual != item["md5"]:
        raise RuntimeError(f"MD5 mismatch for {part}: {actual}")
    part.replace(target)
    print(f"Verified {target.name} md5={actual}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\pathologyData\PathoGaze1.0"))
    parser.add_argument("--camelyon-root", type=Path, default=Path(r"D:\pathologyData\CAMELYON16"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\pathologyData\CAMELYON16\images"))
    args = parser.parse_args()
    manifest = load_manifest(args.data_root, args.camelyon_root)
    total = sum(row["bytes"] for row in manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < total + 1024 ** 3:
        raise OSError("Not enough free space for selected slides and 1 GiB reserve")
    (args.output.parent / "pathogaze_pilot_manifest.json").write_text(
        json.dumps({"source": BASE_URL, "cases": manifest, "total_bytes": total}, indent=2), encoding="utf-8")
    print(f"Selected {len(manifest)} PathoGaze-matched WSIs; total {total:,} bytes ({total / 1e9:.3f} GB)", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_one, item, args.output) for item in manifest]
        for future in futures:
            future.result()
    print("All selected slides are present and match the published MD5 checksums.", flush=True)


if __name__ == "__main__":
    main()
