"""Verify and extract concatenated ZIP volumes without creating a second ZIP copy."""
import argparse
import bisect
import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class VolumeReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.ends = []
        total = 0
        for path in paths:
            total += path.stat().st_size
            self.ends.append(total)
        self.position = 0
        self.handle = None
        self.part = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.ends[-1]) + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.ends[-1] - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size:
            index = bisect.bisect_right(self.ends, self.position)
            start = self.ends[index - 1] if index else 0
            if self.part != index:
                if self.handle:
                    self.handle.close()
                self.handle = self.paths[index].open("rb")
                self.part = index
            self.handle.seek(self.position - start)
            data = self.handle.read(min(size, self.ends[index] - self.position))
            if not data:
                raise EOFError("Unexpected end of volume")
            chunks.append(data)
            self.position += len(data)
            size -= len(data)
        return b"".join(chunks)

    def close(self):
        if self.handle:
            self.handle.close()
        super().close()


def restore(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    manifest = json.loads((source / "VOLUMES.json").read_text(encoding="utf-8-sig"))
    paths = []
    combined = hashlib.sha256()
    for part in manifest["parts"]:
        path = (source / part["name"]).resolve()
        if not path.is_relative_to(source) or path.stat().st_size != part["bytes"]:
            raise ValueError(f"Invalid volume: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                combined.update(chunk)
        if digest.hexdigest() != part["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        paths.append(path)
        print(f"Verified {path.name}", flush=True)
    if sum(p.stat().st_size for p in paths) != manifest["source_bytes"] or combined.hexdigest() != manifest["source_sha256"]:
        raise ValueError("Combined ZIP size/SHA256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        raise FileExistsError(f"Previous incomplete extraction exists: {staging}")
    files = []
    with VolumeReader(paths) as reader, zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        if shutil.disk_usage(destination.parent).free < total + 1024**3:
            raise OSError("Insufficient free space for model extraction")
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if not parts or parts[0] != "Patho-R1-7B" or ".." in parts or "\\" in info.filename or any(":" in p for p in parts):
                raise ValueError(f"Unsafe archive member: {info.filename}")
        staging.mkdir()
        for info in infos:
            relative = Path(*PurePosixPath(info.filename).parts[1:])
            target = staging / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as incoming, target.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    outgoing.write(chunk)
                    digest.update(chunk)
            files.append({"name": relative.as_posix(), "bytes": target.stat().st_size,
                          "sha256": digest.hexdigest()})
            print(f"Extracted {relative.as_posix()} (CRC verified)", flush=True)
    package = staging / "PACKAGE_MANIFEST.json"
    if package.is_file():
        extracted = {entry["name"]: entry for entry in files}
        for entry in json.loads(package.read_text(encoding="utf-8"))["files"]:
            if extracted.get(entry["name"]) != entry:
                raise ValueError(f"Package file hash/size mismatch: {entry['name']}")
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "combined_zip_sha256": combined.hexdigest(), "volume_hashes_verified": len(paths),
              "files": files}
    (staging / "RESTORE_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.rename(destination)
    print(f"Restored model: {destination}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    restore(args.source, args.destination)
