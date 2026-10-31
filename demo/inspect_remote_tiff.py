"""Inspect BigTIFF directory metadata using only byte-range HTTP requests."""
import argparse
import struct
import requests


TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
NAMES = {256: "width", 257: "height", 258: "bits", 259: "compression", 262: "photometric",
         273: "strip_offsets", 277: "samples", 278: "rows_per_strip", 279: "strip_byte_counts",
         284: "planar", 317: "predictor", 322: "tile_width", 323: "tile_height",
         324: "tile_offsets", 325: "tile_byte_counts", 330: "sub_ifds", 338: "extra_samples",
         339: "sample_format"}


class RangeFile:
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.size = int(self.session.head(url, timeout=30).headers["Content-Length"])
        self.cache = {}

    def read(self, offset, length):
        if length <= 0:
            return b""
        block_size = 1024 * 1024
        first = offset // block_size
        last = (offset + length - 1) // block_size
        chunks = []
        for index in range(first, last + 1):
            if index not in self.cache:
                start = index * block_size
                end = min(start + block_size, self.size) - 1
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60))
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError("S3 ignored byte range")
                self.cache[index] = response.content
            chunks.append(self.cache[index])
        data = b"".join(chunks)
        start_in_data = offset - first * block_size
        return data[start_in_data:start_in_data + length]


def parse_value(reader, endian, typ, count, field):
    size = TYPE_SIZE.get(typ)
    if size is None:
        return None, 0
    total = size * count
    raw = field[:8] if total <= 8 else reader.read(struct.unpack(endian + "Q", field)[0], total)
    formats = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}
    fmt = formats[typ]
    n = count * (2 if typ in (5, 10) else 1)
    values = struct.unpack(endian + fmt * n, raw[:size * count])
    if typ == 2:
        values = tuple(v.decode("ascii", errors="replace") for v in values)
    if typ in (5, 10):
        values = tuple(values[i] / values[i + 1] for i in range(0, len(values), 2))
    return (values[0] if len(values) == 1 else values), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    reader = RangeFile(args.url)
    head = reader.read(0, 16)
    endian = "<" if head[:2] == b"II" else ">"
    if struct.unpack(endian + "H", head[2:4])[0] != 43:
        raise ValueError("Expected BigTIFF")
    offset_size, reserved = struct.unpack(endian + "HH", head[4:8])
    if offset_size != 8 or reserved != 0:
        raise ValueError("Unsupported BigTIFF offset format")
    offset = struct.unpack(endian + "Q", head[8:16])[0]
    seen = set()
    while offset and offset not in seen:
        seen.add(offset)
        count = struct.unpack(endian + "Q", reader.read(offset, 8))[0]
        entries = reader.read(offset + 8, count * 20 + 8)
        tags = {}
        for index in range(count):
            item = entries[index * 20:(index + 1) * 20]
            tag, typ = struct.unpack(endian + "HH", item[:4])
            n = struct.unpack(endian + "Q", item[4:12])[0]
            value, _ = parse_value(reader, endian, typ, n, item[12:20])
            tags[tag] = value
        summary = {NAMES.get(tag, str(tag)): value for tag, value in tags.items()
                   if tag not in (324, 325, 270, 347)}
        for array_tag in (324, 325, 330):
            value = tags.get(array_tag)
            if isinstance(value, tuple):
                summary[NAMES[array_tag] + "_count"] = len(value)
                summary[NAMES[array_tag] + "_first"] = value[:4]
            elif value:
                summary[NAMES[array_tag]] = value
        print(summary, "next_ifd=", struct.unpack(endian + "Q", entries[count * 20:count * 20 + 8])[0])
        if tags.get(256, 0) < 1024 or tags.get(257, 0) < 1024:
            break
        next_offset = struct.unpack(endian + "Q", entries[count * 20:count * 20 + 8])[0]
        sub = tags.get(330, ())
        if sub:
            offset = sub[0] if isinstance(sub, tuple) else sub
        else:
            offset = next_offset


if __name__ == "__main__":
    main()
