"""Bounded EPR 21/22 reader and PEAN spatial pilot; no inferred clinical truth.

EPR timestamps are uncalibrated. Frame/FPS times are exported separately and
never used as validated dwell times. Original samples are not called fixations.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME = struct.Struct("<iiiiidd")


def read_epr(path):
    payload = Path(path).read_bytes()
    stream = io.BytesIO(payload)

    def take(count):
        data = stream.read(count)
        if len(data) != count:
            raise ValueError(f"Truncated EPR at byte {stream.tell()}")
        return data

    def marker(value):
        if take(len(value)) != value:
            raise ValueError(f"Expected {value!r} at byte {stream.tell()-len(value)}")

    def unpack(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))

    def string():
        length = 0
        for shift in range(0, 35, 7):
            value = unpack("B")[0]
            length |= (value & 127) << shift
            if not value & 128:
                break
        else:
            raise ValueError("Invalid 7-bit string length")
        if length > 1_000_000:
            raise ValueError("Unbounded EPR string")
        return take(length).decode("utf-8")

    marker(b"EPRFILE")
    version, fps, count = unpack("iii")
    if version not in (21, 22) or fps <= 0 or not 0 < count <= 10_000_000:
        raise ValueError("Unsupported EPR version/FPS/frame count")
    marker(b"DOCTOR")
    doctor = string()
    doctor_attributes=dict(zip(('is_male','age','work_years','title_code','read_count_code'),unpack('?HHHH')))
    marker(b"SLIDE")
    slide_name, quickhash = string(), string()
    width, height, ref_width, ref_height, minlevel = unpack("qqqqi")
    marker(b"SCREEN")
    screen_width, screen_height, screen_cm_width, screen_cm_height = unpack("iiff")
    take(24)
    marker(b"DATA")
    raw = take(count * FRAME.size)
    frames = [dict(zip(("screen_x", "screen_y", "epr_level", "eye_x", "eye_y", "head_z_mm", "timestamp_raw"), values))
              for values in FRAME.iter_unpack(raw)]
    times = np.asarray([x["timestamp_raw"] for x in frames])
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite EPR timestamp")
    dt = np.diff(times)
    tail_start=stream.tell()
    tail={'status':'absent','comment':None,'type_text':None,'mapdata_present':False,'mouse_event_timing_available':False}
    if stream.tell()<len(payload):
        # All currently released records have a direct TAIL. Unknown blocks are
        # retained explicitly instead of being silently interpreted as actions.
        remaining=payload[stream.tell():]
        tail['mapdata_present']=b'MAPDATA' in remaining
        if remaining.startswith(b'TAIL'):
            marker(b'TAIL');tail['comment']=string()
            if version==21:tail['legacy_reserved_hex']=take(2).hex()
            tail['type_text']=string()
            trailer=payload[stream.tell():]
            tail.update(status='parsed_text_no_operation_log',trailer_hex=trailer.hex(),end_marker=trailer.endswith(b'END'))
        else:
            tail.update(status='unsupported_tail_block',unparsed_hex=remaining.hex())
    return {
        "source_file": Path(path).name, "source_sha256": hashlib.sha256(payload).hexdigest(),
        "version": version, "fps_header": fps, "frame_count": count,
        "doctor_field_token": hashlib.sha256(doctor.encode()).hexdigest()[:16],
        "doctor_identity_verified": False, "internal_slide_name": slide_name,
        "doctor_attributes":doctor_attributes,"tail":tail,
        "quickhash": quickhash, "slide_dimensions": [width, height],
        "reference_dimensions": [ref_width, ref_height], "minlevel": minlevel,
        "screen_dimensions": [screen_width, screen_height],
        "screen_cm": [screen_cm_width, screen_cm_height],
        "time_status": "uncalibrated", "timestamp_unit": None,
        "raw_median_interval": float(np.median(dt)) if len(dt) else None,
        "raw_elapsed": float(times[-1]-times[0]),
        "nonpositive_timestamp_intervals": int((dt <= 0).sum()),
        "fps_derived_duration_ms_unvalidated": (count-1) * 1000 / fps,
        "bytes_after_frame_data": len(payload) - tail_start,
        "frames": frames,
    }


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def table(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory(raw_root):
    import openslide
    raw_root = Path(raw_root)
    slides = []
    byquick = defaultdict(list)
    for path in sorted(raw_root.rglob("*.ndpi")):
        with openslide.OpenSlide(str(path)) as slide:
            row = {"path": str(path.resolve()), "filename": path.name,
                   "quickhash": slide.properties.get("openslide.quickhash-1"),
                   "dimensions": list(slide.dimensions),
                   "level_downsamples": list(slide.level_downsamples),
                   "mpp_x": float(slide.properties.get("openslide.mpp-x", 0)) or None,
                   "mpp_y": float(slide.properties.get("openslide.mpp-y", 0)) or None,
                   "objective": slide.properties.get("openslide.objective-power")}
            slides.append(row)
            byquick[row["quickhash"]].append(row)
    content = defaultdict(list)
    failures = []
    for path in sorted(raw_root.rglob("*.epr")):
        if "(1)" in path.name:  # Download duplicate aliases audited separately.
            continue
        try:
            record = read_epr(path)
            record.pop("frames")
            record["path"] = str(path.resolve())
            content[record["source_sha256"]].append(record)
        except (ValueError, UnicodeError, struct.error) as error:
            failures.append({"path": str(path), "error": str(error)})
    records = []
    for digest, aliases in content.items():
        record = dict(aliases[0])
        matches = byquick.get(record["quickhash"], [])
        record.update(aliases=[r["source_file"] for r in aliases], content_id=digest[:16],
                      matched_slide=matches[0] if len(matches) == 1 else None,
                      matching_slides=len(matches))
        record["identity_verified"] = bool(len(matches) == 1 and record["slide_dimensions"] == matches[0]["dimensions"])
        records.append(record)
    matched = {r["matched_slide"]["filename"] for r in records if r["identity_verified"]}
    return {"wsi_count": len(slides), "epr_files": sum(map(len, content.values())),
            "unique_epr_contents": len(records), "identity_verified_records": sum(r["identity_verified"] for r in records),
            "duplicate_group_sizes": dict(Counter(len(x) for x in content.values())),
            "slides_without_epr": [s["filename"] for s in slides if s["filename"] not in matched],
            "parse_failures": failures, "records": records, "slides": slides}


def project_frame(frame, metadata, downsamples):
    """EPR format sheet: minlevel is level-0; actual NDPI scales are used."""
    row = dict(frame)
    level = frame["epr_level"] - metadata["minlevel"]
    reasons = []
    if not 0 <= level < len(downsamples):
        reasons.append("invalid_pyramid_level")
    if not (0 <= frame["eye_x"] < metadata["screen_dimensions"][0]
            and 0 <= frame["eye_y"] < metadata["screen_dimensions"][1]):
        reasons.append("outside_screen")
    if not math.isfinite(frame["head_z_mm"]) or frame["head_z_mm"] <= 0:
        reasons.append("invalid_head_distance")
    ds = float(downsamples[level]) if 0 <= level < len(downsamples) else None
    x = (frame["eye_x"]-frame["screen_x"]) * ds if ds else None
    y = (frame["eye_y"]-frame["screen_y"]) * ds if ds else None
    if x is not None and not (0 <= x < metadata["slide_dimensions"][0] and 0 <= y < metadata["slide_dimensions"][1]):
        reasons.append("outside_wsi")
    row.update(wsi_level=level, downsample=ds, gaze_x_level0=x, gaze_y_level0=y,
               gaze_x_norm=x/metadata["slide_dimensions"][0] if x is not None else None,
               gaze_y_norm=y/metadata["slide_dimensions"][1] if y is not None else None,
               display_left_level0=-frame["screen_x"] * ds if ds else None,
               display_top_level0=-frame["screen_y"] * ds if ds else None,
               display_width_level0=metadata["screen_dimensions"][0] * ds if ds else None,
               display_height_level0=metadata["screen_dimensions"][1] * ds if ds else None,
               viewport_source="screen_extent_projection_not_application_viewport_validated",
               time_ms=None, time_status="uncalibrated", exclusion_reason=";".join(reasons))
    return row


def tissue_mask(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    # H&E stain relative to green rejects the olive scanner background in NDPI.
    # Saturation/darkness alone incorrectly calls this coloured glass tissue.
    from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation
    stain = (rgb.mean(axis=2) < 245) & ((rgb[:,:,0]-rgb[:,:,1] > 8) | (rgb[:,:,2]-rgb[:,:,1] > 8))
    dark_near_stain = (rgb.mean(axis=2) < 90) & binary_dilation(stain,iterations=3)
    return binary_fill_holes(binary_closing(stain | dark_near_stain,iterations=2))


def spatial_visits(rows, mpp=None, radius_um=None):
    """Compatibility entry; all grouping now delegates to the established rules."""
    if radius_um is not None:raise ValueError('Use shared Config pixel radii; no separate PEAN thresholds')
    from .pean_aligned import align_samples
    adapted=[]
    for row in rows:
        adapted.append({'downsample':1.,'display_left_level0':0.,'display_top_level0':0.,
                        'display_width_level0':100000.,'display_height_level0':100000.,**row})
    result=align_samples(adapted,(100000,100000),50.,do_compress=False)
    regions=[{**r,'x_level0':r['x_px'],'y_level0':r['y_px']} for r in result['regions']]
    return regions,result['steps']



def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()





def render_quality_control(image,mask,rows,slide_size,path,title):
    canvas=Image.new("RGB",(2310,1150),"white");draw=ImageDraw.Draw(canvas)
    draw.text((35,20),title+" | coordinate and foreground checks",font=font(31),fill="#16354c")
    names=["A  Projected samples (before foreground filter)","B  Spatially valid samples","C  H&E foreground mask"]
    for column in range(3):
        ox=30+column*770
        draw.text((ox,90),names[column],font=font(22),fill="#16354c")
        scale=min(730/slide_size[0],930/slide_size[1])
        size=(round(slide_size[0]*scale),round(slide_size[1]*scale))
        offset=(ox+(730-size[0])//2,150+(930-size[1])//2)
        source=Image.fromarray((mask*255).astype("uint8")).convert("RGB") if column==2 else image
        canvas.paste(source.resize(size),offset)
        if column<2:
            for row in rows:
                x,y=row["gaze_x_level0"],row["gaze_y_level0"]
                if x is None or not (0<=x<slide_size[0] and 0<=y<slide_size[1]):continue
                if column==1 and not row["valid_spatial_sample"]:continue
                px,py=offset[0]+x*scale,offset[1]+y*scale
                draw.ellipse((px-2,py-2,px+2,py+2),fill="#246590" if column else "#536674")
    draw.text((35,1110),"Samples are measured frames, not precomputed fixations. Time is uncalibrated. Mask is foreground QC, not a tumour label.",font=font(21),fill="#805b17")
    canvas.save(path,dpi=(300,300))


def run_pilot(raw_root, output):
    from .pean_aligned import run_aligned
    return run_aligned(raw_root,output)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run_pilot(args.raw_root,args.output)


if __name__ == "__main__":
    main()
