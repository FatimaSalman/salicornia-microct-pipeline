#!/usr/bin/env python3
"""
extract_slice.py
================
Reads a raw .vol volume using its paired .nhdr header, pulls out ONE 2D
slice, saves it as a small PNG (a few MB instead of the whole volume),
and optionally overlays the known geometric boundary circles at the
correct pixel scale (so you get a true, calibrated overlay instead of
the earlier side-by-side approximation).

Usage
-----
    python3 extract_slice.py <path_to.nhdr> [--slice Z] [--out out.png]
                              [--center Y X] [--radii r1 r2 r3]

Example (Salicornia 1 / I-0, using the values already in the manuscript):
    python3 extract_slice.py "Salicornia 1 scan 2.nhdr" \
        --slice 500 \
        --center 520 591 \
        --radii 90 190 250 \
        --out I0_slice_overlay.png

Notes
-----
- The .vol file must be in the SAME folder as the .nhdr (the header
  just points to it by filename, e.g. "Salicornia 1 scan 2.vol").
- --slice is the Z-index (0 to sizes[2]-1). If you don't know the exact
  slice used for the manual annotation, start with the middle slice
  (this script defaults to sizes[2]//2) and compare visually.
- --center is in (Y, X) PIXEL coordinates within that slice, matching
  the convention used in the manuscript's technical parameter table.
- --radii are in MICROMETERS (µm); the script converts them to pixels
  using the voxel spacing read from the header, so you don't have to
  do that math by hand.
- Output is a normalized 8-bit PNG (for easy viewing/uploading), not
  the original 16-bit data. If you need the raw 16-bit slice instead,
  add --raw16 and it will save a .tif alongside the PNG.
"""
import argparse
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_nhdr(nhdr_path: Path):
    text = nhdr_path.read_text(errors="replace")
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip().lower()] = val.strip()

    dtype_map = {
        "unsigned short": np.uint16,
        "short": np.int16,
        "unsigned char": np.uint8,
        "uchar": np.uint8,
        "float": np.float32,
        "double": np.float64,
    }
    dtype = dtype_map[fields["type"].strip()]
    sizes = [int(x) for x in fields["sizes"].split()]          # [X, Y, Z]
    spacings_mm = [float(x) for x in fields["spacings"].split()]  # mm/voxel
    spacings_um = [s * 1000.0 for s in spacings_mm]
    endian = fields.get("endian", "little")
    data_file = fields["data file"].strip()
    return dict(dtype=dtype, sizes=sizes, spacings_um=spacings_um,
                endian=endian, data_file=data_file)


def read_slice(vol_path: Path, meta: dict, z: int) -> np.ndarray:
    nx, ny, nz = meta["sizes"]
    if not (0 <= z < nz):
        raise ValueError(f"--slice {z} out of range (volume has {nz} slices, 0-{nz-1})")

    dtype = np.dtype(meta["dtype"])
    if meta["endian"] == "little":
        dtype = dtype.newbyteorder("<")
    else:
        dtype = dtype.newbyteorder(">")

    slice_elements = nx * ny
    bytes_per_element = dtype.itemsize
    offset_bytes = z * slice_elements * bytes_per_element

    with open(vol_path, "rb") as f:
        f.seek(offset_bytes)
        raw = f.read(slice_elements * bytes_per_element)

    if len(raw) != slice_elements * bytes_per_element:
        raise IOError("Unexpected EOF while reading slice — check --slice index "
                       "and that the .vol file matches the .nhdr sizes.")

    arr = np.frombuffer(raw, dtype=dtype).reshape((ny, nx))  # row-major: Y rows, X cols
    return arr


def normalize_to_8bit(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = ((arr - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nhdr", type=Path, help="Path to the .nhdr header file")
    ap.add_argument("--slice", type=int, default=None,
                    help="Z-index of the slice to extract (default: middle slice)")
    ap.add_argument("--out", type=Path, default=Path("slice_out.png"))
    ap.add_argument("--center", type=float, nargs=2, metavar=("Y", "X"),
                    default=None, help="Stem center in pixel coords, for circle overlay")
    ap.add_argument("--radii", type=float, nargs="+", default=None,
                    help="One or more boundary radii in MICROMETERS to overlay as circles")
    ap.add_argument("--raw16", action="store_true",
                    help="Also save the untouched 16-bit slice as a .tif")
    args = ap.parse_args()

    meta = parse_nhdr(args.nhdr)
    vol_path = args.nhdr.parent / meta["data_file"]
    if not vol_path.exists():
        raise FileNotFoundError(
            f"Expected raw volume at '{vol_path}' (from 'data file:' in the .nhdr) "
            f"but it wasn't found. Make sure the .vol sits next to the .nhdr."
        )

    z = args.slice if args.slice is not None else meta["sizes"][2] // 2
    print(f"Volume: {meta['sizes']} voxels, spacing {meta['spacings_um']} µm/voxel")
    print(f"Reading slice Z={z} from {vol_path.name} ...")

    arr16 = read_slice(vol_path, meta, z)

    if args.raw16:
        Image.fromarray(arr16).save(args.out.with_suffix(".tif"))
        print(f"Saved raw 16-bit slice -> {args.out.with_suffix('.tif')}")

    arr8 = normalize_to_8bit(arr16)
    img = Image.fromarray(arr8).convert("RGB")

    if args.center is not None and args.radii:
        cy, cx = args.center
        spacing_x, spacing_y = meta["spacings_um"][0], meta["spacings_um"][1]
        draw = ImageDraw.Draw(img)
        colors = [(0, 255, 0), (255, 165, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255)]
        for i, r_um in enumerate(args.radii):
            r_px_x = r_um / spacing_x
            r_px_y = r_um / spacing_y
            bbox = [cx - r_px_x, cy - r_px_y, cx + r_px_x, cy + r_px_y]
            draw.ellipse(bbox, outline=colors[i % len(colors)], width=3)
        draw.line([(cx - 10, cy), (cx + 10, cy)], fill=(255, 255, 0), width=2)
        draw.line([(cx, cy - 10), (cx, cy + 10)], fill=(255, 255, 0), width=2)

    img.save(args.out)
    print(f"Saved slice (with overlay if requested) -> {args.out}")


if __name__ == "__main__":
    main()
