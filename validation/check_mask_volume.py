"""
check_mask_volume.py
Count the labelled voxels in a mask and convert them to a physical volume,
so you can tell which candidate file produced a published number.

Reads NRRD / NHDR, attached or detached, raw or gzip encoded.
Needs only numpy.

Usage
-----
    python3 check_mask_volume.py  mask1.nhdr  mask2.nrrd ...
    python3 check_mask_volume.py  --expect 0.0735  Salicornia_1_cc_labelmap.nhdr
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np

NRRD_DTYPES = {
    "uchar": np.uint8, "unsigned char": np.uint8, "uint8": np.uint8,
    "signed char": np.int8, "int8": np.int8,
    "short": np.int16, "signed short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "unsigned short": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "float": np.float32, "double": np.float64,
}


def read_header(path):
    """Return (fields, offset_of_attached_data_or_None)."""
    fields, raw = {}, b""
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            raw += line
            stripped = line.strip()
            if stripped == b"":            # blank line ends an attached header
                return fields, fh.tell()
            text = stripped.decode("ascii", "replace")
            if text.startswith("#") or text.startswith("NRRD"):
                continue
            if ":=" in text:
                k, v = text.split(":=", 1)
            elif ":" in text:
                k, v = text.split(":", 1)
            else:
                continue
            fields[k.strip().lower()] = v.strip()
    return fields, None


def spacing_from(fields):
    sd = fields.get("space directions")
    if sd:
        out = []
        for vec in re.findall(r"\(([^)]*)\)", sd):
            comps = [float(c) for c in vec.split(",")]
            out.append(max(abs(c) for c in comps))
        if out:
            return out
    if fields.get("spacings"):
        return [float(x) for x in fields["spacings"].split()]
    return None


def load_array(path):
    fields, offset = read_header(path)
    sizes = [int(x) for x in fields["sizes"].split()]
    dtype = NRRD_DTYPES[fields["type"].lower()]
    enc = fields.get("encoding", "raw").lower()
    datafile = fields.get("data file") or fields.get("datafile")

    if datafile:
        src, start = Path(path).parent / datafile, 0
        if not src.exists():
            raise FileNotFoundError(f"data file not found: {datafile}")
    else:
        src, start = Path(path), offset

    count = int(np.prod(sizes))
    if enc in ("raw",):
        arr = np.memmap(src, dtype=dtype, mode="r", offset=start, shape=(count,))
    elif enc in ("gzip", "gz"):
        with open(src, "rb") as fh:
            fh.seek(start)
            arr = np.frombuffer(gzip.decompress(fh.read()), dtype=dtype, count=count)
    else:
        raise ValueError(f"unsupported encoding: {enc}")
    return arr, sizes, spacing_from(fields), fields


def analyse(path, expect_mm3, label):
    print(f"\n=== {Path(path).name} ===")
    try:
        arr, sizes, spacing, fields = load_array(path)
    except Exception as e:
        print(f"  !! could not read: {e}")
        return

    print(f"  dimensions : {sizes}")
    print(f"  type       : {fields.get('type')}  encoding: {fields.get('encoding','raw')}")
    print(f"  spacing mm : {spacing}")

    if not spacing:
        print("  !! no spacing in header; cannot convert to a volume")
        return
    voxel_mm3 = float(np.prod(spacing))
    print(f"  voxel vol  : {voxel_mm3:.6e} mm^3")

    # count labels in chunks so a 1 GB mask stays comfortable
    counts = np.zeros(256, dtype=np.int64)
    if arr.dtype == np.uint8:
        for i in range(0, arr.size, 50_000_000):
            counts += np.bincount(arr[i:i + 50_000_000], minlength=256)
        present = {int(v): int(c) for v, c in enumerate(counts) if c}
    else:
        vals, cnt = np.unique(np.asarray(arr), return_counts=True)
        present = {int(v): int(c) for v, c in zip(vals, cnt)}

    print(f"  label values and voxel counts:")
    for v, c in sorted(present.items()):
        print(f"      {v:>5} : {c:>12,}   -> {c * voxel_mm3:.6f} mm^3")

    if label is not None:
        n = present.get(label, 0)
        tag = f"label {label}"
    else:
        n = sum(c for v, c in present.items() if v != 0)
        tag = "all non-zero"
    vol = n * voxel_mm3
    print(f"  {tag}: {n:,} voxels  =  {vol:.6f} mm^3")

    if expect_mm3:
        diff = (vol - expect_mm3) / expect_mm3 * 100
        verdict = "MATCH" if abs(diff) < 0.5 else "DOES NOT MATCH"
        print(f"  expected {expect_mm3:.6f} mm^3  ->  {diff:+.2f}%   {verdict}")
        need = expect_mm3 / voxel_mm3
        print(f"  (a matching mask would hold about {need:,.0f} voxels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect", type=float, default=None,
                    help="published volume in mm^3, e.g. 0.0735")
    ap.add_argument("--label", type=int, default=None,
                    help="count only this label value (e.g. 153); "
                         "default counts every non-zero voxel")
    args = ap.parse_args()
    for f in args.files:
        analyse(f, args.expect, args.label)


if __name__ == "__main__":
    main()
