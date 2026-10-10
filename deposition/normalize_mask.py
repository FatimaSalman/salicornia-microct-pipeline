"""
normalize_mask.py
Rewrite a reference mask into the form used for deposition:
correct voxel spacing, 8-bit labels, gzip-compressed single-file NRRD.

The voxel COUNT is never altered, so the physical volume the mask encodes
is preserved exactly; only the header and the storage format change.

Usage
-----
    # inspect only
    python3 normalize_mask.py in.nrrd --out out.nrrd --spacing-mm 0.00250000064 --dry-run

    # write it
    python3 normalize_mask.py in.nrrd --out out.nrrd --spacing-mm 0.00250000064
"""
import argparse
import gzip
from pathlib import Path

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cmv", str(Path(__file__).with_name("check_mask_volume.py")))
cmv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmv)


def write_nrrd_gzip(path, arr, sizes, spacing_mm, space="left-posterior-superior"):
    dirs = " ".join(
        "(" + ",".join(f"{spacing_mm if i == j else 0:.11g}" for j in range(3)) + ")"
        for i in range(3)
    )
    header = (
        "NRRD0004\n"
        "# Written by normalize_mask.py\n"
        "type: unsigned char\n"
        "dimension: 3\n"
        f"space: {space}\n"
        f"sizes: {sizes[0]} {sizes[1]} {sizes[2]}\n"
        f"space directions: {dirs}\n"
        "kinds: domain domain domain\n"
        "endian: little\n"
        "encoding: gzip\n"
        "space origin: (0,0,0)\n"
        "\n"
    ).encode("ascii")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(gzip.compress(np.ascontiguousarray(arr, dtype=np.uint8).tobytes(),
                               compresslevel=6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing-mm", type=float, required=True,
                    help="correct isotropic voxel spacing in MILLIMETRES, "
                         "e.g. 0.00250000064")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.spacing_mm > 0.1:
        raise SystemExit(
            f"--spacing-mm {args.spacing_mm} looks like micrometres. "
            f"A 2.5 um voxel is 0.0025 mm."
        )

    arr, sizes, old_spacing, fields = cmv.load_array(args.infile)
    arr = np.asarray(arr)
    vals, counts = np.unique(arr, return_counts=True)

    print(f"  input      : {args.infile}")
    print(f"  dimensions : {sizes}")
    print(f"  type       : {fields.get('type')}  encoding: {fields.get('encoding','raw')}")
    print(f"  spacing in : {old_spacing}")
    print(f"  spacing out: {args.spacing_mm}")
    print(f"  labels     : {dict(zip(vals.tolist(), counts.tolist()))}")

    if vals.max() > 255 or vals.min() < 0:
        raise SystemExit("label values do not fit in 8 bits; aborting")

    nonzero = int(counts[vals != 0].sum())
    vol = nonzero * args.spacing_mm ** 3
    print(f"  non-zero   : {nonzero:,} voxels")
    print(f"  volume     : {vol:.6f} mm^3   <- unchanged by this operation")

    old_bytes = Path(args.infile).stat().st_size
    if args.dry_run:
        print("  dry run: nothing written")
        return

    write_nrrd_gzip(args.out, arr, sizes, args.spacing_mm)
    new_bytes = Path(args.out).stat().st_size
    print(f"  wrote      : {args.out}")
    print(f"  size       : {old_bytes/1e6:.1f} MB  ->  {new_bytes/1e6:.1f} MB")

    # read the new file back and confirm nothing changed
    arr2, sizes2, sp2, f2 = cmv.load_array(args.out)
    n2 = int(np.count_nonzero(np.asarray(arr2)))
    ok = (n2 == nonzero) and (sizes2 == sizes)
    print(f"  verify     : {n2:,} non-zero voxels, spacing {sp2}  "
          f"-> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("round-trip verification failed; do not use this file")


if __name__ == "__main__":
    main()
