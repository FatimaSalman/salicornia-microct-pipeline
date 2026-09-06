#!/usr/bin/env python3
"""
prepare_grid_phase.py
======================
Converts a 16-bit microCT TIFF stack (from vol_to_tiff.py) into an 8-bit
GRID-8bit.tif stack for the leaf-traits-microct pipeline, by linear
contrast-stretching a fixed intensity window to [0, 255], processed
slice-by-slice to avoid loading the full volume into memory.

Because leaf-traits-microct expects dual-channel (phase + absorption)
input and the present data are single-channel, a PHASE-8bit.tif symlink
pointing at the same GRID-8bit.tif file is also created.

Usage
-----
    python3 prepare_grid_phase.py --input Salicornia_1_stack.tif \
        --sample-name Salicornia1_ --low 12000 --high 50000
"""
import argparse
import os
import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="16-bit input TIFF stack (from vol_to_tiff.py)")
    ap.add_argument("--sample-name", required=True, help="e.g. Salicornia1_")
    ap.add_argument("--low", type=float, default=12000, help="Lower clip bound (maps to 0)")
    ap.add_argument("--high", type=float, default=50000, help="Upper clip bound (maps to 255)")
    ap.add_argument("--out-folder", default=None, help="Default: image_folder/<sample-name>/")
    args = ap.parse_args()

    out_folder = args.out_folder or os.path.join("image_folder", args.sample_name)
    os.makedirs(out_folder, exist_ok=True)

    grid_path = os.path.join(out_folder, args.sample_name + "GRID-8bit.tif")
    phase_path = os.path.join(out_folder, args.sample_name + "PHASE-8bit.tif")

    print(f"Opening {args.input} (memory-mapped, not fully loaded)...")
    with tifffile.TiffFile(args.input) as tif:
        n_pages = len(tif.pages)
        print(f"Total slices: {n_pages}")

        with tifffile.TiffWriter(grid_path) as writer:
            for i, page in enumerate(tif.pages):
                slice_16 = page.asarray()
                clipped = np.clip(slice_16.astype(np.float32), args.low, args.high)
                slice_8 = ((clipped - args.low) / (args.high - args.low) * 255.0).astype(np.uint8)
                writer.write(slice_8, contiguous=True)
                if i % 100 == 0:
                    print(f"  processed slice {i}/{n_pages}")

    print(f"Saved GRID: {grid_path}")

    if os.path.exists(phase_path) or os.path.islink(phase_path):
        os.remove(phase_path)
    os.symlink(os.path.abspath(grid_path), phase_path)
    print(f"Created symlink for PHASE -> {phase_path}")


if __name__ == "__main__":
    main()
