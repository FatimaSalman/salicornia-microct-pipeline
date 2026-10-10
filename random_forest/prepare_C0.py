#!/usr/bin/env python3
import numpy as np
import tifffile
import os

INPUT = "Salicornia_5_stack.tif"
SAMPLE_NAME = "Salicornia5_"
OUT_FOLDER = os.path.join("image_folder", SAMPLE_NAME)
os.makedirs(OUT_FOLDER, exist_ok=True)

grid_path = os.path.join(OUT_FOLDER, SAMPLE_NAME + "GRID-8bit.tif")
phase_path = os.path.join(OUT_FOLDER, SAMPLE_NAME + "PHASE-8bit.tif")

LOW, HIGH = 12000, 50000

print("Opening stack (memory-mapped, not fully loaded)...")
with tifffile.TiffFile(INPUT) as tif:
    n_pages = len(tif.pages)
    print(f"Total slices: {n_pages}")

    first = tif.pages[0].asarray()
    h, w = first.shape
    print(f"Slice shape: {h}x{w}")

    with tifffile.TiffWriter(grid_path) as writer:
        for i, page in enumerate(tif.pages):
            slice_16 = page.asarray()
            clipped = np.clip(slice_16.astype(np.float32), LOW, HIGH)
            slice_8 = ((clipped - LOW) / (HIGH - LOW) * 255.0).astype(np.uint8)
            writer.write(slice_8, contiguous=True)
            if i % 100 == 0:
                print(f"  processed slice {i}/{n_pages}")

print("Saved GRID:", grid_path)

if os.path.exists(phase_path) or os.path.islink(phase_path):
    os.remove(phase_path)
os.symlink(os.path.abspath(grid_path), phase_path)
print("Created symlink for PHASE ->", phase_path)