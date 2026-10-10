#!/usr/bin/env python3
import numpy as np
import tifffile
import os

INPUT = "Salicornia_8_stack.tif"
SAMPLE_NAME = "Salicornia8_"
OUT_FOLDER = os.path.join("image_folder", SAMPLE_NAME)
os.makedirs(OUT_FOLDER, exist_ok=True)

grid_path = os.path.join(OUT_FOLDER, SAMPLE_NAME + "GRID-8bit.tif")
phase_path = os.path.join(OUT_FOLDER, SAMPLE_NAME + "PHASE-8bit.tif")

print("Loading 16-bit stack...")
arr16 = tifffile.imread(INPUT)

# Contrast-stretch a meaningful window instead of the full 0-65535 range
LOW, HIGH = 12000, 50000
print(f"Applying contrast stretch: [{LOW}, {HIGH}] -> [0, 255]")
arr_clipped = np.clip(arr16.astype(np.float32), LOW, HIGH)
arr8 = ((arr_clipped - LOW) / (HIGH - LOW) * 255.0).astype(np.uint8)
print("New 8-bit stats: mean=", arr8.mean(), "percentiles:", np.percentile(arr8, [5,25,50,75,95]))

tifffile.imwrite(grid_path, arr8, photometric='minisblack')
print("Saved GRID:", grid_path)

if os.path.exists(phase_path) or os.path.islink(phase_path):
    os.remove(phase_path)
os.symlink(os.path.abspath(grid_path), phase_path)
print("Created symlink for PHASE ->", phase_path)