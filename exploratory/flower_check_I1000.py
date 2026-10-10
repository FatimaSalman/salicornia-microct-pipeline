#!/usr/bin/env python3
"""Flag slices where the cc mask has an internal hole before fill_holes
would have removed it only if enclosed in 2D — i.e., slices whose filled
area differs from raw area."""
import numpy as np, nrrd
from pathlib import Path
from scipy.ndimage import binary_fill_holes

p = Path.home()/"Desktop"/"Salicornia_Project"/"Salicornia 4, Inowroclaw 1000 mM"/"cc_lm_v5.nrrd"
arr, _ = nrrd.read(str(p))
vol = np.transpose(arr, (2, 1, 0)).astype(bool)

hole_frac = []
for z in range(vol.shape[0]):
    m = vol[z]
    filled = binary_fill_holes(m)
    hole_frac.append((filled.sum() - m.sum()) / max(filled.sum(), 1))
hole_frac = np.array(hole_frac)

flagged = np.where(hole_frac > 0.05)[0]   # >5% of disc area missing as a 2D-enclosed hole
print(f"{len(flagged)} slices with a 2D-enclosed hole >5% of area")
print("z values:", flagged.tolist())
print("max hole fraction:", hole_frac.max(), "at z =", hole_frac.argmax())