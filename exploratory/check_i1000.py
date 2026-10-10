#!/usr/bin/env python3
"""Per-slice diagnostics of the I-1000 reference mask (cc_lm_v5.nrrd)."""
import numpy as np, nrrd
from pathlib import Path
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path.home()/"Desktop"/"Salicornia_Project"/"Salicornia 4, Inowroclaw 1000 mM"/"cc_lm_v5.nrrd"
arr, _ = nrrd.read(str(p))
vol = np.transpose(arr, (2, 1, 0)).astype(bool)
VOX = 2.5**3 / 1e9                      # mm^3 per voxel (mask is full resolution)
n = vol.shape[0]
area = np.zeros(n); sol = np.zeros(n); touch = np.zeros(n, bool)
for z in range(n):
    m = vol[z]; area[z] = m.sum()
    if area[z] == 0: continue
    rp = max(regionprops(label(m)), key=lambda r: r.area)
    sol[z] = rp.solidity
    touch[z] = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()

ok = (sol >= 0.8) & ~touch
print(f"slices: {n}, total volume = {area.sum()*VOX:.4f} mm3 (Table 2: 0.7180)")
print(f"slices with solidity<0.8: {(sol<0.8).sum()} | touching image border: {touch.sum()}")

# runs of not-ok slices (0-based z)
runs, start = [], None
for z in range(n):
    if not ok[z] and start is None: start = z
    if ok[z] and start is not None: runs.append((start, z-1)); start = None
if start is not None: runs.append((start, n-1))
print("not-ok runs (0-based z):", runs[:30], "..." if len(runs) > 30 else "")

# longest contiguous ok segment
best, cur = (0, -1), None
for z in range(n):
    if ok[z] and cur is None: cur = z
    if (not ok[z] or z == n-1) and cur is not None:
        end = z if ok[z] else z-1
        if end-cur > best[1]-best[0]: best = (cur, end)
        cur = None
a, b = best
print(f"longest ok segment: z {a}-{b} ({b-a+1} slices), volume {area[a:b+1].sum()*VOX:.4f} mm3")
print(f"volume per slice, ok slices: {area[ok].mean()*VOX*1e3:.3f} e-3 mm3 vs all: {area.mean()*VOX*1e3:.3f} e-3")
print(f"fold change vs I-0 (0.0735): all slices {area.sum()*VOX/0.0735:.2f} | "
      f"mean-area-of-ok x all slices {area[ok].mean()*n*VOX/0.0735:.2f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(area*2.5**2/1e3); ax[0].set_ylabel("area (x10^3 um^2)")
ax[1].plot(sol); ax[1].axhline(0.8, color="r"); ax[1].set_ylabel("solidity"); ax[1].set_xlabel("z (0-based)")
plt.tight_layout(); plt.savefig("i1000_profile.png", dpi=120)
print("saved i1000_profile.png")