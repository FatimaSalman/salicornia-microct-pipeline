#!/usr/bin/env python3
"""Visual check of flagged dip regions in I-1000 reference mask."""
import numpy as np, nrrd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path.home()/"Desktop"/"Salicornia_Project"/"Salicornia 4, Inowroclaw 1000 mM"/"cc_lm_v5.nrrd"
arr, _ = nrrd.read(str(p))
vol = np.transpose(arr, (2, 1, 0)).astype(bool)

# dip centers read from the area plot, each shown with 2 neighbours before/after
dip_centers = [75, 195, 340, 435, 785]   # includes one near the tapering end
zs = []
for c in dip_centers:
    zs += [c-10, c, c+10]
zs = sorted(set(z for z in zs if 0 <= z < vol.shape[0]))

n = len(zs)
cols = 5
rows = int(np.ceil(n/cols))
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axes = np.array(axes).reshape(-1)
for ax, z in zip(axes, zs):
    m = vol[z]
    ys, xs = np.nonzero(m)
    if len(ys):
        y0,y1,x0,x1 = ys.min()-10, ys.max()+10, xs.min()-10, xs.max()+10
        y0,x0 = max(y0,0), max(x0,0)
        ax.imshow(m[y0:y1, x0:x1], cmap="gray")
    ax.set_title(f"z={z} area={m.sum()}", fontsize=9)
    ax.axis("off")
for ax in axes[n:]: ax.axis("off")
plt.tight_layout(); plt.savefig("i1000_dip_check.png", dpi=120)
print("saved i1000_dip_check.png")