#!/usr/bin/env python3
"""Per-slice circularity of the reference cc masks (I-0, I-1000, C-0, C-1000)."""
import numpy as np, nrrd
from pathlib import Path
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "Desktop" / "Salicornia_Project"
MASKS = {
    "I-0 (S1)":    BASE / "Salicornia 1, Inowroclaw 0 mM"    / "S1_cc_labelmap_proper.nrrd",
    "I-1000 (S4)": BASE / "Salicornia 4, Inowroclaw 1000 mM" / "cc_lm_v5.nrrd",
    "C-0 (S5)":    BASE / "Salicornia_5"                      / "C-0_cc_reference_mask.nrrd",
    "C-1000 (S8)": BASE / "Salicornia_8"                      / "cc_labelmap.nrrd",
}

def disc_dice(m):
    ys, xs = np.nonzero(m)
    A = len(ys)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(A / np.pi)
    y0, y1 = int(cy - r - 3), int(cy + r + 4)
    x0, x1 = int(cx - r - 3), int(cx + r + 4)
    y0, x0 = max(y0, 0), max(x0, 0)
    sub = m[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    return 2 * (sub & disc).sum() / (sub.sum() + disc.sum()), cy, cx, r

summary = {}
for name, path in MASKS.items():
    arr, _ = nrrd.read(str(path))
    vol = np.transpose(arr, (2, 1, 0)).astype(bool)          # Z, Y, X
    zs = np.where(vol.reshape(vol.shape[0], -1).any(1))[0]
    dice, ecc, sol, ncomp, req = [], [], [], [], []
    for z in zs:
        m = vol[z]
        d, cy, cx, r = disc_dice(m)
        lab = label(m)
        big = max(regionprops(lab), key=lambda p: p.area)
        dice.append(d); ecc.append(big.eccentricity); sol.append(big.solidity)
        ncomp.append(lab.max()); req.append(r)
    dice, ecc, sol, ncomp, req = map(np.array, (dice, ecc, sol, ncomp, req))
    print(f"\n=== {name}: {len(zs)} slices (z {zs[0]}-{zs[-1]}), shape {vol.shape} ===")
    print(f"Dice to equal-area disc : min {dice.min():.3f} | 5th pct {np.percentile(dice,5):.3f} | median {np.median(dice):.3f}")
    print(f"Eccentricity            : median {np.median(ecc):.3f} | max {ecc.max():.3f}")
    print(f"Solidity                : median {np.median(sol):.3f} | min {sol.min():.3f}")
    print(f"Max components in slice : {ncomp.max()}")
    print(f"Equivalent radius (px)  : mean {req.mean():.2f} | sd {req.std():.2f}"
          f"  (x2.5 um if mask is at full resolution: mean {req.mean()*2.5:.1f} um)")
    summary[name] = (vol, zs)

# visual check: 6 slices per sample, mask outline (red) vs equal-area disc (cyan)
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
for row, (name, (vol, zs)) in enumerate(summary.items()):
    for col, z in enumerate(np.linspace(zs[0], zs[-1], 6).astype(int)):
        m = vol[z]
        d, cy, cx, r = disc_dice(m)
        h = int(r * 1.8)
        y0, x0 = max(int(cy) - h, 0), max(int(cx) - h, 0)
        ax = axes[row, col]
        ax.imshow(m[y0:int(cy) + h, x0:int(cx) + h], cmap="gray")
        ax.contour(m[y0:int(cy) + h, x0:int(cx) + h], [0.5], colors="red", linewidths=1)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx - x0 + r * np.cos(t), cy - y0 + r * np.sin(t), "c", lw=1)
        ax.set_title(f"{name} z={z} Dice={d:.3f}", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig("circularity_check.png", dpi=120)
print("\nSaved circularity_check.png")