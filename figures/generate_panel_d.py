#!/usr/bin/env python3
"""generate_panel_d.py
Figure 2, Panel D: Three boundary-determination methods illustrated
side-by-side, using the actual data/results from each sample's
volumetric quantification pipeline:
  d1 - S1 (I-0):    radial texture profile + half-maximum boundary
  d2 - S5 (C-0):    radial texture profile + maximum-gradient boundary
  d3 - S8 (C-1000): texture-value histogram + Otsu threshold

TPJ compliance:
  - lower-case sub-panel designators (d1, d2, d3)
  - sans-serif (Helvetica / Arial) throughout
  - nothing below 8 pt AT THE FINAL PRINTED SIZE

WHY THE FONT SIZES LOOK ABSURD
------------------------------
This sub-panel is drawn 15 in wide but is placed into Figure 2, where it
ends up ~166 mm (6.55 in) wide. Everything is therefore shrunk by a factor
of 6.55/15 = 0.44 before it reaches the reader: a nominal 10 pt title
arrives as 4.4 pt. SCALE below undoes that. Set SCALE = 1.0 only if you
ever use this sub-panel as a stand-alone figure.

Run from the data root:
    cd ~/Desktop/Salicornia_Project
    python3 generate_panel_d.py
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")           # no interactive window; nothing blocks
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# Size compensation - see docstring
# ==========================================
SCALE = 2.3          # 15 in drawn  ->  6.55 in printed
S = lambda pt: pt * SCALE

# ==========================================
# Typography - TPJ: Helvetica or Arial, nothing below 8 pt when printed
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize":  S(9),     # -> 9.0 pt printed
    "axes.labelsize":  S(8),     # -> 8.0 pt printed
    "xtick.labelsize": S(8),
    "ytick.labelsize": S(8),
    "legend.fontsize": S(8),
    "axes.linewidth":  S(0.8),
    "xtick.major.width": S(0.8),
    "ytick.major.width": S(0.8),
    "xtick.major.size":  S(3.5),
    "ytick.major.size":  S(3.5),
    "grid.linewidth":    S(0.6),
})

# ==========================================
# Paths - EDIT to match your actual folder structure
# ==========================================
S1_PROFILE_PATH = os.path.join(DATA, "Salicornia 1, Inowroclaw 0 mM/S1_profile_z500.npy")
S5_PROFILE_PATH = os.path.join(DATA, "Salicornia_5/S5_profile_z500.npy")
S8_TEXVALS_PATH = os.path.join(DATA, "Salicornia_8/S8_slice492_texture_values.npy")

for p in (S1_PROFILE_PATH, S5_PROFILE_PATH, S8_TEXVALS_PATH):
    if not os.path.exists(p):
        raise SystemExit(
            f"NOT FOUND: {p}\n"
            f"You are in: {os.getcwd()}\n"
            f"SALICORNIA_DATA = {DATA}\n"
            f"cd to the data root, or export SALICORNIA_DATA."
        )

SPACING_UM = 2.5

# ==========================================
# d1: S1 (I-0) - half-maximum method
# ==========================================
def find_halfmax_radius_robust(profile, min_r=15, peak_search_max_r=45,
                               bg_range=(60, 100)):
    smooth = savgol_filter(profile, window_length=9, polyorder=2)
    search_region = smooth[min_r:peak_search_max_r]
    peak_idx = np.argmax(search_region) + min_r
    peak_val = smooth[peak_idx]
    bg_val = np.mean(smooth[bg_range[0]:bg_range[1]])
    half_val = (peak_val + bg_val) / 2
    for r in range(peak_idx, len(smooth)):
        if smooth[r] <= half_val:
            return r, peak_idx, peak_val, bg_val, half_val, smooth
    return None, peak_idx, peak_val, bg_val, half_val, smooth

s1_profile = np.load(S1_PROFILE_PATH)
r_bound_1, peak_idx_1, peak_val_1, bg_val_1, half_val_1, s1_smooth = \
    find_halfmax_radius_robust(s1_profile)
r_um_1 = r_bound_1 * SPACING_UM
radii_1 = np.arange(len(s1_profile)) * SPACING_UM

# ==========================================
# d2: S5 (C-0) - maximum-gradient method
# ==========================================
def method_grad(p, peak_range=(50, 95), bg_local=(115, 135)):
    idx = np.arange(*peak_range)
    valid = ~np.isnan(p[idx])
    r_peak = idx[valid][np.nanargmax(p[idx][valid])]
    seg = np.arange(r_peak, bg_local[0])
    grad = np.gradient(p)
    seg_valid = seg[~np.isnan(grad[seg])]
    r_grad = seg_valid[np.argmin(grad[seg_valid])]
    return r_grad, r_peak, grad

s5_profile = np.load(S5_PROFILE_PATH)  # already smoothed (saved as profile_smooth)
r_bound_2, r_peak_2, grad_2 = method_grad(s5_profile)
print(f"r_peak_2 = {r_peak_2} px = {r_peak_2*SPACING_UM} um, value = {s5_profile[r_peak_2]}")
r_um_2 = r_bound_2 * SPACING_UM
radii_2 = np.arange(len(s5_profile)) * SPACING_UM

# ==========================================
# d3: S8 (C-1000) - Otsu threshold on texture histogram
# ==========================================
s8_texvals = np.load(S8_TEXVALS_PATH)
OTSU_THRESH = 2841.73

# ==========================================
# PLOT: 3 sub-panels side by side
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)

# NO LEGEND BOXES. At a true 8 pt a five-entry legend either covers the
# curves or squashes them. The line styles are defined in the figure
# caption instead, which is the usual TPJ practice.
ANN = dict(fontsize=S(8), ha='left', va='top', zorder=20)

# --- d1: half-maximum (S1, I-0) ---
ax = axes[0]
ax.plot(radii_1, s1_profile, color='gray', linewidth=S(0.8), alpha=0.5,
        )
ax.plot(radii_1, s1_smooth, color='#1f77b4', linewidth=S(2),
        )
ax.axhline(half_val_1, color='red', linestyle='--', linewidth=S(1),
           )
ax.axvline(r_um_1, color='red', linestyle=':', linewidth=S(1.8),
           )
ax.plot(radii_1[peak_idx_1], peak_val_1, 'k^', markersize=S(8),
        )
ax.set_title('d1: Half-maximum profile\n(Inowroc\u0142aw, 0 mM)')
ax.set_xlabel('Radial distance from stem centre (\u00b5m)')
ax.set_ylabel('Local texture (std. dev.)')
ax.set_xlim(0, 150)
ax.text(0.66, 0.95, f'boundary\n{r_um_1:.1f} \u00b5m',
        transform=ax.transAxes, color='red', **ANN)
ax.margins(y=0.10)
ax.grid(alpha=0.3)

# --- d2: maximum-gradient (S5, C-0) ---
ax = axes[1]
ax2 = ax.twinx()
ax.plot(radii_2, s5_profile, color='#1f77b4', linewidth=S(2),
        )
ax2.plot(radii_2, grad_2, color='#ff7f0e', linewidth=S(1.2), alpha=0.7,
         )
ax.axvline(r_um_2, color='green', linestyle=':', linewidth=S(1.8),
           )
ax.plot(radii_2[r_peak_2], s5_profile[r_peak_2], 'k^', markersize=S(8),
        )
ax.set_title('d2: Maximum-gradient profile\n(Ciechocinek, 0 mM)')
ax.set_xlabel('Radial distance from stem centre (\u00b5m)')
ax.set_ylabel('Local texture (std. dev.)', color='#1f77b4')
ax2.set_ylabel('Gradient', color='#ff7f0e')
ax.set_xlim(0, 250)
ax2.tick_params(labelsize=S(8), width=S(0.8), size=S(3.5))
# drawn on ax2: the twin axes sits on top and would otherwise hide it
ax2.text(0.05, 0.95, f'boundary\n{r_um_2:.1f} \u00b5m',
         transform=ax.transAxes, color='green', **ANN)
ax.margins(y=0.10)
ax.grid(alpha=0.3)

# --- d3: Otsu threshold (S8, C-1000) ---
ax = axes[2]
ax.hist(s8_texvals.ravel(), bins=150, color='#2ca02c', alpha=0.7, edgecolor='none')
ax.axvline(OTSU_THRESH, color='purple', linestyle='--', linewidth=S(2),
           )
ax.set_title('d3: Otsu threshold on texture\n(Ciechocinek, 1000 mM)')
ax.set_xlabel('Local texture value (std. dev.)')
ax.set_ylabel('Voxel count')
ax.set_yscale('log')
# axes fraction, not data: y = 0 is invalid on a log axis and the
# annotation would be dropped without any warning
ax.text(0.40, 0.95, f'Otsu threshold\n{OTSU_THRESH:.0f}',
        transform=ax.transAxes, color='purple', **ANN)
ax.grid(alpha=0.3)

plt.tight_layout()
# 'tight' is CORRECT here: this is a sub-panel that gets re-scaled inside
# Figure 2, so trimming the white margin enlarges the content in the panel.
plt.savefig('Figure2_PanelD_boundary_methods.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: Figure2_PanelD_boundary_methods.png")

print("\nSummary:")
print(f"  d1 (S1, half-max):     {r_um_1:.2f} um")
print(f"  d2 (S5, max-gradient): {r_um_2:.2f} um")
print(f"  d3 (S8, Otsu):         threshold = {OTSU_THRESH:.2f}")
print(f"\nSCALE = {SCALE}  ->  nominal 8 pt arrives as ~8 pt at 166 mm panel width.")
