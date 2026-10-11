#!/usr/bin/env python3
"""generate_panel_d.py
Figure 2, Panel D: Three boundary-determination methods illustrated
side-by-side, using the actual data/results from each sample's
volumetric quantification pipeline:
  D1 - S1 (I-0):    radial texture profile + half-maximum boundary
  D2 - S5 (C-0):    radial texture profile + maximum-gradient boundary
  D3 - S8 (C-1000): texture-value histogram + Otsu threshold
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# Paths - EDIT to match your actual folder structure
# ==========================================
S1_PROFILE_PATH = os.path.join(DATA, "Salicornia 1, Inowroclaw 0 mM/S1_profile_z500.npy")
S5_PROFILE_PATH = os.path.join(DATA, "Salicornia_5/S5_profile_z500.npy")
S8_TEXVALS_PATH = os.path.join(DATA, "Salicornia_8/S8_slice492_texture_values.npy")

SPACING_UM = 2.5

# ==========================================
# D1: S1 (I-0) - half-maximum method
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
# D2: S5 (C-0) - maximum-gradient method
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
print(f"r_peak_2 = {r_peak_2} px = {r_peak_2*SPACING_UM} µm, value = {s5_profile[r_peak_2]}")
r_um_2 = r_bound_2 * SPACING_UM
radii_2 = np.arange(len(s5_profile)) * SPACING_UM

# ==========================================
# D3: S8 (C-1000) - Otsu threshold on texture histogram
# ==========================================
s8_texvals = np.load(S8_TEXVALS_PATH)
OTSU_THRESH = 2841.73

# ==========================================
# PLOT: 3 panels side by side
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)

# --- D1: Half-Max (S1, I-0) ---
ax = axes[0]
ax.plot(radii_1, s1_profile, color='gray', linewidth=0.8, alpha=0.5, label='Raw profile')
ax.plot(radii_1, s1_smooth, color='#1f77b4', linewidth=2, label='Smoothed (Savitzky-Golay)')
ax.axhline(half_val_1, color='red', linestyle='--', linewidth=1, label='Half-maximum level')
ax.axvline(r_um_1, color='red', linestyle=':', linewidth=1.8,
           label=f'Boundary = {r_um_1:.1f} µm')
ax.plot(radii_1[peak_idx_1], peak_val_1, 'k^', markersize=8, label='Profile peak')
ax.set_title('D1: Half-maximum profile\n(Inowroc\u0142aw, 0 mM)', fontsize=10, fontweight='bold')
ax.set_xlabel('Radial distance from stem centre (µm)')
ax.set_ylabel('Local texture (std. dev.)')
ax.set_xlim(0, 150)
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# --- D2: Max-Gradient (S5, C-0) ---
ax = axes[1]
ax2 = ax.twinx()
ax.plot(radii_2, s5_profile, color='#1f77b4', linewidth=2, label='Radial texture profile')
ax2.plot(radii_2, grad_2, color='#ff7f0e', linewidth=1.2, alpha=0.7, label='Gradient')
ax.axvline(r_um_2, color='green', linestyle=':', linewidth=1.8,
           label=f'Boundary = {r_um_2:.1f} µm')
ax.plot(radii_2[r_peak_2], s5_profile[r_peak_2], 'k^', markersize=8, label='Principal peak')
ax.set_title('D2: Maximum-gradient profile\n(Ciechocinek, 0 mM)', fontsize=10, fontweight='bold')
ax.set_xlabel('Radial distance from stem centre (µm)')
ax.set_ylabel('Local texture (std. dev.)', color='#1f77b4')
ax2.set_ylabel('Gradient', color='#ff7f0e')
ax.set_xlim(0, 250)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# --- D3: Otsu Threshold (S8, C-1000) ---
ax = axes[2]
ax.hist(s8_texvals.ravel(), bins=150, color='#2ca02c', alpha=0.7, edgecolor='none')
ax.axvline(OTSU_THRESH, color='purple', linestyle='--', linewidth=2,
           label=f'Otsu threshold = {OTSU_THRESH}')
ax.set_title('D3: Otsu threshold on texture\n(Ciechocinek, 1000 mM)', fontsize=10, fontweight='bold')
ax.set_xlabel('Local texture value (std. dev.)')
ax.set_ylabel('Voxel count')
ax.set_yscale('log')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure2_PanelD_boundary_methods.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: Figure2_PanelD_boundary_methods.png")

print(f"\nSummary:")
print(f"  D1 (S1, half-max):    {r_um_1:.2f} µm")
print(f"  D2 (S5, max-gradient): {r_um_2:.2f} µm")
print(f"  D3 (S8, Otsu):         threshold = {OTSU_THRESH:.2f}")