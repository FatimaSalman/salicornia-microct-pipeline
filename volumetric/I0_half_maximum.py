#!/usr/bin/env python3
"""S1_full_volume_halfmax_method.py
==============================================================
Final protocol for I-0 (Salicornia_1, Inowroclaw 0 mM):
Z-denoising (rolling buffer) + local-texture-weighted center estimation
+ radial texture profile + half-maximum boundary criterion.

This is the SINGLE dominant-peak method, selected for I-0 because this
sample's small central cylinder produced one well-defined radial profile
peak (unlike C-0's multi-peaked, discrete vessel architecture, for which
the maximum-gradient criterion was used instead; see
S5_full_volume_grad_method.py).

NOTE ON TERMINOLOGY: the manuscript text describes this as a radial
"intensity" profile; the boundary is in fact computed from a radial
TEXTURE profile (windowed standard deviation, 5x5-pixel window) of the
Z-averaged slice, not from raw intensity directly. See manuscript
Materials and Methods for the corrected description.

Outputs:
  - s1_radii_results.pkl      : per-slice radius/area results + total volume
  - S1_profile_z<N>.npy       : representative full radial profile array,
                                 saved for representative slices (for
                                 Figure 2, Panel D1 plotting)
"""

import os
import time
import pickle
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from collections import deque

# ==========================================
# Fixed parameters (as used for I-0 in the manuscript)
# ==========================================
NHDR_PATH = "Salicornia 1 scan 2.nhdr"
SPACING = 2.50001973        # µm/voxel, isotropic
WINDOW = 5                  # local std-dev window size for texture map
HALF_WINDOW_Z = 15          # +/- 15 slices => 31-slice Z-averaging window
FULL_WINDOW_Z = 2 * HALF_WINDOW_Z + 1
NZ = 1000                   # total slices in volume
REF_Y, REF_X = 505, 567     # approximate stem centre (manual seed)
MAX_R = 150                 # max radius (px) for radial profile
FIJI_CC_RADIUS_UM = 88.25   # manual Fiji reference radius, for cross-check

# Representative slices to save the FULL profile array for
# (used for manuscript Figure 2, Panel D1)
SAVE_PROFILE_SLICES = [200, 500, 800]


# ==========================================
# I/O helpers (raw .vol/.nhdr reading)
# ==========================================
def parse_nhdr(nhdr_path):
    info = {}
    with open(nhdr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line or line.startswith('NRRD'):
                continue
            key, val = line.split(':', 1)
            info[key.strip()] = val.strip()
    return info


def load_single_slice(nhdr_path, slice_index, sizes):
    """Read one Z-slice directly from the raw binary .vol file referenced
    by the .nhdr header, without loading the full volume into memory."""
    info = parse_nhdr(nhdr_path)
    nx, ny, nz = sizes
    slice_bytes = nx * ny * 2  # uint16
    vol_path = os.path.join(os.path.dirname(nhdr_path) or ".", info['data file'])
    with open(vol_path, 'rb') as f:
        f.seek(slice_index * slice_bytes)
        raw = f.read(slice_bytes)
    return np.frombuffer(raw, dtype=np.uint16).reshape(ny, nx)


# ==========================================
# Core processing functions
# ==========================================
def local_std(img, w):
    """Windowed standard deviation (local texture map)."""
    img = img.astype(np.float64)
    mean = ndimage.uniform_filter(img, size=w)
    mean_sq = ndimage.uniform_filter(img ** 2, size=w)
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)


def estimate_center(raw, ref_y, ref_x, radius=90):
    """Texture-weighted local centre estimate within a search window
    around the approximate stem centre. Avoids using the whole image,
    which would risk pulling the centroid toward unrelated background
    texture."""
    y0, y1 = max(0, ref_y - radius), min(raw.shape[0], ref_y + radius)
    x0, x1 = max(0, ref_x - radius), min(raw.shape[1], ref_x + radius)
    roi = raw[y0:y1, x0:x1].astype(float)
    tex = local_std(roi, WINDOW)
    w_ = np.maximum(tex - np.percentile(tex, 80), 0)
    if w_.sum() == 0:
        return ref_y, ref_x
    yy, xx = np.indices(roi.shape)
    cy = y0 + (yy * w_).sum() / w_.sum()
    cx = x0 + (xx * w_).sum() / w_.sum()
    return cy, cx


def radial_profile(image, cy, cx, max_r=150):
    """Mean texture value within concentric 1-pixel-wide annuli
    (effectively averaging over all angular directions, not a fixed
    360-angle sample)."""
    y, x = np.indices(image.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    valid = r <= max_r
    rb = r[valid]
    vals = image[valid]
    sums = np.bincount(rb, weights=vals, minlength=max_r + 1)
    counts = np.bincount(rb, minlength=max_r + 1)
    return sums / np.maximum(counts, 1)


def find_halfmax_radius_robust(profile, min_r=15, peak_search_max_r=45,
                                bg_range=(60, 100)):
    """Identify the central-cylinder boundary as the first radius after
    the main profile peak at which the (smoothed) texture value falls
    to the half-maximum position between the peak and local background."""
    smooth = savgol_filter(profile, window_length=9, polyorder=2)
    search_region = smooth[min_r:peak_search_max_r]
    peak_idx = np.argmax(search_region) + min_r
    peak_val = smooth[peak_idx]
    bg_val = np.mean(smooth[bg_range[0]:bg_range[1]])
    half_val = (peak_val + bg_val) / 2
    for r in range(peak_idx, len(smooth)):
        if smooth[r] <= half_val:
            return r, peak_idx, peak_val, bg_val
    return None, peak_idx, peak_val, bg_val


# ==========================================
# Main processing loop (rolling Z-buffer)
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    info = parse_nhdr(NHDR_PATH)
    sizes = [int(x) for x in info['sizes'].split()]
    nx, ny, nz = sizes
    print(f"Volume: {nx}x{ny}x{nz}, spacing={SPACING}µm")

    t0 = time.time()
    buffer = deque()
    running_sum = None

    all_radii = []
    all_areas = []
    failed_slices = []
    saved_profiles = {}

    print("Loading initial Z window...")
    for z in range(FULL_WINDOW_Z):
        s = load_single_slice(NHDR_PATH, z, sizes).astype(np.float64)
        buffer.append(s)
        running_sum = s.copy() if running_sum is None else running_sum + s

    zc = HALF_WINDOW_Z
    print(f"Processing slices (rolling buffer, window={FULL_WINDOW_Z})...")

    while True:
        raw_denoised = running_sum / FULL_WINDOW_Z

        cy, cx = estimate_center(raw_denoised, REF_Y, REF_X)
        tex = local_std(raw_denoised, WINDOW)
        prof = radial_profile(tex, cy, cx, max_r=MAX_R)

        r_bound, peak_idx, peak_val, bg_val = find_halfmax_radius_robust(prof)

        # --- Save full profile array for representative slices (for Figure 2, Panel D1) ---
        if zc in SAVE_PROFILE_SLICES:
            out_path = os.path.join(f"S1_profile_z{zc}.npy")
            np.save(out_path, prof)
            r_um_dbg = r_bound * SPACING if r_bound is not None else float("nan")
            print(f"[SAVED] z={zc} profile -> {out_path} | "
                  f"r_bound={r_bound}px ({r_um_dbg:.2f}µm) | "
                  f"peak_idx={peak_idx} peak_val={peak_val:.1f} bg_val={bg_val:.1f}")
            saved_profiles[zc] = prof

        if r_bound is not None:
            r_um = r_bound * SPACING
            area_um2 = np.pi * (r_um ** 2)
            all_radii.append((zc, r_um))
            all_areas.append((zc, area_um2))
        else:
            failed_slices.append(zc)

        if zc % 100 == 0:
            print(f"  z={zc}/{NZ} done ({time.time() - t0:.1f}s elapsed)")

        next_z_in = zc + HALF_WINDOW_Z + 1
        if next_z_in >= NZ:
            break

        oldest = buffer.popleft()
        running_sum -= oldest
        new_slice = load_single_slice(NHDR_PATH, next_z_in, sizes).astype(np.float64)
        buffer.append(new_slice)
        running_sum += new_slice
        zc += 1

    print(f"\n✅ Total time: {time.time() - t0:.1f}s")
    print(f"✅ Succeeded: {len(all_radii)} slices | ⚠️ Failed: {len(failed_slices)} slices")

    r_vals = np.array([r for _, r in all_radii])
    print(f"\n📊 Radius statistics across full volume:")
    print(f"   mean={r_vals.mean():.2f}µm, median={np.median(r_vals):.2f}µm, "
          f"std={r_vals.std():.2f}µm")
    print(f"   min={r_vals.min():.1f}, max={r_vals.max():.1f}")

    areas = np.array([a for _, a in all_areas])
    volume_um3 = areas.sum() * SPACING
    volume_mm3 = volume_um3 / 1e9
    n_slices_used = len(all_areas)
    print(f"\n🎯 CC VOLUME (S1) = {volume_mm3:.4f} mm³ = {volume_um3:,.0f} µm³")
    print(f"   (from {n_slices_used} slices out of {NZ} total)")

    r_equiv = np.sqrt(volume_um3 / (np.pi * n_slices_used * SPACING))
    print(f"\nr_equiv from volume = {r_equiv:.1f}µm")
    print(f"Fiji cc radius = {FIJI_CC_RADIUS_UM}µm | "
          f"difference = {(r_equiv / FIJI_CC_RADIUS_UM - 1) * 100:+.1f}%")

    pkl_path = os.path.join(OUTPUT_DIR, "s1_radii_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "all_radii": all_radii,
            "all_areas": all_areas,
            "failed_slices": failed_slices,
            "volume_mm3": volume_mm3,
            "saved_profiles": saved_profiles,
        }, f)
    print(f"\n💾 Full results saved: {pkl_path}")


if __name__ == "__main__":
    main()