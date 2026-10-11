#!/usr/bin/env python3
"""S5_full_volume_grad_method.py
Final protocol for S5: Z-denoising + ring-refined centre + radial profile
+ the GRAD criterion (steepest descent after the peak), the least biased method (+3.2%),
selected after a systematic comparison of eight candidate criteria.
Uses a rolling buffer for efficiency, and saves the full results plus a labelmap for 3D rendering.

Also saves full radial-profile arrays for representative slices,
   for manuscript Figure 2, Panel D2 (max-gradient method illustration)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
from collections import deque
import os
import csv
import time

# ADDED: output directory + representative slices to save profiles for
OUTPUT_DIR = "."   # output directory; set SALICORNIA_DATA or edit this path
SAVE_PROFILE_SLICES = [200, 500, 800]


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
    info = parse_nhdr(nhdr_path)
    nx, ny, nz = sizes
    slice_bytes = nx * ny * 2
    vol_path = os.path.join(os.path.dirname(nhdr_path), info['data file'])
    with open(vol_path, 'rb') as f:
        f.seek(slice_index * slice_bytes)
        raw = f.read(slice_bytes)
    return np.frombuffer(raw, dtype=np.uint16).reshape(ny, nx)

def local_std(image, window=5):
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    return np.sqrt(np.maximum(mean_sq - mean**2, 0))

def estimate_center_blob(texture_img, approx_cy, approx_cx, search_radius=100):
    ny, nx = texture_img.shape
    y0, y1 = max(0, int(approx_cy-search_radius)), min(ny, int(approx_cy+search_radius))
    x0, x1 = max(0, int(approx_cx-search_radius)), min(nx, int(approx_cx+search_radius))
    region = texture_img[y0:y1, x0:x1]
    mask = region >= np.percentile(region, 80)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    w = region[ys, xs]
    return np.average(ys, weights=w)+y0, np.average(xs, weights=w)+x0

def refine_center_ring(texture_img, cy, cx, r_in=55, r_out=95, n_iter=3):
    ny, nx = texture_img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    for _ in range(n_iter):
        rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        ring = (rr >= r_in) & (rr <= r_out)
        w = texture_img[ring]
        w = np.where(w >= np.percentile(w, 70), w, 0)
        if w.sum() == 0:
            break
        cy = np.average(yy[ring], weights=w)
        cx = np.average(xx[ring], weights=w)
    return cy, cx

def radial_profile(texture_img, cy, cx, max_r=250, n_angles=360):
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    profile = np.full(max_r, np.nan)
    ny, nx = texture_img.shape
    for r in range(max_r):
        ys = (cy + r*np.sin(angles)).astype(int)
        xs = (cx + r*np.cos(angles)).astype(int)
        v = (ys>=0)&(ys<ny)&(xs>=0)&(xs<nx)
        if v.sum(): profile[r] = texture_img[ys[v], xs[v]].mean()
    return profile

def method_grad(p, peak_range=(50,95), bg_local=(115,135)):
    """The adopted criterion: the point of steepest descent after the peak."""
    idx = np.arange(*peak_range)
    valid = ~np.isnan(p[idx])
    if valid.sum() == 0:
        return None
    r_peak = idx[valid][np.nanargmax(p[idx][valid])]
    seg = np.arange(r_peak, bg_local[0])
    grad = np.gradient(p)
    seg_valid = seg[~np.isnan(grad[seg])]
    if len(seg_valid) == 0:
        return None
    r_grad = seg_valid[np.argmin(grad[seg_valid])]
    return r_grad

def draw_circle_on_labelmap(labelmap, cy, cx, r, value=1):
    ny, nx = labelmap.shape
    y0, y1 = max(0, int(cy-r)), min(ny, int(cy+r)+1)
    x0, x1 = max(0, int(cx-r)), min(nx, int(cx+r)+1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    labelmap[y0:y1, x0:x1][dist <= r] = value

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]
    nx, ny, nz = sizes
    spacing_um = 2.50000064
    print(f"Volume: {nx}x{ny}x{nz}, spacing={spacing_um}µm")

    # === final adopted parameters for S5 ===
    HALF_WINDOW = 15
    PEAK_RANGE = (50, 95)
    BG_LOCAL = (115, 135)
    MAX_R = 250
    SEARCH_RADIUS = 100
    RING_R_IN, RING_R_OUT = 55, 95
    APPROX_CENTER = (321.8, 318.8)
    FIJI_R = 71.6

    FULL_WINDOW = 2*HALF_WINDOW + 1
    buffer = deque(maxlen=FULL_WINDOW)
    running_sum = np.zeros((ny, nx), dtype=np.float64)

    print("filling the initial buffer...")
    for z in range(FULL_WINDOW):
        s = load_single_slice(nhdr_path, z, sizes).astype(np.float64)
        buffer.append(s)
        running_sum += s

    results = []
    labelmap_volume = np.zeros((nz, ny, nx), dtype=np.uint16)
    failed_slices = []
    saved_profiles = {}   # ADDED

    t_start = time.time()

    for zc in range(HALF_WINDOW + 1, nz - HALF_WINDOW):
        if zc > HALF_WINDOW:
            running_sum -= buffer[0]
            new_slice = load_single_slice(nhdr_path, zc + HALF_WINDOW, sizes).astype(np.float64)
            buffer.append(new_slice)
            running_sum += new_slice

        denoised = running_sum / len(buffer)

        tex9 = local_std(denoised, window=9)
        center_b = estimate_center_blob(tex9, APPROX_CENTER[0], APPROX_CENTER[1],
                                         search_radius=SEARCH_RADIUS)
        if center_b is None:
            failed_slices.append(zc)
            continue
        cy_b, cx_b = center_b

        tex5 = local_std(denoised, window=5)
        cy_r, cx_r = refine_center_ring(tex5, cy_b, cx_b, RING_R_IN, RING_R_OUT)

        profile = radial_profile(tex5, cy_r, cx_r, max_r=MAX_R)
        profile_smooth = savgol_filter(profile, window_length=9, polyorder=2)

        r_px = method_grad(profile_smooth, peak_range=PEAK_RANGE, bg_local=BG_LOCAL)
        if r_px is None:
            failed_slices.append(zc)
            continue

        results.append((zc, cy_r, cx_r, r_px))
        draw_circle_on_labelmap(labelmap_volume[zc], cy_r, cx_r, r_px)

        # ADDED: save full profile array for representative slices
        if zc in SAVE_PROFILE_SLICES:
            out_path = f"S5_profile_z{zc}.npy"
            np.save(out_path, profile_smooth)
            saved_profiles[zc] = profile_smooth
            print(f"[SAVED] z={zc} profile -> {out_path} | "
                  f"r_px={r_px} ({r_px*spacing_um:.2f}µm) | "
                  f"cy={cy_r:.1f} cx={cx_r:.1f}")

        if (zc - HALF_WINDOW) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  z={zc}/{nz-HALF_WINDOW-1} | elapsed={elapsed:.1f}s | r={r_px}px")

    elapsed = time.time() - t_start
    print(f"\ncompleted in {elapsed:.1f}s | succeeded: {len(results)} | failed: {len(failed_slices)}")

    radii_arr = np.array([r[3] for r in results])
    radii_um = radii_arr * spacing_um

    print(f"\n{'='*50}")
    print(f"cc radius statistics (S5, method=grad):")
    print(f"   mean={radii_um.mean():.2f}µm ({radii_arr.mean():.2f}px)")
    print(f"   median={np.median(radii_um):.2f}µm")
    print(f"   std={radii_um.std():.2f}µm ({radii_um.std()/radii_um.mean()*100:.1f}%)")
    print(f"   Fiji cc radius = {FIJI_R*spacing_um:.2f}µm")
    print(f"   diff vs Fiji = {(radii_um.mean()-FIJI_R*spacing_um)/(FIJI_R*spacing_um)*100:+.1f}%")

    volume_um3 = np.sum(np.pi * radii_um**2 * spacing_um)
    volume_mm3 = volume_um3 / 1e9
    print(f"\nCC VOLUME (S5) = {volume_mm3:.4f} mm³")

    height_um = len(results) * spacing_um
    r_equiv = np.sqrt(volume_um3 / (np.pi * height_um))
    print(f"   r_equiv = {r_equiv:.2f}µm | height={height_um:.0f}µm")

    with open('S5_grad_method_results.csv', 'w', newline='') as f:  # path
        writer = csv.writer(f)
        writer.writerow(['z','cy','cx','r_px','r_um'])
        for zc, cy, cx, r_px in results:
            writer.writerow([zc, f"{cy:.2f}", f"{cx:.2f}", r_px, f"{r_px*spacing_um:.2f}"])
    print("Saved: S5_grad_method_results.csv")

    np.save('S5_cc_labelmap.npy', labelmap_volume)  # path

    # ADDED: save saved_profiles dict as well, for later re-plotting
    import pickle
    with open('S5_profiles_dict.pkl', 'wb') as f:
        pickle.dump(saved_profiles, f)
    print("Saved: S5_profiles_dict.pkl")

    check_zs = [100, 300, 500, 700, 900]
    check_zs = [z for z in check_zs if HALF_WINDOW < z < nz-HALF_WINDOW]
    fig, axes = plt.subplots(1, len(check_zs), figsize=(4*len(check_zs), 5))
    for i, zc in enumerate(check_zs):
        slice_raw = load_single_slice(nhdr_path, zc, sizes)
        axes[i].imshow(slice_raw, cmap='gray')
        match = [r for r in results if r[0] == zc]
        if match:
            _, cy, cx, r_px = match[0]
            axes[i].add_patch(plt.Circle((cx,cy), r_px, color='red', fill=False, lw=2))
            axes[i].scatter([cx],[cy], c='red', s=30, marker='+')
        axes[i].set_title(f'z={zc}, r={r_px if match else "N/A"}px')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('S5_visual_check_grad.png', dpi=150)  # path
    plt.show()
    print("Saved: S5_visual_check_grad.png")