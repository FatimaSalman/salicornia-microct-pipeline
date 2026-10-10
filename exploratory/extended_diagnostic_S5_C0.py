#!/usr/bin/env python3
"""extended_diagnostic_S5.py
تشخيص موسّع: تطبيق كل المعايير الخمسة (peak/grad/hm_local/hm_far/last_peak)
على ~30 شريحة موزعة بانتظام عبر كامل الفوليوم، لتحديد الطريقة الأكثر
استقراراً (أقل std) والأقرب لـFiji (أقل bias) قبل معالجة الفوليوم الكامل."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter, find_peaks
import os

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

def method_raw_peak(p, rng=(50,95)):
    idx = np.arange(*rng)
    return idx[np.nanargmax(p[idx])]

def method_grad(p, r_peak, bg_local=(115,135)):
    seg = np.arange(r_peak, bg_local[0])
    grad = np.gradient(p)
    return seg[np.argmin(grad[seg])]

def method_half_max(p, r_peak, peak_val, bg_range):
    bg = np.nanmean(p[bg_range[0]:bg_range[1]])
    thr = (peak_val+bg)/2
    for r in range(r_peak, bg_range[0]):
        if p[r] <= thr:
            return r
    return None

def method_last_peak(p, search_range=(40,105), prominence_frac=0.15):
    seg = p[search_range[0]:search_range[1]]
    local_range = np.nanmax(seg) - np.nanmin(seg)
    peaks, _ = find_peaks(seg, prominence=local_range*prominence_frac)
    if len(peaks)==0:
        return search_range[0] + np.nanargmax(seg)
    return peaks[-1] + search_range[0]

def method_shoulder_70pct(p, r_peak, peak_val, bg_range=(115,135), frac=0.7):
    """نقطة تقاطع عند 70% من المسافة بين القمة والخلفية (بدل 50% half-max)
    - أكثر تحفظاً، يلتقط 'الكتف' بدل الانتظار لمنتصف الهبوط الكامل."""
    bg = np.nanmean(p[bg_range[0]:bg_range[1]])
    thr = peak_val - frac*(peak_val-bg)
    for r in range(r_peak, bg_range[0]):
        if p[r] <= thr:
            return r
    return None

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes
    FIJI_R = 71.6
    approx_cy, approx_cx = 321.8, 318.8
    half_window = 15

    # 30 شريحة موزعة بانتظام (تفادي الحواف حيث الشكل قد يتغير كثيراً)
    z_list = np.linspace(50, nz-50, 30).astype(int)

    all_results = {m: [] for m in ['peak','grad','hm_local','hm_far','last_peak','shoulder70']}

    print(f"{'z':>5} | {'peak':>6} {'grad':>6} {'hm_loc':>7} {'hm_far':>7} {'lastpk':>7} {'sh70':>6}")
    print("-"*70)

    for zc in z_list:
        accum = np.zeros((ny, nx)); count = 0
        for z in range(zc-half_window, zc+half_window+1):
            if 0 <= z < nz:
                accum += load_single_slice(nhdr_path, z, sizes); count += 1
        den = accum/count

        tex9 = local_std(den, 9)
        cy_b, cx_b = estimate_center_blob(tex9, approx_cy, approx_cx)
        tex5 = local_std(den, 5)
        cy_r, cx_r = refine_center_ring(tex5, cy_b, cx_b)

        prof = savgol_filter(radial_profile(tex5, cy_r, cx_r), 9, 2)

        r_peak = method_raw_peak(prof)
        peak_val = prof[r_peak]
        r_grad = method_grad(prof, r_peak)
        r_hml = method_half_max(prof, r_peak, peak_val, (115,135))
        r_hmf = method_half_max(prof, r_peak, peak_val, (220,250))
        r_last = method_last_peak(prof)
        r_sh70 = method_shoulder_70pct(prof, r_peak, peak_val, (115,135))

        vals = dict(peak=r_peak, grad=r_grad, hm_local=r_hml, hm_far=r_hmf,
                    last_peak=r_last, shoulder70=r_sh70)
        for k,v in vals.items():
            if v is not None: all_results[k].append(v)

        print(f"{zc:5d} | {r_peak:6d} {r_grad:6d} {str(r_hml):>7} {str(r_hmf):>7} "
              f"{r_last:7d} {str(r_sh70):>6}")

    print(f"\n{'='*70}")
    print(f"📊 ملخص إحصائي (n={len(z_list)} شريحة):")
    print(f"{'Method':<12}{'mean':>8}{'median':>8}{'std':>8}{'std%':>8}{'diff_vs_Fiji%':>16}")
    for m, vals in all_results.items():
        arr = np.array(vals)
        if len(arr) == 0:
            print(f"{m:<12}  (لا توجد نتائج)")
            continue
        mean, med, std = arr.mean(), np.median(arr), arr.std()
        diff = (mean-FIJI_R)/FIJI_R*100
        print(f"{m:<12}{mean:8.1f}{med:8.1f}{std:8.1f}{std/mean*100:7.1f}%{diff:15.1f}%")

    # رسم مقارن لكل الطرق
    fig, ax = plt.subplots(figsize=(12,6))
    for m, vals in all_results.items():
        ax.plot(z_list[:len(vals)], vals, marker='o', label=m, alpha=0.7)
    ax.axhline(FIJI_R, color='cyan', ls='--', lw=2, label='Fiji=71.6px')
    ax.set_xlabel('z slice'); ax.set_ylabel('r (px)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('مقارنة استقرار الطرق عبر 30 شريحة موزعة - S5')
    plt.tight_layout()
    plt.savefig('extended_diagnostic_S5.png', dpi=150)
    plt.show()
    print("\nSaved: extended_diagnostic_S5.png")