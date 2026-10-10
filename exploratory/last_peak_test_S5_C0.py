#!/usr/bin/env python3
"""last_peak_test_S5.py
اختبار معيار جديد: 'آخر قمة محلية' (last local maximum) بدل 'القمة الأعلى مطلقاً'
- الفرضية: القمة الثانية (الأبعد) هي الحد الحقيقي لحلقة cc، وليست القمة الأعلى
  التي تتذبذب عشوائياً بين شريحة وأخرى حسب البنية الداخلية العشوائية."""

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

def find_last_significant_peak(p, search_range=(40, 105), prominence_frac=0.15):
    """
    يجد كل القمم المحلية ضمن النطاق، ثم يختار الأبعد (أكبر r) من بين
    القمم ذات prominence كافية (لتفادي التقاط ضجيج بسيط).
    """
    seg = p[search_range[0]:search_range[1]]
    local_range = np.nanmax(seg) - np.nanmin(seg)
    min_prominence = local_range * prominence_frac

    peaks, props = find_peaks(seg, prominence=min_prominence)
    if len(peaks) == 0:
        # fallback: القمة العالمية إن لم توجد قمم واضحة
        return search_range[0] + np.nanargmax(seg), []
    real_peaks = peaks + search_range[0]
    last_peak = real_peaks[-1]  # أبعد قمة (أكبر نصف قطر)
    return last_peak, real_peaks

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes
    FIJI_R = 71.6
    approx_cy, approx_cx = 321.8, 318.8
    half_window = 15
    z_list = [300, 501, 700, 150, 850]  # نطاق أوسع للتأكد من الاستقرار

    fig, axes = plt.subplots(1, len(z_list), figsize=(5*len(z_list), 5))

    print(f"{'z':>5} | {'last_peak(r)':>13} | {'diff vs Fiji':>13} | all_peaks_found")
    print("-"*70)

    results = []
    for col, zc in enumerate(z_list):
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
        last_r, all_peaks = find_last_significant_peak(prof)
        diff_pct = (last_r - FIJI_R)/FIJI_R*100

        print(f"{zc:5d} | {last_r:13d} | {diff_pct:+12.1f}% | {list(all_peaks)}")
        results.append((zc, last_r, diff_pct))

        ax = axes[col]
        ax.plot(prof, 'k', lw=2)
        for p in all_peaks:
            ax.axvline(p, color='gray', ls=':', alpha=0.5)
        ax.axvline(last_r, color='green', lw=2.5, label=f'last_peak={last_r}')
        ax.axvline(FIJI_R, color='cyan', ls='--', lw=2, label='Fiji=71.6')
        ax.set_xlim(0, 150); ax.grid(alpha=.3); ax.legend(fontsize=9)
        ax.set_title(f'z={zc}')
        ax.set_xlabel('r (px)')

    plt.tight_layout()
    plt.savefig('last_peak_test_S5.png', dpi=150)
    plt.show()

    radii = np.array([r[1] for r in results])
    print(f"\n📊 إحصائيات (5 شرائح موزعة):")
    print(f"   mean={radii.mean():.1f}px | std={radii.std():.1f}px "
          f"({radii.std()/radii.mean()*100:.1f}%)")
    print(f"   فرق المتوسط عن Fiji: {(radii.mean()-FIJI_R)/FIJI_R*100:+.1f}%")
    print("\nSaved: last_peak_test_S5.png")