#!/usr/bin/env python3
"""denoising_strength_test_S5.py
اختبار: هل زيادة نافذة Z-averaging (denoising) ونافذة texture تُقلل
التذبذب العشوائي بين الشرائح؟ - نفس الـ30 شريحة، بمعلمات أقوى."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
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

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes
    FIJI_R = 71.6
    approx_cy, approx_cx = 321.8, 318.8

    z_list = np.linspace(50, nz-50, 30).astype(int)

    # === اختبار 3 إعدادات مختلفة ===
    configs = [
        {"name": "baseline (hw=15, tex=5)",  "half_window": 15, "tex_window": 5},
        {"name": "stronger Z (hw=30, tex=5)", "half_window": 30, "tex_window": 5},
        {"name": "stronger tex (hw=15, tex=9)", "half_window": 15, "tex_window": 9},
        {"name": "both stronger (hw=30, tex=9)", "half_window": 30, "tex_window": 9},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for cfg in configs:
        hw = cfg["half_window"]
        twin = cfg["tex_window"]

        peak_results = []
        grad_results = []

        for zc in z_list:
            accum = np.zeros((ny, nx)); count = 0
            for z in range(zc-hw, zc+hw+1):
                if 0 <= z < nz:
                    accum += load_single_slice(nhdr_path, z, sizes); count += 1
            den = accum/count

            tex_c = local_std(den, 9)
            cy_b, cx_b = estimate_center_blob(tex_c, approx_cy, approx_cx)
            tex_p = local_std(den, twin)
            cy_r, cx_r = refine_center_ring(tex_p, cy_b, cx_b)

            prof = savgol_filter(radial_profile(tex_p, cy_r, cx_r), 9, 2)
            r_peak = method_raw_peak(prof)
            r_grad = method_grad(prof, r_peak)

            peak_results.append(r_peak)
            grad_results.append(r_grad)

        peak_arr = np.array(peak_results)
        grad_arr = np.array(grad_results)

        print(f"\n=== {cfg['name']} ===")
        print(f"  PEAK: mean={peak_arr.mean():.1f} std={peak_arr.std():.1f} "
              f"({peak_arr.std()/peak_arr.mean()*100:.1f}%) diff={((peak_arr.mean()-FIJI_R)/FIJI_R*100):+.1f}%")
        print(f"  GRAD: mean={grad_arr.mean():.1f} std={grad_arr.std():.1f} "
              f"({grad_arr.std()/grad_arr.mean()*100:.1f}%) diff={((grad_arr.mean()-FIJI_R)/FIJI_R*100):+.1f}%")

        axes[0].plot(z_list, peak_arr, marker='o', label=cfg['name'], alpha=0.7)
        axes[1].plot(z_list, grad_arr, marker='o', label=cfg['name'], alpha=0.7)

    for ax, title in zip(axes, ['PEAK method', 'GRAD method']):
        ax.axhline(FIJI_R, color='cyan', ls='--', lw=2, label='Fiji=71.6px')
        ax.set_xlabel('z slice'); ax.set_ylabel('r (px)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig('denoising_strength_test_S5.png', dpi=150)
    plt.show()
    print("\nSaved: denoising_strength_test_S5.png")