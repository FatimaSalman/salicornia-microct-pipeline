#!/usr/bin/env python3
"""edge_visual_check_S5.py
1) تحسين المركز عبر centroid موزون ضمن حلقة cc فقط (r∈[55,95])
2) حساب 4 معايير حافة: raw peak / max-gradient / half-max محلي / half-max بعيد
3) رسم الدوائر المرشحة فوق الشريحة الخام المُنظّفة مكبّرة، على 3 مستويات z"""

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
    """centroid موزون ضمن حلقة cc فقط - أقل تأثراً بالفصوص الخارجية."""
    ny, nx = texture_img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    for _ in range(n_iter):
        rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        ring = (rr >= r_in) & (rr <= r_out)
        w = texture_img[ring]
        w = np.where(w >= np.percentile(w, 70), w, 0)  # أعلى 30% داخل الحلقة
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

def edge_criteria(p, peak_range=(50,95), bg_local=(115,135), bg_far=(220,250)):
    idx = np.arange(*peak_range)
    r_peak = idx[np.argmax(p[idx])]
    peak_val = p[r_peak]
    # أشد انحدار بعد القمة (حتى بداية الخلفية المحلية)
    seg = np.arange(r_peak, bg_local[0])
    grad = np.gradient(p)
    r_grad = seg[np.argmin(grad[seg])]
    def half_max(bg_range):
        bg = p[bg_range[0]:bg_range[1]].mean()
        thr = (peak_val+bg)/2
        for r in range(r_peak, bg_range[0]):
            if p[r] <= thr: return r
        return None
    return dict(peak=r_peak, grad=r_grad,
                hm_local=half_max(bg_local), hm_far=half_max(bg_far))

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes
    FIJI_R = 71.6
    approx_cy, approx_cx = 321.8, 318.8
    half_window = 15
    z_list = [300, 501, 700]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

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
        prof_b = savgol_filter(radial_profile(tex5, cy_b, cx_b), 9, 2)
        ec = edge_criteria(prof)

        print(f"\n=== z={zc} ===")
        print(f"center blob=({cx_b:.1f},{cy_b:.1f})  ring-refined=({cx_r:.1f},{cy_r:.1f})  "
              f"shift={np.hypot(cx_r-cx_b, cy_r-cy_b):.1f}px")
        print(f"peak height blob-center={np.nanmax(prof_b[50:95]):.0f} | ring-center={np.nanmax(prof[50:95]):.0f} "
              f"(الأعلى = مركز أدق)")
        for k, v in ec.items():
            if v is not None:
                print(f"  {k:9s}: r={v:3d}px  ({(v-FIJI_R)/FIJI_R*100:+5.1f}% vs Fiji)")

        # لوحة علوية: مكبّرة على cc مع الدوائر المرشحة
        ax = axes[0, col]
        pad = 130
        ax.imshow(den, cmap='gray',
                  vmin=np.percentile(den,1), vmax=np.percentile(den,99))
        colors = dict(peak='lime', grad='yellow', hm_local='orange', hm_far='red')
        for k, v in ec.items():
            if v is not None:
                ax.add_patch(plt.Circle((cx_r, cy_r), v, color=colors[k], fill=False,
                                        lw=1.5, label=f'{k}={v}px'))
        ax.add_patch(plt.Circle((cx_r, cy_r), FIJI_R, color='cyan', fill=False,
                                lw=1.5, ls='--', label=f'Fiji={FIJI_R}px'))
        ax.plot(cx_r, cy_r, 'r+', ms=12, mew=2)
        ax.set_xlim(cx_r-pad, cx_r+pad); ax.set_ylim(cy_r+pad, cy_r-pad)
        ax.set_title(f'z={zc} (denoised, zoom)'); ax.legend(fontsize=8, loc='lower right')

        # لوحة سفلية: البروفايل مع الخطوط
        ax = axes[1, col]
        ax.plot(prof, 'k', lw=2)
        for k, v in ec.items():
            if v is not None: ax.axvline(v, color=colors[k], ls=':', lw=2, label=f'{k}={v}')
        ax.axvline(FIJI_R, color='cyan', ls='--', lw=2, label='Fiji')
        ax.set_xlim(0, 200); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_xlabel('r (px)'); ax.set_title(f'radial profile z={zc}')

    plt.tight_layout()
    plt.savefig('edge_visual_check_S5.png', dpi=150)
    plt.show()
    print("\nSaved: edge_visual_check_S5.png")