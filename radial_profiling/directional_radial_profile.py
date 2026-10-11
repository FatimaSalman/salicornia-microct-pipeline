#!/usr/bin/env python3
"""
directional_radial_profile.py
Measures the same texture-versus-radius curve, but along a single ray (one angle)
instead of the full azimuthal average, in order to reveal directional anisotropy.
"""
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import uniform_filter

def local_std(img, w):
    imgf = img.astype(np.float64)
    m1 = uniform_filter(imgf, size=w)
    m2 = uniform_filter(imgf**2, size=w)
    return np.sqrt(np.maximum(m2 - m1**2, 0))

def bilinear_sample(img, xs, ys):
    H, W = img.shape
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0+1, 0, W-1);   y1 = np.clip(y0+1, 0, H-1)
    x0 = np.clip(x0, 0, W-1);     y0 = np.clip(y0, 0, H-1)
    fx = xs - x0; fy = ys - y0
    return (img[y0,x0]*(1-fx)*(1-fy) + img[y0,x1]*fx*(1-fy) +
            img[y1,x0]*(1-fx)*fy   + img[y1,x1]*fx*fy)

def find_peaks_near(radii, vals, target, window=60):
    mask = (radii > target - window) & (radii < target + window)
    if not mask.any():
        return None, None
    idx = np.argmax(vals[mask])
    return radii[mask][idx], vals[mask][idx]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--slice', type=int, required=True)
    p.add_argument('--cx', type=float, required=True)
    p.add_argument('--cy', type=float, required=True)
    p.add_argument('--voxel-size-um', type=float, default=2.5)
    p.add_argument('--max-radius-um', type=float, default=1200)
    p.add_argument('--window', type=int, default=9)
    p.add_argument('--expect-cc', type=float, default=290)
    p.add_argument('--expect-pcpt', type=float, default=480)
    p.add_argument('--expect-ptbg', type=float, default=1030)
    p.add_argument('--save-plot', default=None)
    args = p.parse_args()

    with tifffile.TiffFile(args.raw) as tif:
        img = tif.pages[args.slice].asarray().astype(np.float64)
    print(f"Slice shape: {img.shape}, dtype: {img.dtype}")

    tex = local_std(img, args.window)

    vox = args.voxel_size_um
    rmax_px = args.max_radius_um / vox
    radii_px = np.arange(1, rmax_px)
    radii_um = radii_px * vox

    angles = {'0° (droite)': 0, '90° (haut)': 90,
              '180° (gauche)': 180, '270° (bas)': 270}

    results = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.imshow(img, cmap='gray',
               vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

    colors = {'0° (droite)': 'red', '90° (haut)': 'blue',
              '180° (gauche)': 'green', '270° (bas)': 'orange'}

    for label, deg in angles.items():
        th = np.deg2rad(deg)
        xs = args.cx + radii_px * np.cos(th)
        ys = args.cy + radii_px * np.sin(th)
        vals = bilinear_sample(tex, xs, ys)

        r_cc, v_cc = find_peaks_near(radii_um, vals, args.expect_cc, 60)
        r_pcpt, v_pcpt = find_peaks_near(radii_um, vals, args.expect_pcpt, 60)
        r_ptbg, v_ptbg = find_peaks_near(radii_um, vals, args.expect_ptbg, 80)

        results[label] = (r_cc, r_pcpt, r_ptbg)

        ax2.plot(radii_um, vals, label=label, color=colors[label], alpha=0.85)
        c = colors[label]
        ax1.plot(xs, ys, color=c, lw=1.2, alpha=0.7)

    ax1.plot(args.cx, args.cy, '+', color='yellow', ms=14, mew=2)
    ax1.set_title('four directions on the raw slice')
    ax1.set_xlim(args.cx-600, args.cx+600)
    ax1.set_ylim(args.cy+600, args.cy-600)

    ax2.set_xlabel('Distance from centre (µm)')
    ax2.set_ylabel('Local texture (std dev)')
    ax2.set_title('Directional radial profiles')
    ax2.legend()
    ax2.grid(alpha=0.3)

    print("\n=== boundary results per direction (µm) ===")
    print(f"{'Direction':<15}{'cc':>10}{'pc/pt':>10}{'pt/bg':>10}")
    for label, (r_cc, r_pcpt, r_ptbg) in results.items():
        print(f"{label:<15}{r_cc:>10.1f}{r_pcpt:>10.1f}{r_ptbg:>10.1f}")

    cc_vals = [v[0] for v in results.values()]
    pcpt_vals = [v[1] for v in results.values()]
    ptbg_vals = [v[2] for v in results.values()]
    print(f"\ncc:    range = {max(cc_vals)-min(cc_vals):.1f} µm, "
          f"mean = {np.mean(cc_vals):.1f} µm")
    print(f"pc/pt: range = {max(pcpt_vals)-min(pcpt_vals):.1f} µm, "
          f"mean = {np.mean(pcpt_vals):.1f} µm")
    print(f"pt/bg: range = {max(ptbg_vals)-min(ptbg_vals):.1f} µm, "
          f"mean = {np.mean(ptbg_vals):.1f} µm")

    if args.save_plot:
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=130, bbox_inches='tight')
        print(f"\nSaved: {args.save_plot}")

if __name__ == '__main__':
    main()