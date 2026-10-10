#!/usr/bin/env python3
"""
radial_boundary_estimation.py
==============================
Estimates tissue boundary radii from a radial texture profile.

  1. Local intensity std-dev in a sliding window (optionally after Z-averaging).
  2. Average over concentric annuli centred on the stem centre.
  3. Candidate boundaries from (a) profile peaks, (b) half-prominence point on
     the descending side of each peak, (c) steepest descending slope.
  4. Also computes the radial MEAN-INTENSITY profile (tissue/air interface).
  5. Optional overlay of manual-reference radii (DIAGNOSTIC ONLY - do not use
     them to tune parameters per sample without documenting it).

Manual reference convention (Ján's Fiji table):
  cc  = full diameter           -> R_cc     = cc/2
  pc  = one-sided thickness     -> R_pc_out = R_cc + pc
  pt  = one-sided thickness     -> R_pt_out = R_pc_out + pt
"""
import argparse
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def windowed_std(img, window=9):
    img = img.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.clip(mean_sq - mean**2, 0, None)
    return np.sqrt(variance)


def load_slice_with_z_average(raw_path, slice_idx, z_half_window=0):
    with tifffile.TiffFile(raw_path) as tif:
        n_slices = len(tif.pages)
        if z_half_window <= 0:
            img = tif.pages[slice_idx].asarray()
            print(f"Loaded single slice {slice_idx} (no Z-averaging).")
            return img.astype(np.float64)

        z_start = max(0, slice_idx - z_half_window)
        z_end = min(n_slices - 1, slice_idx + z_half_window)
        n_used = z_end - z_start + 1
        print(f"Z-averaging slices [{z_start}, {z_end}] "
              f"({n_used} slices, target window ±{z_half_window}) "
              f"around slice {slice_idx}...")
        accum = None
        for z in range(z_start, z_end + 1):
            page = tif.pages[z].asarray().astype(np.float64)
            accum = page if accum is None else accum + page
        print(f"Z-averaging complete ({n_used} slices used).")
        return accum / n_used


def radial_profile(value_map, cx, cy, max_radius_px):
    """
    Mean of value_map over 1-px-wide annuli, r = 0..max_radius_px.
    Pixels beyond max_radius_px are EXCLUDED (the old version clipped them
    into the last bin, which created an artificial drop at the end).
    """
    h, w = value_map.shape
    y, x = np.indices((h, w))
    r_int = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    m = r_int <= max_radius_px
    sums = np.bincount(r_int[m], weights=value_map[m],
                       minlength=max_radius_px + 1)
    counts = np.bincount(r_int[m], minlength=max_radius_px + 1)
    prof = np.full(max_radius_px + 1, np.nan)
    ok = counts > 0
    prof[ok] = sums[ok] / counts[ok]
    return prof


def expected_radii(cc_diam, pc, pt):
    """Returns dict name -> expected radius (µm) from manual reference."""
    out = {}
    if cc_diam is None:
        return out
    r_cc = cc_diam / 2.0
    out["cc"] = r_cc
    if pc is not None:
        out["pc_out"] = r_cc + pc
        if pt is not None:
            out["pt_out"] = r_cc + pc + pt
    return out


def nearest(value, candidates):
    if len(candidates) == 0:
        return None, None
    c = np.asarray(candidates)
    i = np.argmin(np.abs(c - value))
    return c[i], c[i] - value


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--slice", type=int, required=True, help="0-based slice index")
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--voxel-size-um", type=float, default=2.5)
    ap.add_argument("--max-radius-um", type=float, default=500)
    ap.add_argument("--prominence-frac", type=float, default=0.3)
    ap.add_argument("--z-average", type=int, default=0)
    ap.add_argument("--smooth-um", type=float, default=25.0,
                    help="Savitzky-Golay window (µm) for slope-based edge detection")
    # manual reference (diagnostic overlay only)
    ap.add_argument("--manual-cc-diam-um", type=float, default=None,
                    help="Manual cc FULL diameter (µm)")
    ap.add_argument("--manual-pc-um", type=float, default=None,
                    help="Manual pc one-sided thickness (µm)")
    ap.add_argument("--manual-pt-um", type=float, default=None,
                    help="Manual pt one-sided thickness (µm)")
    ap.add_argument("--save-plot", default=None)
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    print(f"Loading data from {args.raw} ...")
    img = load_slice_with_z_average(args.raw, args.slice, z_half_window=args.z_average)
    print(f"Slice shape: {img.shape}, dtype after averaging: {img.dtype}")

    vox = args.voxel_size_um
    max_radius_px = int(args.max_radius_um / vox)
    print(f"Max radius: {args.max_radius_um} µm = {max_radius_px} px")

    # warn if annuli get clipped by the image border
    h, w = img.shape
    edge_px = min(args.cx, args.cy, w - 1 - args.cx, h - 1 - args.cy)
    if max_radius_px > edge_px:
        print(f"[WARNING] max radius ({max_radius_px} px) exceeds the distance to the "
              f"nearest image edge ({edge_px:.0f} px = {edge_px*vox:.0f} µm). "
              f"Outer annuli are partially outside the image.")

    print(f"Computing windowed std-dev (window={args.window})...")
    texture = windowed_std(img, window=args.window)

    print("Computing radial profiles...")
    tex_prof = radial_profile(texture, args.cx, args.cy, max_radius_px)
    int_prof = radial_profile(img, args.cx, args.cy, max_radius_px)

    # trim trailing NaNs (annuli fully outside the image)
    valid = np.where(~np.isnan(tex_prof))[0]
    last = valid[-1]
    tex_prof, int_prof = tex_prof[:last + 1], int_prof[:last + 1]
    radii_px = np.arange(len(tex_prof))
    radii_um = radii_px * vox

    # ---- (a) peaks ----
    prom = np.nanstd(tex_prof) * args.prominence_frac
    print(f"[DEBUG] std(profile)={np.nanstd(tex_prof):.2f}, "
          f"prominence_frac={args.prominence_frac}, prominence threshold={prom:.2f}")
    peaks, _ = find_peaks(tex_prof, prominence=prom)

    # ---- (b) half-prominence point on descending side of each peak ----
    half_edges_um = []
    if peaks.size:
        _, _, _, right_ips = peak_widths(tex_prof, peaks, rel_height=0.5)
        half_edges_um = list(right_ips * vox)

    # ---- (c) steepest descending slope (smoothed derivative) ----
    win = int(round(args.smooth_um / vox))
    win = max(5, win + (1 - win % 2))            # odd, >=5
    win = min(win, len(tex_prof) - (1 - len(tex_prof) % 2))
    smooth = savgol_filter(tex_prof, win, 2)
    deriv = np.gradient(smooth, vox)             # texture units per µm
    slope_thr = np.std(deriv) * args.prominence_frac
    sl_idx, _ = find_peaks(-deriv, prominence=slope_thr)
    slope_edges_um = list(radii_um[sl_idx])

    print("\n(a) Candidate radii - profile PEAKS:")
    for p in peaks:
        print(f"    r = {radii_um[p]:.1f} µm (texture = {tex_prof[p]:.1f})")
    if not peaks.size:
        print("    none")
    print("(b) Candidate radii - HALF-PROMINENCE on descending side of each peak:")
    for pk, e in zip(peaks, half_edges_um):
        print(f"    peak {radii_um[pk]:.1f} µm -> edge {e:.1f} µm")
    print("(c) Candidate radii - STEEPEST DESCENDING SLOPES:")
    for r in slope_edges_um:
        print(f"    r = {r:.1f} µm")

    # ---- manual reference (diagnostic) ----
    exp = expected_radii(args.manual_cc_diam_um, args.manual_pc_um, args.manual_pt_um)
    if exp:
        print("\n--- Manual reference (DIAGNOSTIC ONLY) ---")
        print(f"{'boundary':8s} {'expected':>9s} | {'nearest peak':>18s} | "
              f"{'nearest half-edge':>19s} | {'nearest slope':>17s}")
        for name, r_exp in exp.items():
            cells = []
            for cand in (list(radii_um[peaks]), half_edges_um, slope_edges_um):
                v, d = nearest(r_exp, cand)
                cells.append("-" if v is None else f"{v:.1f} ({100*d/r_exp:+.1f}%)")
            print(f"{name:8s} {r_exp:9.1f} | {cells[0]:>18s} | {cells[1]:>19s} | {cells[2]:>17s}")
        print("NOTE: choose the detection RULE before looking at these numbers; "
              "picking the closest candidate per sample is post-hoc, not automatic.")

    # ---- CSV ----
    if args.save_csv:
        data = np.column_stack([radii_um, tex_prof, int_prof, deriv])
        np.savetxt(args.save_csv, data, delimiter=",",
                   header="radius_um,texture_std,mean_intensity,texture_slope", comments="")
        print(f"Saved profiles to {args.save_csv}")

    # ---- plot ----
    if args.save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        axes[0].imshow(img, cmap='gray',
                       vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        axes[0].plot(args.cx, args.cy, 'y+', markersize=15, markeredgewidth=2)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(peaks), 1)))
        for idx, p in enumerate(peaks):
            axes[0].add_patch(plt.Circle((args.cx, args.cy), radii_px[p], fill=False,
                                         color=colors[idx], linewidth=1.2,
                                         label=f"peak {radii_um[p]:.0f} µm"))
        for name, r_exp in exp.items():
            axes[0].add_patch(plt.Circle((args.cx, args.cy), r_exp / vox, fill=False,
                                         color='k', linestyle='--', linewidth=2,
                                         label=f"manual {name} {r_exp:.0f} µm"))
        z_note = f" (Z-avg ±{args.z_average})" if args.z_average > 0 else ""
        axes[0].set_title(f"Slice{z_note} + candidates (colour) + manual ref (black dashed)")
        axes[0].legend(loc='upper right', fontsize=7)
        axes[0].axis('off')

        for ax, prof, title, ylab in (
                (axes[1], tex_prof, "Radial texture profile", "Local texture (std dev)"),
                (axes[2], int_prof, "Radial mean-intensity profile", "Mean intensity")):
            ax.plot(radii_um, prof, 'k-', linewidth=1)
            for idx, p in enumerate(peaks):
                ax.axvline(radii_um[p], color=colors[idx], linestyle=':', linewidth=1)
            for e in half_edges_um:
                ax.axvline(e, color='tab:green', linestyle='-.', linewidth=0.8, alpha=0.7)
            for name, r_exp in exp.items():
                ax.axvline(r_exp, color='k', linestyle='--', linewidth=1.5)
                ax.text(r_exp, ax.get_ylim()[1], f" {name}", rotation=90,
                        va='top', ha='right', fontsize=8)
            ax.set_xlabel("Distance from stem centre (µm)")
            ax.set_ylabel(ylab)
            ax.set_title(title + "  (dotted=peaks, green=half-edges, black=manual)")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150)
        print(f"Saved diagnostic plot to {args.save_plot}")


if __name__ == "__main__":
    main()