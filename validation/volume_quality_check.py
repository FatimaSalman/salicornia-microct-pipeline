#!/usr/bin/env python3
"""
volume_quality_check.py
========================
Computes objective image-quality metrics from a raw VGStudio .vol volume
(paired with its .nhdr header produced by vgl_to_nrrd.py), to give a
data-driven answer to "is this scan good enough to recognize internal
tissue structures?" instead of a subjective visual impression.

Metrics computed (per volume):
  - Global intensity histogram stats (mean, std, min, max, percentiles)
  - Background noise (std-dev in an empty corner region -> lower is better)
  - Signal region stats (std-dev + mean inside the sample -> for contrast)
  - Global CNR estimate = |mean_signal - mean_background| / noise_std
  - Simple edge-sharpness proxy: mean gradient magnitude across mid-slices
    (higher generally means crisper boundaries between structures)

Requires: numpy  (pip install numpy --break-system-packages if missing)

Usage
-----
    python3 volume_quality_check.py "/path/to/Salicornia_5.nhdr"
    python3 volume_quality_check.py "/path/to/Salicornia_8.nhdr"

You can also compare two volumes directly:
    python3 volume_quality_check.py "vol1.nhdr" "vol2.nhdr"
"""

import sys
import os
import re
import numpy as np


NRRD_TYPE_TO_NUMPY = {
    "unsigned char": np.uint8,
    "signed char": np.int8,
    "unsigned short": np.uint16,
    "short": np.int16,
    "unsigned int": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


def parse_nhdr(nhdr_path):
    info = {}
    with open(nhdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "NRRD0004":
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()

    dtype = NRRD_TYPE_TO_NUMPY.get(info["type"])
    if dtype is None:
        raise ValueError(f"Unsupported type: {info['type']}")

    sizes = [int(v) for v in info["sizes"].split()]
    endian = info.get("endian", "little")
    data_file = info["data file"]
    byte_skip = int(info.get("byte skip", "0"))

    folder = os.path.dirname(os.path.abspath(nhdr_path))
    data_path = os.path.join(folder, data_file)

    return {
        "dtype": dtype,
        "sizes": sizes,  # [X, Y, Z]
        "endian": endian,
        "data_path": data_path,
        "byte_skip": byte_skip,
    }


def load_volume_memmap(nhdr_path):
    info = parse_nhdr(nhdr_path)
    x, y, z = info["sizes"]
    dtype = info["dtype"]
    if info["endian"] == "big":
        dtype = np.dtype(dtype).newbyteorder(">")

    if not os.path.exists(info["data_path"]):
        raise FileNotFoundError(f"Raw data file not found: {info['data_path']}")

    # numpy memmap avoids loading the whole (multi-GB) file into RAM
    arr = np.memmap(info["data_path"], dtype=dtype, mode="r",
                     offset=info["byte_skip"], shape=(z, y, x))
    return arr


def otsu_threshold(values, n_bins=256):
    """Simple Otsu's method implementation (no extra dependencies)."""
    hist, bin_edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight_bg = np.cumsum(hist)
    weight_fg = np.cumsum(hist[::-1])[::-1]

    sum_total = (hist * bin_centers).sum()
    sum_bg = np.cumsum(hist * bin_centers)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    between_var = np.nan_to_num(between_var)
    idx = np.argmax(between_var)
    return bin_centers[idx]


def analyze(nhdr_path, n_slices=5):
    print(f"\n=== {os.path.basename(nhdr_path)} ===")
    arr = load_volume_memmap(nhdr_path)
    z, y, x = arr.shape
    print(f"Volume shape (Z,Y,X): {z} x {y} x {x}")

    slice_idxs = np.linspace(int(z * 0.2), int(z * 0.8), n_slices, dtype=int)
    slices = [np.asarray(arr[i]).astype(np.float32) for i in slice_idxs]
    stack = np.stack(slices)
    print(f"Sampled slices: {list(slice_idxs)}")

    # Global stats
    mean = stack.mean()
    std = stack.std()
    p1, p50, p99 = np.percentile(stack, [1, 50, 99])
    print(f"Intensity  mean={mean:.1f}  std={std:.1f}  "
          f"p1={p1:.1f}  median={p50:.1f}  p99={p99:.1f}")

    # Histogram-based separation instead of assuming a spatial region:
    # Otsu threshold splits voxels into two populations by intensity,
    # regardless of where they sit in the frame. This works even when
    # the sample fills most/all of the field of view.
    flat = stack.ravel()
    # subsample for speed if huge
    if flat.size > 5_000_000:
        rng = np.random.default_rng(0)
        flat = rng.choice(flat, size=5_000_000, replace=False)

    thresh = otsu_threshold(flat)
    low_pop = flat[flat < thresh]
    high_pop = flat[flat >= thresh]

    print(f"Otsu threshold: {thresh:.1f}  "
          f"(low population: {low_pop.size/flat.size*100:.1f}% of voxels, "
          f"high population: {high_pop.size/flat.size*100:.1f}%)")

    low_mean, low_std = low_pop.mean(), low_pop.std()
    high_mean, high_std = high_pop.mean(), high_pop.std()
    print(f"Low-intensity population : mean={low_mean:.1f}  std={low_std:.1f}")
    print(f"High-intensity population: mean={high_mean:.1f}  std={high_std:.1f}")

    noise_std = min(low_std, high_std)  # use the tighter population as noise proxy
    if noise_std > 0:
        cnr = abs(high_mean - low_mean) / noise_std
    else:
        cnr = float("inf")
    print(f"Estimated CNR (Otsu-based, contrast-to-noise): {cnr:.2f}")

    # Edge sharpness proxy: mean gradient magnitude on a central patch
    sx0, sx1 = int(x * 0.35), int(x * 0.65)
    sy0, sy1 = int(y * 0.35), int(y * 0.65)
    fg_patch = stack[:, sy0:sy1, sx0:sx1]
    gy, gx = np.gradient(fg_patch.mean(axis=0))
    grad_mag = np.sqrt(gx**2 + gy**2).mean()
    print(f"Mean edge/gradient magnitude (sharpness proxy): {grad_mag:.2f}")

    print("\nRule-of-thumb interpretation:")
    print("  CNR > 5   : structures should be clearly separable, good for AI/segmentation")
    print("  CNR 2-5   : structures visible but noisy, denoising recommended before AI")
    print("  CNR < 2   : structures likely hard to distinguish reliably from noise")

    return {
        "mean": mean, "std": std, "thresh": thresh,
        "low_mean": low_mean, "low_std": low_std,
        "high_mean": high_mean, "high_std": high_std,
        "cnr": cnr, "grad_mag": grad_mag,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 volume_quality_check.py file1.nhdr [file2.nhdr ...]")
        sys.exit(1)

    results = {}
    for path in sys.argv[1:]:
        results[path] = analyze(path)

    if len(results) > 1:
        print("\n\n=== Comparison ===")
        for path, r in results.items():
            print(f"{os.path.basename(path):30s}  CNR={r['cnr']:.2f}   "
                  f"overall_mean={r['mean']:.1f}   sharpness={r['grad_mag']:.2f}")
        means = [r["mean"] for r in results.values()]
        spread_pct = (max(means) - min(means)) / min(means) * 100
        print(f"\nOverall intensity mean differs by {spread_pct:.1f}% between volumes.")
        if spread_pct > 10:
            print("  [!] This is a large difference. Before trusting any AI-detected "
                  "class difference as biological, verify the scans used the same "
                  "X-ray tube settings / calibration, or plan to normalize intensities "
                  "(e.g. histogram matching) before training.")


if __name__ == "__main__":
    main()