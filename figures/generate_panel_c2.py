#!/usr/bin/env python3
"""
generate_panel_c2.py
====================
Panel C2 of Figure 2: the local texture map (windowed standard deviation) of
Ciechocinek 1000 mM, slice 492.

This is the second of the two sample-adaptive preprocessing branches shown in
panel C. Panel C1 shows Z-axis averaging, used for samples with a small,
low-contrast central cylinder; this panel shows the texture branch, used for
samples with a large, continuous central cylinder.

Slice 492 is deliberate: it is the same slice whose texture histogram and Otsu
threshold are shown in panel D3, so a reader can follow C2 -> D3 as one
sequence.

The texture volume itself is produced by
texture/compute_texture_volume_streaming_C1000.py.

Usage
-----
    export SALICORNIA_DATA=/path/to/data
    python3 generate_panel_c2.py

    # or point at the file directly
    python3 generate_panel_c2.py --texture Salicornia_8_texture.nhdr
"""
import argparse
import os
import sys

import numpy as np

DATA = os.environ.get("SALICORNIA_DATA", ".")

SLICE_INDEX = 492          # 0-based; the slice used in panel D3
OUT_NAME = "PanelC2_s8_texture_z492.tif"
DPI = 600
PANEL_WIDTH_IN = 2.3       # one third of the 180 mm figure width
PCT_LOW, PCT_HIGH = 1.0, 99.5   # display stretch, percentile based


def load_slice(path, index):
    """Read one slice from a NRRD/NHDR texture volume without loading it all."""
    try:
        import SimpleITK as sitk
    except ImportError:
        sys.exit("SimpleITK is required: pip install SimpleITK")

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    size = list(reader.GetSize())           # x, y, z
    if index >= size[2]:
        sys.exit(f"slice {index} is out of range; the volume has {size[2]} slices")

    reader.SetExtractIndex([0, 0, index])
    reader.SetExtractSize([size[0], size[1], 0])
    return sitk.GetArrayFromImage(reader.Execute()).squeeze(), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture",
                    default=os.path.join(DATA, "Salicornia_8/Salicornia_8_texture.nhdr"),
                    help="texture volume for C-1000 (.nhdr or .nrrd)")
    ap.add_argument("--slice", type=int, default=SLICE_INDEX)
    ap.add_argument("--out", default=OUT_NAME)
    args = ap.parse_args()

    if not os.path.exists(args.texture):
        sys.exit(f"texture volume not found: {args.texture}\n"
                 "Run texture/compute_texture_volume_streaming_C1000.py first, "
                 "or pass --texture.")

    img, size = load_slice(args.texture, args.slice)
    print(f"volume {size[0]}x{size[1]}x{size[2]}, slice {args.slice}, "
          f"range {img.min():.1f}-{img.max():.1f}")

    # Percentile stretch. A plain min-max stretch is dominated by a handful of
    # extreme texture values at the outer edge and leaves the tissue flat.
    lo, hi = np.percentile(img, [PCT_LOW, PCT_HIGH])
    disp = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    print(f"display stretch: {lo:.1f} to {hi:.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = disp.shape
    fig_h = PANEL_WIDTH_IN * h / w
    fig = plt.figure(figsize=(PANEL_WIDTH_IN, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.axis("off")

    fig.savefig(args.out, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=DPI)
    print(f"saved {args.out} and .png at {DPI} dpi")

    # The Otsu threshold reported in the manuscript for this volume, printed
    # here so the panel and the number in Table S4 can be checked together.
    try:
        from skimage.filters import threshold_otsu
        print(f"Otsu threshold on this slice: {threshold_otsu(img):.2f} "
              f"(full-volume value used in the manuscript: 2841.73)")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
