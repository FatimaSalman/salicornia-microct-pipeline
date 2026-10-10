#!/usr/bin/env python3
"""compute_texture_volume_full.py
حساب texture map (windowed std-dev) لكامل فوليوم S8
وحفظه كـ .nrrd محاذي تمامًا للفوليوم الأصلي (نفس spacing/origin/direction)
عشان نقدر نحمّله بـSlicer جنب الأصلي مباشرة."""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter
import time

def compute_texture_volume(nhdr_path, window=9, out_path="Salicornia_8_texture.nrrd",
                             save_as_uint16=True):
    print(f"Reading {nhdr_path} via SimpleITK ...")
    img = sitk.ReadImage(nhdr_path)
    arr = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X) = (985, 1079, 884)
    print(f"Volume shape (Z,Y,X): {arr.shape}, dtype: {arr.dtype}")

    n_slices = arr.shape[0]
    texture = np.zeros_like(arr, dtype=np.float32)

    t0 = time.time()
    for z in range(n_slices):
        s = arr[z].astype(np.float64)
        mean = uniform_filter(s, size=window)
        mean_sq = uniform_filter(s**2, size=window)
        var = np.maximum(mean_sq - mean**2, 0)
        texture[z] = np.sqrt(var).astype(np.float32)
        if z % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Slice {z}/{n_slices} done ({elapsed:.1f}s elapsed)")

    print(f"Total texture computation time: {time.time()-t0:.1f}s")
    print(f"Texture value range: {texture.min():.1f} - {texture.max():.1f}")

    if save_as_uint16:
        # توفير مساحة قرص (نصف الحجم)؛ آمن لأن قيم std-dev هنا أقل بكثير من 65535
        texture_out = np.clip(texture, 0, 65535).astype(np.uint16)
    else:
        texture_out = texture

    texture_img = sitk.GetImageFromArray(texture_out)
    texture_img.CopyInformation(img)  # ينسخ spacing/origin/direction بالضبط من الأصلي
    sitk.WriteImage(texture_img, out_path)
    print(f"Saved aligned texture volume to: {out_path}")

if __name__ == "__main__":
    compute_texture_volume("Salicornia_8.nhdr", window=9,
                            out_path="Salicornia_8_texture.nrrd",
                            save_as_uint16=True)