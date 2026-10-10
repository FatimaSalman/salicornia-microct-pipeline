
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

from pathlib import Path
import numpy as np
import nrrd
import tifffile
from scipy import ndimage as ndi


# -----------------------------
# Label values compatible with old RF workflow
# -----------------------------
LABEL_FAR_NONCC = 51
LABEL_NEAR_NONCC = 102
LABEL_CC = 153


def read_nrrd_mask_as_zyx(mask_path):
    """
    NRRD masks are stored/read as (X, Y, Z).
    RF/TIFF stack convention is (Z, Y, X).
    Therefore we transpose (X,Y,Z) -> (Z,Y,X).
    """
    mask_xyz, header = nrrd.read(str(mask_path))
    print(f"\nReading mask: {mask_path}")
    print("Original NRRD shape (X,Y,Z):", mask_xyz.shape)
    print("Unique values:", np.unique(mask_xyz))

    mask_zyx = np.transpose(mask_xyz, (2, 1, 0)).astype(bool)
    print("Converted shape (Z,Y,X):", mask_zyx.shape)

    return mask_zyx


def make_context_label_slice(cc_mask_slice, voxel_um=2.5,
                             gap_um=5,
                             near_um=50,
                             far_um=125):
    """
    Build one training label slice from a final cc binary mask.

    153 = confident cc interior
    102 = near non-cc context outside cc
    51  = farther non-cc context outside cc
    0   = ignored/unlabeled

    A small gap around the cc boundary is left unlabeled to avoid ambiguous edge pixels.
    """

    cc = cc_mask_slice.astype(bool)

    # Convert physical distances to pixels
    gap_px = max(1, int(round(gap_um / voxel_um)))
    near_px = max(gap_px + 1, int(round(near_um / voxel_um)))
    far_px = max(near_px + 1, int(round(far_um / voxel_um)))

    # Distance inside cc and outside cc
    dist_inside = ndi.distance_transform_edt(cc)
    dist_outside = ndi.distance_transform_edt(~cc)

    # Confident cc: slightly eroded cc, avoiding uncertain boundary
    cc_train = dist_inside > gap_px

    # Near/far non-cc context outside cc
    near_noncc = (dist_outside > gap_px) & (dist_outside <= near_px)
    far_noncc = (dist_outside > near_px) & (dist_outside <= far_px)

    lbl = np.zeros(cc.shape, dtype=np.uint8)
    lbl[far_noncc] = LABEL_FAR_NONCC
    lbl[near_noncc] = LABEL_NEAR_NONCC
    lbl[cc_train] = LABEL_CC

    return lbl


def build_label_stack(sample_name, mask_path, reference_slices,
                      output_path, voxel_um=2.5):
    mask_zyx = read_nrrd_mask_as_zyx(mask_path)

    zdim = mask_zyx.shape[0]
    print(f"{sample_name}: Z dimension = {zdim}")

    label_slices = []

    for z in reference_slices:
        if z < 0 or z >= zdim:
            raise ValueError(f"{sample_name}: slice {z} out of range 0-{zdim-1}")

        cc_slice = mask_zyx[z]
        lbl = make_context_label_slice(cc_slice, voxel_um=voxel_um)

        label_slices.append(lbl)

        vals, counts = np.unique(lbl, return_counts=True)
        count_dict = dict(zip(vals.tolist(), counts.tolist()))
        print(f"{sample_name} z={z}: labels/counts = {count_dict}")

    label_stack = np.stack(label_slices, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(output_path),
        label_stack,
        photometric="minisblack"
    )

    print(f"\nSaved {sample_name} labelled stack:")
    print(output_path)
    print("Shape:", label_stack.shape)
    print("Unique:", np.unique(label_stack))


# ==========================================================
# EDIT THESE PATHS
# ==========================================================

BASE = Path(DATA)

samples = {
    "S1": {
        "mask": BASE / "Salicornia 1, Inowroclaw 0 mM" / "S1_cc_labelmap_proper.nrrd",
        "slices": [50, 130, 220, 310, 400, 500, 600, 700, 780, 850, 920, 970],
        "voxel_um": 2.50002,
        "out": BASE / "Salicornia 1, Inowroclaw 0 mM" / "image_folder" / "Salicornia1_" / "labelled-stack.tif",
    },

    "S4": {
        "mask": BASE / "Salicornia 4, Inowroclaw 1000 mM" / "cc_lm_v5.nrrd",
        "slices": [80, 200, 320, 480, 600, 720],
        "voxel_um": 2.50002,
        "out": BASE / "Salicornia 4, Inowroclaw 1000 mM" / "image_folder" / "Salicornia4_" / "labelled-stack.tif",
    },

    "S5": {
        "mask": BASE / "Salicornia_5" / "S5_cc_labelmap_bufferfix.nrrd",
        "slices": [80, 200, 400, 600, 800, 950],
        "voxel_um": 2.50000,
        "out": BASE / "Salicornia_5" / "image_folder" / "Salicornia5_" / "labelled-stack.tif",
    },

    "S8": {
        "mask": BASE / "Salicornia_8" / "cc_labelmap.nrrd",
        "slices": [150, 300, 450, 550, 700, 850],
        "voxel_um": 2.50003,
        "out": BASE / "Salicornia_8" / "image_folder" / "Salicornia8_" / "labelled-stack.tif",
    },
}


if __name__ == "__main__":
    for sample_name, cfg in samples.items():
        build_label_stack(
            sample_name=sample_name,
            mask_path=cfg["mask"],
            reference_slices=cfg["slices"],
            output_path=cfg["out"],
            voxel_um=cfg["voxel_um"]
        )