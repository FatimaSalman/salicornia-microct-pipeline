import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

 
# ============================================================
# Sample configs, pulled from the volumetric logs.
# FILL IN: pred_path and rescale_factor for each sample below.
# ============================================================
samples = {
    "S1": dict(
        pred_path="/path/to/Salicornia 1, Inowroclaw 0 mM/image_folder/Salicornia1_/MLresults/Salicornia1_fullstack_prediction.tif",  # already known, fill if needed
        z=500,
        com_y_full=520, com_x_full=591,
        r_cc_um=87.5,
        rescale_factor=2,   # confirmed for S1
        voxel_um_original=2.5,
    ),
    "S4": dict(
        pred_path="/path/to/Salicornia 4, Inowroclaw 1000 mM/image_folder/Salicornia4_/MLresults/Salicornia4_fullstack_prediction.tif",
        z=400,  # near the logged center Z~400
        com_y_full=795, com_x_full=1051,
        r_cc_um=279,
        rescale_factor=4,   # <-- confirm: same as used when you trained S4
        voxel_um_original=2.50001997,
    ),
    "S5": dict(
        pred_path="/path/to/Salicornia_5/image_folder/Salicornia5_/MLresults/Salicornia5_fullstack_prediction.tif",  # <-- fill
        z=501,  # near the logged center Z~501
        com_y_full=325, com_x_full=320,
        r_cc_um=90,
        rescale_factor=2,   # <-- confirm
        voxel_um_original=2.5000006400000005,
    ),
    "S8": dict(
        pred_path="/path/to/Salicornia_8/image_folder/Salicornia8_/MLresults/Salicornia8_fullstack_prediction.tif",  # <-- fill
        z=492,  # logged center Z=492
        com_y_full=539, com_x_full=441,
        r_cc_um=215,
        rescale_factor=2,   # <-- confirm
        voxel_um_original=2.50002788,
    ),
}


for sample_name, cfg in samples.items():
    if "/path/to/" in cfg["pred_path"]:
        print(f"[{sample_name}] SKIPPED - fill the pred_pat first.")
        continue
    pred = tifffile.imread(cfg["pred_path"])
    print(f"[{sample_name}] Full stack shape:", pred.shape)

    pred_slice = pred[cfg["z"]]
    unique_vals = np.unique(pred_slice)
    print(f"[{sample_name}] Unique values in slice {cfg['z']}:", unique_vals)

    # Effective voxel size and center after rescaling
    voxel_um_eff = cfg["voxel_um_original"] * cfg["rescale_factor"]
    com_y = cfg["com_y_full"] / cfg["rescale_factor"]
    com_x = cfg["com_x_full"] / cfg["rescale_factor"]
    r_cc_vox = cfg["r_cc_um"] / voxel_um_eff

    cmap = plt.get_cmap('tab10', len(unique_vals))
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    idx_slice = np.vectorize(val_to_idx.get)(pred_slice)

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(idx_slice, cmap=cmap, vmin=-0.5, vmax=len(unique_vals) - 0.5)

    cbar = plt.colorbar(im, ax=ax, ticks=range(len(unique_vals)))
    cbar.ax.set_yticklabels([str(v) for v in unique_vals])
    cbar.set_label("Predicted class value")

    circle = plt.Circle((com_x,com_y), r_cc_vox, fill= False,
                        color ='white', linewidth= 2, linestyle ='--',
                        label=f'cc radius ={cfg["r_cc_um"]}um (from labels)')
    ax.add_patch(circle)
    ax.plot(com_x, com_y, '+', color='yellow', markersize=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"{sample_name} - predicted classes, slice {cfg['z']}\nWhich color falls inside the dashed cc circles?")

    plt.tight_layout()
    out_name = f"{sample_name}_prediction_class_identification.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"Saved {out_name}")