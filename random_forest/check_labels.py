
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import tifffile
import numpy as np

labels_path = os.path.join(DATA, "Salicornia_8/image_folder/Salicornia8_/labelled-stack.tif")
labels_stack = tifffile.imread(labels_path)

# Full RF prediction stack (984 slices, full volume)
prediction_path = os.path.join(DATA, "Salicornia_8/image_folder/Salicornia8_/MLresults/Salicornia8_fullstack_prediction.tif")
prediction_stack = np.squeeze(tifffile.imread(prediction_path))

# Candidate training slice indices, in ascending order (from Table S3 / Methods)
training_slices_full_volume = [151, 301, 451, 551, 701, 851]

print("Labels stack shape:", labels_stack.shape)
print("Prediction stack shape:", prediction_stack.shape)

# For each labels_stack index, compare its CC region against the
# prediction at the CORRESPONDING full-volume z, to check consistency
# (a rough sanity check - they should roughly overlap since z=451 was
# literally used to train the model that produced this same prediction)
print("\nSanity check: comparing labels_stack[i] (as CC=153) against "
      "prediction_stack[full_z] (as CC=-1):\n")

for i, full_z in enumerate(training_slices_full_volume):
    label_cc_mask = labels_stack[i] == 153
    pred_cc_mask = prediction_stack[full_z] == -1  # CC class confirmed earlier

    label_count = np.count_nonzero(label_cc_mask)
    pred_count = np.count_nonzero(pred_cc_mask)

    # crude overlap ratio just as a sanity indicator
    if label_cc_mask.shape == pred_cc_mask.shape:
        intersection = np.count_nonzero(label_cc_mask & pred_cc_mask)
        overlap_pct = 100 * intersection / max(label_count, 1)
    else:
        overlap_pct = None

    print(f"index={i} -> full_z={full_z}: "
          f"label_CC_pixels={label_count}, pred_CC_pixels={pred_count}, "
          f"shapes_match={label_cc_mask.shape == pred_cc_mask.shape}, "
          f"overlap%={overlap_pct}")