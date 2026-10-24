
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Dataset location
DATA_ROOT = Path.home() / "Downloads" / "database"
TRAIN_DIR = DATA_ROOT / "training"

# Start with one patient
PATIENT_DIR = TRAIN_DIR / "patient015"

image_path = PATIENT_DIR / "patient015_frame01.nii.gz"
mask_path = PATIENT_DIR / "patient015_frame01_gt.nii.gz"

# Check that both files exist
if not image_path.exists():
    raise FileNotFoundError(f"MRI image not found: {image_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

# Load MRI and reference segmentation
image_nii = nib.load(str(image_path))
mask_nii = nib.load(str(mask_path))

image = image_nii.get_fdata()
mask = mask_nii.get_fdata()

print("MRI shape:", image.shape)
print("Mask shape:", mask.shape)
print("Voxel spacing (mm):", image_nii.header.get_zooms())
print("Segmentation labels:", np.unique(mask))
print("Image intensity range:", image.min(), image.max())

# Display the central slice
slice_index = image.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
axes[0].set_title("Cardiac MRI")
axes[0].axis("off")

axes[1].imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
axes[1].imshow(
    mask[:, :, slice_index].T,
    cmap="jet",
    alpha=0.45,
    origin="lower",
    vmin=0,
    vmax=max(3, int(mask.max())),
)
axes[1].set_title("MRI + Ground Truth")
axes[1].axis("off")

plt.tight_layout()
plt.show()
