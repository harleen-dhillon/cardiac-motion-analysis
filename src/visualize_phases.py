
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

patient_dir = Path.home() / "Downloads/database/training/patient015"

# ACDC metadata: ED = frame 1, ES = frame 10
phases = [
    ("End-diastole (ED)", "frame01"),
    ("End-systole (ES)", "frame10"),
]

# ACDC reference labels:
# 0 = background, 1 = right ventricle,
# 2 = myocardium, 3 = left ventricle
colors = ["black", "cyan", "yellow", "red"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

fig, axes = plt.subplots(2, 3, figsize=(13, 9))

for row, (title, frame) in enumerate(phases):
    image_path = patient_dir / f"patient015_{frame}.nii.gz"
    mask_path = patient_dir / f"patient015_{frame}_gt.nii.gz"

    image = nib.load(str(image_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()

    # Use the middle short-axis slice for initial comparison
    z = image.shape[2] // 2
    img_slice = image[:, :, z].T
    mask_slice = mask[:, :, z].T

    axes[row, 0].imshow(img_slice, cmap="gray", origin="lower")
    axes[row, 0].set_title(f"{title}: MRI")

    axes[row, 1].imshow(mask_slice, cmap=cmap, norm=norm, origin="lower")
    axes[row, 1].set_title(f"{title}: Ground truth")

    axes[row, 2].imshow(img_slice, cmap="gray", origin="lower")
    axes[row, 2].imshow(
        mask_slice,
        cmap=cmap,
        norm=norm,
        alpha=0.45,
        origin="lower",
    )
    axes[row, 2].set_title(f"{title}: Overlay")

    for ax in axes[row]:
        ax.axis("off")

fig.suptitle("Cardiac MRI: ED vs ES", fontsize=15)
plt.tight_layout()

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "ed_es_comparison.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved comparison to: {output_path.resolve()}")
plt.show()
