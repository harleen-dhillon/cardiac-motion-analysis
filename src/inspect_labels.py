
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

patient_dir = Path.home() / "Downloads/database/training/patient015"
mask_path = patient_dir / "patient015_frame01_gt.nii.gz"
image_path = patient_dir / "patient015_frame01.nii.gz"

mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
image = nib.load(str(image_path)).get_fdata()

# Match the slice selected in our MedSAM experiment.
z = 2

img = image[:, :, z].T
labels = mask[:, :, z].T

colors = ["black", "cyan", "yellow", "red"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("MRI")

for ax, label, title in zip(
    axes[1:],
    [1, 2, 3],
    ["RV cavity", "Myocardium", "LV cavity"]
):
    ax.imshow(img, cmap="gray")
    ax.contour(
        (labels == label).astype(float),
        levels=[0.5],
        colors=["lime"],
        linewidths=1.5
    )
    ax.set_title(f"{title} (label {label})")
    print(f"{title}: {int((labels == label).sum())} pixels")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "results" / "cardiac_labels.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", output_path)
