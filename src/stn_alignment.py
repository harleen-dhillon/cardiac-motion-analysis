
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Settings
PROJECT_DIR = Path(__file__).resolve().parents[1]
PATIENT_DIR = Path.home() / "Downloads/database/training/patient015"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SLICE_INDEX = 2
ITERATIONS = 100
LEARNING_RATE = 0.001
IDENTITY_WEIGHT = 0.01
device = torch.device("cpu")


# Load MRI images and masks
def load_nifti(filename):
    path = PATIENT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(str(path))


ed_nii = load_nifti("patient015_frame01.nii.gz")
es_nii = load_nifti("patient015_frame10.nii.gz")
ed_gt = load_nifti("patient015_frame01_gt.nii.gz").get_fdata()
es_gt = load_nifti("patient015_frame10_gt.nii.gz").get_fdata()

ed_volume = ed_nii.get_fdata()
es_volume = es_nii.get_fdata()

if ed_volume.shape != es_volume.shape or ed_gt.shape != ed_volume.shape or es_gt.shape != es_volume.shape:
    raise ValueError("ED, ES and mask dimensions must match.")

if not 0 <= SLICE_INDEX < ed_volume.shape[2]:
    raise ValueError("Invalid slice index.")

ed = ed_volume[:, :, SLICE_INDEX].T
es = es_volume[:, :, SLICE_INDEX].T
ed_mask = ed_gt[:, :, SLICE_INDEX].T.astype(np.uint8)
es_mask = es_gt[:, :, SLICE_INDEX].T.astype(np.uint8)


# Normalize image intensities
def normalize(image):
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError("Invalid image intensity range.")
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


ed_norm, es_norm = normalize(ed), normalize(es)
fixed = torch.from_numpy(ed_norm)[None, None]
moving = torch.from_numpy(es_norm)[None, None]


# Affine STN transformation
theta = torch.nn.Parameter(torch.tensor(
    [[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float32
))
identity = theta.detach().clone()
optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)


def warp(image, transform):
    grid = F.affine_grid(transform, image.size(), align_corners=False)
    result = F.grid_sample(
        image, grid, mode="bilinear",
        padding_mode="zeros", align_corners=False
    )
    return result, grid


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).mean() / torch.sqrt(
        (a ** 2).mean() * (b ** 2).mean() + 1e-8
    )


# Train the STN
print("Training affine STN...")

for i in range(ITERATIONS):
    optimizer.zero_grad()
    aligned, _ = warp(moving, theta)

    similarity_loss = 1 - ncc(fixed, aligned)
    identity_loss = ((theta - identity) ** 2).mean()
    loss = similarity_loss + IDENTITY_WEIGHT * identity_loss

    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(f"Iteration {i + 1}: loss = {loss.item():.6f}")


# Align ES image and mask
with torch.no_grad():
    aligned, grid = warp(moving, theta)
    mask_tensor = torch.from_numpy(es_mask.astype(np.float32))[None, None]
    aligned_mask = F.grid_sample(
        mask_tensor, grid, mode="nearest",
        padding_mode="zeros", align_corners=False
    )[0, 0].numpy().round().astype(np.uint8)

aligned_es = aligned[0, 0].numpy()


# Evaluation metrics
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total else 1.0


def mse(a, b):
    return np.mean((a - b) ** 2)


print("\nAlignment results")
print(f"MSE before: {mse(ed_norm, es_norm):.6f}")
print(f"MSE after:  {mse(ed_norm, aligned_es):.6f}")
print(f"NCC before: {ncc(torch.tensor(ed_norm), torch.tensor(es_norm)).item():.6f}")
print(f"NCC after:  {ncc(torch.tensor(ed_norm), torch.tensor(aligned_es)).item():.6f}")

ed_myo = ed_mask == 2
es_myo = es_mask == 2
aligned_myo = aligned_mask == 2

print(f"Dice before: {dice(ed_myo, es_myo):.4f}")
print(f"Dice after:  {dice(ed_myo, aligned_myo):.4f}")


# Display alignment results
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

ax[0, 0].imshow(ed_norm, cmap="gray")
ax[0, 0].set_title("ED Reference")

ax[0, 1].imshow(es_norm, cmap="gray")
ax[0, 1].set_title("ES Before Alignment")

ax[0, 2].imshow(aligned_es, cmap="gray")
ax[0, 2].set_title("ES After Alignment")

ax[1, 0].imshow(ed_norm, cmap="gray")
ax[1, 0].contour(ed_myo, levels=[0.5], colors="lime")
ax[1, 0].set_title("ED Myocardium")

ax[1, 1].imshow(es_norm, cmap="gray")
ax[1, 1].contour(es_myo, levels=[0.5], colors="cyan")
ax[1, 1].set_title("ES Myocardium")

ax[1, 2].imshow(aligned_es, cmap="gray")
ax[1, 2].contour(ed_myo, levels=[0.5], colors="lime")
ax[1, 2].contour(aligned_myo, levels=[0.5], colors="cyan")
ax[1, 2].set_title("ED vs Aligned ES")

for a in ax.flat:
    a.axis("off")

fig.suptitle(f"Patient015 - Affine STN (Slice {SLICE_INDEX})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "stn_ed_es_alignment.png", dpi=150,
            bbox_inches="tight")
plt.close()


# Save aligned slice in the original ES volume
aligned_volume = es_volume.copy()
aligned_volume[:, :, SLICE_INDEX] = aligned_es.T

output = nib.Nifti1Image(
    aligned_volume.astype(np.float32), es_nii.affine, es_nii.header
)
nib.save(output, str(RESULTS_DIR / "patient015_ES_aligned_slice2.nii.gz"))

print("\nAlignment figure and NIfTI saved in results/")
