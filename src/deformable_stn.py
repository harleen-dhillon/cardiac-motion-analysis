
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Settings
PROJECT = Path(__file__).resolve().parents[1]
PATIENT = Path.home() / "Downloads/database/training/patient015"
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SLICE = 2
ITERATIONS = 200
LR = 0.001
SMOOTH_WEIGHT = 0.1
FLOW_WEIGHT = 0.1
MAX_FLOW = 0.05

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")


# Load ED, ES and their masks
def load_nifti(name):
    path = PATIENT / name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(str(path))


ed_nii = load_nifti("patient015_frame01.nii.gz")
es_nii = load_nifti("patient015_frame10.nii.gz")
ed_gt = load_nifti("patient015_frame01_gt.nii.gz").get_fdata()
es_gt = load_nifti("patient015_frame10_gt.nii.gz").get_fdata()

ed_volume = ed_nii.get_fdata()
es_volume = es_nii.get_fdata()

if ed_volume.shape != es_volume.shape:
    raise ValueError("ED and ES image shapes do not match.")
if ed_gt.shape != ed_volume.shape or es_gt.shape != es_volume.shape:
    raise ValueError("Image and mask shapes do not match.")
if not 0 <= SLICE < ed_volume.shape[2]:
    raise ValueError("Invalid slice index.")

ed = ed_volume[:, :, SLICE].T
es = es_volume[:, :, SLICE].T
ed_mask = ed_gt[:, :, SLICE].T.astype(np.uint8)
es_mask = es_gt[:, :, SLICE].T.astype(np.uint8)


# Normalize images
def normalize(image):
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError("Invalid image intensity range.")
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


fixed_np, moving_np = normalize(ed), normalize(es)
fixed = torch.from_numpy(fixed_np)[None, None]
moving = torch.from_numpy(moving_np)[None, None]


# Deformable STN predicts a displacement field
class DeformableSTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed_image, moving_image):
        flow = torch.tanh(
            self.net(torch.cat([fixed_image, moving_image], dim=1))
        ) * MAX_FLOW

        identity = torch.tensor(
            [[[1., 0., 0.], [0., 1., 0.]]],
            dtype=fixed_image.dtype,
            device=fixed_image.device
        )
        grid = F.affine_grid(identity, fixed_image.size(),
                             align_corners=False)
        grid = grid + flow.permute(0, 2, 3, 1)

        warped = F.grid_sample(
            moving_image, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False
        )
        return warped, flow, grid


# Loss functions and evaluation metrics
def ncc_loss(a, b):
    a, b = a - a.mean(), b - b.mean()
    return 1 - (a * b).mean() / torch.sqrt(
        (a.square().mean() * b.square().mean()) + 1e-8
    )


def smoothness_loss(flow):
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.square().mean() + dy.square().mean()


def dice(a, b):
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total else 1.0


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.mean(a**2) * np.mean(b**2))
    return float(np.mean(a * b) / denominator) if denominator > 1e-8 else 0.0


ed_myo, es_myo = ed_mask == 2, es_mask == 2

print("Before registration:")
print(f"MSE: {mse(fixed_np, moving_np):.6f}")
print(f"NCC: {ncc(fixed_np, moving_np):.6f}")
print(f"Myocardium Dice: {dice(ed_myo, es_myo):.4f}")


# Train the model
model = DeformableSTN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nTraining deformable STN...")

for i in range(ITERATIONS):
    optimizer.zero_grad()
    warped, flow, grid = model(fixed, moving)

    loss = (
        ncc_loss(fixed, warped)
        + SMOOTH_WEIGHT * smoothness_loss(flow)
        + FLOW_WEIGHT * flow.square().mean()
    )

    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(f"Iteration {i + 1}: loss = {loss.item():.6f}")


# Align ES image and mask
model.eval()
with torch.no_grad():
    aligned, flow, grid = model(fixed, moving)

    mask_tensor = torch.from_numpy(es_mask.astype(np.float32))[None, None]
    aligned_mask = F.grid_sample(
        mask_tensor, grid, mode="nearest",
        padding_mode="zeros", align_corners=False
    )[0, 0].numpy().round().astype(np.uint8)

aligned_np = aligned[0, 0].numpy()
flow_np = flow[0].numpy()
aligned_myo = aligned_mask == 2

print("\nAfter registration:")
print(f"MSE: {mse(fixed_np, aligned_np):.6f}")
print(f"NCC: {ncc(fixed_np, aligned_np):.6f}")
print(f"Myocardium Dice: {dice(ed_myo, aligned_myo):.4f}")
print(f"Mean absolute normalized flow: {np.abs(flow_np).mean():.6f}")
print(f"Maximum absolute normalized flow: {np.abs(flow_np).max():.6f}")


# Display results
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

ax[0, 0].imshow(fixed_np, cmap="gray")
ax[0, 0].set_title("ED Reference")

ax[0, 1].imshow(moving_np, cmap="gray")
ax[0, 1].set_title("ES Before Alignment")

ax[0, 2].imshow(aligned_np, cmap="gray")
ax[0, 2].set_title("ES After Deformable STN")

ax[1, 0].imshow(fixed_np, cmap="gray")
ax[1, 0].contour(ed_myo, levels=[0.5], colors="lime")
ax[1, 0].set_title("ED Myocardium")

ax[1, 1].imshow(moving_np, cmap="gray")
ax[1, 1].contour(es_myo, levels=[0.5], colors="cyan")
ax[1, 1].set_title("ES Before Alignment")

ax[1, 2].imshow(aligned_np, cmap="gray")
ax[1, 2].contour(ed_myo, levels=[0.5], colors="lime")
ax[1, 2].contour(aligned_myo, levels=[0.5], colors="cyan")
ax[1, 2].set_title(f"ED vs Aligned ES (Dice={dice(ed_myo, aligned_myo):.3f})")

for a in ax.flat:
    a.axis("off")

fig.suptitle(f"Patient015 - Deformable STN - Slice {SLICE}")
plt.tight_layout()
figure_path = RESULTS / "deformable_stn_alignment.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close()


# Save flow and outputs for motion analysis
np.savez_compressed(
    RESULTS / "deformable_stn_patient015.npz",
    displacement_flow=flow_np,
    aligned_es=aligned_np,
    aligned_mask=aligned_mask,
    ed_mask=ed_mask,
    es_mask=es_mask,
    slice_index=SLICE
)

print("\nSaved figure:", figure_path)
print("Saved flow and outputs in results/")
print("Deformable STN experiment completed.")
