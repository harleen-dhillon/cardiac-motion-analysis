
from pathlib import Path
import re
import csv

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- SETTINGS --------------------

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Downloads/database/training"
RESULTS = PROJECT / "results"

PATIENTS = [
    "015", "020", "021", "022", "023", "024",
    "025", "026", "027", "028", "029"
]

SLICE = 2
ITERATIONS = 200
MAX_FLOW = 0.05
LR = 0.001

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")


# -------------------- DEFORMABLE STN --------------------

class DeformableSTN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1)
        )

        # Start with zero displacement
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed, moving):
        _, _, height, width = moving.shape

        # Use both ED and ES images to predict the flow
        inputs = torch.cat((fixed, moving), dim=1)

        flow = torch.tanh(self.net(inputs)) * MAX_FLOW

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=moving.device),
            torch.linspace(-1, 1, width, device=moving.device),
            indexing="ij"
        )

        grid = torch.stack((x, y), dim=-1).unsqueeze(0)

        # Add predicted deformation to the identity grid
        grid = grid + flow.permute(0, 2, 3, 1)

        warped = F.grid_sample(
            moving,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        return warped, flow


# -------------------- METRICS --------------------

def normalize(image):
    low, high = np.percentile(image, [1, 99])

    if high <= low:
        raise ValueError("Invalid image intensity range")

    image = np.clip(image, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def calculate_mse(a, b):
    return float(np.mean((a - b) ** 2))


def calculate_ncc(a, b):
    a = a.astype(np.float64) - np.mean(a)
    b = b.astype(np.float64) - np.mean(b)

    denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if denominator < 1e-12:
        return float("nan")

    return float(np.sum(a * b) / denominator)


def calculate_dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    total = a.sum() + b.sum()

    if total == 0:
        return float("nan")

    return float(2 * np.logical_and(a, b).sum() / total)


def smoothness_loss(flow):
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    return dx.square().mean() + dy.square().mean()


def get_frame(info_path, key):
    text = info_path.read_text()

    match = re.search(
        rf"^{key}:\s*(\d+)",
        text,
        re.MULTILINE
    )

    if not match:
        raise ValueError(f"{key} frame not found in Info.cfg")

    return int(match.group(1))


# -------------------- EVALUATE ONE PATIENT --------------------

def evaluate_patient(patient):
    folder = DATA / f"patient{patient}"

    if not folder.exists():
        raise FileNotFoundError(f"Patient folder missing: {folder}")

    ed_frame = get_frame(folder / "Info.cfg", "ED")
    es_frame = get_frame(folder / "Info.cfg", "ES")

    def load_file(filename):
        path = folder / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.name}")

        return nib.load(str(path)).get_fdata()

    ed = load_file(
        f"patient{patient}_frame{ed_frame:02d}.nii.gz"
    )
    es = load_file(
        f"patient{patient}_frame{es_frame:02d}.nii.gz"
    )
    ed_gt = load_file(
        f"patient{patient}_frame{ed_frame:02d}_gt.nii.gz"
    )
    es_gt = load_file(
        f"patient{patient}_frame{es_frame:02d}_gt.nii.gz"
    )

    if not (ed.shape == es.shape == ed_gt.shape == es_gt.shape):
        raise ValueError("ED/ES image and label shapes do not match")

    if ed.ndim != 3 or SLICE >= ed.shape[2]:
        raise ValueError(f"Invalid slice index for shape {ed.shape}")

    # Extract the same slice from ED and ES
    fixed = normalize(ed[:, :, SLICE].T)
    moving = normalize(es[:, :, SLICE].T)

    fixed_gt = ed_gt[:, :, SLICE].T
    moving_gt = es_gt[:, :, SLICE].T

    fixed_tensor = torch.from_numpy(fixed)[None, None].to(DEVICE)
    moving_tensor = torch.from_numpy(moving)[None, None].to(DEVICE)

    # Before registration
    mse_before = calculate_mse(fixed, moving)
    ncc_before = calculate_ncc(fixed, moving)
    dice_before = calculate_dice(fixed_gt == 2, moving_gt == 2)

    # Create a fresh model for this patient pair
    model = DeformableSTN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Optimize the deformation
    for _ in range(ITERATIONS):
        warped, flow = model(fixed_tensor, moving_tensor)

        similarity_loss = 1 - F.cosine_similarity(
            fixed_tensor.flatten(),
            warped.flatten(),
            dim=0
        )

        loss = (
            similarity_loss
            + 0.1 * smoothness_loss(flow)
            + 0.1 * flow.square().mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get final warped image and deformation
    model.eval()

    with torch.no_grad():
        warped, flow = model(fixed_tensor, moving_tensor)

        warped_image = warped.squeeze().cpu().numpy()
        flow = flow.cpu()

        height, width = moving.shape

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=DEVICE),
            torch.linspace(-1, 1, width, device=DEVICE),
            indexing="ij"
        )

        grid = torch.stack((x, y), dim=-1)[None]
        grid = grid + flow.permute(0, 2, 3, 1)

        # Warp ES ground-truth labels using nearest-neighbour sampling
        moving_mask = torch.from_numpy(
            moving_gt.astype(np.float32)
        )[None, None].to(DEVICE)

        aligned_mask = F.grid_sample(
            moving_mask,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False
        ).squeeze().cpu().numpy()

    # After registration
    mse_after = calculate_mse(fixed, warped_image)
    ncc_after = calculate_ncc(fixed, warped_image)
    dice_after = calculate_dice(fixed_gt == 2, aligned_mask == 2)

    return {
        "patient": f"patient{patient}",
        "ED_frame": ed_frame,
        "ES_frame": es_frame,
        "slice": SLICE,
        "MSE_before": mse_before,
        "MSE_after": mse_after,
        "NCC_before": ncc_before,
        "NCC_after": ncc_after,
        "Dice_before": dice_before,
        "Dice_after": dice_after
    }


# -------------------- RUN ALL PATIENTS --------------------

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 60)
    print("MULTI-PATIENT DEFORMABLE STN EVALUATION")
    print(f"Patients: {len(PATIENTS)}")
    print(f"Slice index: {SLICE}")
    print(f"Iterations: {ITERATIONS}")
    print("=" * 60)

    for patient in PATIENTS:
        print(f"\nProcessing patient{patient}...")

        try:
            result = evaluate_patient(patient)
            rows.append(result)

            print(
                f"MSE:  {result['MSE_before']:.4f} "
                f"-> {result['MSE_after']:.4f}"
            )
            print(
                f"NCC:  {result['NCC_before']:.4f} "
                f"-> {result['NCC_after']:.4f}"
            )
            print(
                f"Dice: {result['Dice_before']:.4f} "
                f"-> {result['Dice_after']:.4f}"
            )

        except Exception as error:
            print(f"ERROR: {error}")

    if not rows:
        print("\nNo patients were processed successfully.")
        return

    output_file = RESULTS / "multi_patient_deformable_metrics.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"Successful patients: {len(rows)}/{len(PATIENTS)}")
    print(f"Results saved to: {output_file}")

    print("\nAverage metrics:")
    for metric in [
        "MSE_before", "MSE_after",
        "NCC_before", "NCC_after",
        "Dice_before", "Dice_after"
    ]:
        values = [
            row[metric] for row in rows
            if np.isfinite(row[metric])
        ]

        if values:
            print(f"{metric}: {np.mean(values):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
