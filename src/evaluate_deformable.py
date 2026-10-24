
from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

# File locations
PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
PATIENT = Path.home() / "Downloads/database/training/patient015"

# Load saved registration results
data = np.load(RESULTS / "deformable_stn_patient015.npz")

ed_mask = data["ed_mask"]
es_mask = data["es_mask"]
aligned_mask = data["aligned_mask"]
slice_index = int(data["slice_index"])

# Get pixel spacing in mm
nii = nib.load(str(PATIENT / "patient015_frame01_gt.nii.gz"))
spacing_x, spacing_y = nii.header.get_zooms()[:2]


# Estimate wall thickness from a myocardium mask
def measure_thickness(mask, phase):
    wall = mask == 2

    if not wall.any():
        return [phase, 0, "", ""]

    distance = distance_transform_edt(
        wall, sampling=(spacing_y, spacing_x)
    )

    thickness = 2 * distance[wall]

    return [
        phase,
        int(wall.sum()),
        float(thickness.mean()),
        float(np.median(thickness))
    ]


# Compare ED and ES masks
rows = [
    measure_thickness(ed_mask, "ED reference"),
    measure_thickness(es_mask, "ES before alignment"),
    measure_thickness(aligned_mask, "ES after alignment")
]

print(f"\nPatient015, slice {slice_index}: thickness estimates")
print(f"{'Phase':<24} {'Pixels':>8} {'Mean (mm)':>12} {'Median (mm)':>13}")

for row in rows:
    print(f"{row[0]:<24} {row[1]:>8} {str(row[2]):>12} {str(row[3]):>13}")

# Save results
output = RESULTS / "deformable_thickness_comparison.csv"

with open(output, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["phase", "pixels", "mean_proxy_mm", "median_proxy_mm"])
    writer.writerows(rows)

print("\nSaved:", output)
