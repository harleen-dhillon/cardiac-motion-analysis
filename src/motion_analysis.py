
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Load registration results
data = np.load("results/deformable_stn_patient015.npz")

flow = data["displacement_flow"]
ed_mask = data["ed_mask"].astype(bool)
es_mask = data["es_mask"].astype(bool)
aligned_mask = data["aligned_mask"]
slice_index = int(data["slice_index"])

height, width = ed_mask.shape

# Convert normalized displacement to pixels, then mm
flow_x = flow[0] * width / 2 * 1.36719
flow_y = flow[1] * height / 2 * 1.36719

# Calculate displacement magnitude
magnitude = np.sqrt(flow_x**2 + flow_y**2)

# Measure displacement inside the ED myocardium
motion = magnitude[ed_mask]

print("\nCARDIAC MOTION ANALYSIS")
print("Patient: 015")
print(f"Slice: {slice_index}")
print(f"Mean displacement: {motion.mean():.4f} mm")
print(f"Median displacement: {np.median(motion):.4f} mm")
print(f"Maximum displacement: {motion.max():.4f} mm")

# Save measurements
os.makedirs("results", exist_ok=True)

with open("results/cardiac_motion_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value", "Unit"])
    writer.writerow(["Mean displacement", motion.mean(), "mm"])
    writer.writerow(["Median displacement", np.median(motion), "mm"])
    writer.writerow(["Maximum displacement", motion.max(), "mm"])

# Create four-panel figure
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# 1. Displacement heatmap
im = ax[0, 0].imshow(magnitude, cmap="turbo")
ax[0, 0].contour(ed_mask, levels=[0.5], colors="white")
ax[0, 0].set_title("Displacement Magnitude (mm)")
fig.colorbar(im, ax=ax[0, 0])

# 2. Displacement arrows
ax[0, 1].imshow(np.zeros_like(ed_mask), cmap="gray")
ax[0, 1].contour(ed_mask, levels=[0.5], colors="white")

step = 8
y, x = np.mgrid[0:height:step, 0:width:step]
region = ed_mask[::step, ::step]

ax[0, 1].quiver(
    x[region], y[region],
    flow_x[::step, ::step][region],
    -flow_y[::step, ::step][region],
    color="red", scale_units="xy", angles="xy", scale=1
)
ax[0, 1].set_title("Estimated Motion Vectors")

# 3. ED vs original ES
ax[1, 0].contour(ed_mask, levels=[0.5], colors="lime")
ax[1, 0].contour(es_mask, levels=[0.5], colors="red")
ax[1, 0].set_title("ED vs ES Before Alignment")

# 4. ED vs aligned ES
ax[1, 1].contour(ed_mask, levels=[0.5], colors="lime")
ax[1, 1].contour(aligned_mask, levels=[0.5], colors="cyan")
ax[1, 1].set_title("ED vs ES After Alignment")

# Clean up all panels
for a in ax.flat:
    a.set_aspect("equal")
    a.axis("off")

fig.suptitle("Cardiac Motion Analysis - Patient015")
plt.tight_layout()
plt.savefig("results/cardiac_motion_analysis.png", dpi=200,
            bbox_inches="tight")
plt.close()

print("Motion analysis figure and CSV saved in results/")
