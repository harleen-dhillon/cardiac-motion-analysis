
from pathlib import Path
import nibabel as nib
import numpy as np

patient_dir = Path.home() / "Downloads/database/training/patient015"

volume_path = patient_dir / "patient015_4d.nii.gz"
info_path = patient_dir / "Info.cfg"

# Read patient metadata
print("=== PATIENT INFORMATION ===")
if info_path.exists():
    print(info_path.read_text())
else:
    print("Info.cfg not found")

# Load the full cardiac sequence
print("\n=== 4D MRI VOLUME ===")
if not volume_path.exists():
    raise FileNotFoundError(volume_path)

nii = nib.load(str(volume_path))
volume = nii.get_fdata(dtype=np.float32)

print("Volume shape:", volume.shape)
print("Voxel spacing:", nii.header.get_zooms())
print("Intensity range:", float(volume.min()), float(volume.max()))

if volume.ndim == 4:
    print("Number of cardiac frames:", volume.shape[3])
else:
    print("Warning: expected a 4D cardiac sequence.")

# Inspect annotated ED and ES frames
print("\n=== AVAILABLE ANNOTATIONS ===")
for phase in ("frame01", "frame10", "frame11", "frame13"):
    image_path = patient_dir / f"patient015_{phase}.nii.gz"
    mask_path = patient_dir / f"patient015_{phase}_gt.nii.gz"

    if image_path.exists():
        print(f"{phase}: image available")
    if mask_path.exists():
        mask_nii = nib.load(str(mask_path))
        labels = np.unique(mask_nii.get_fdata())
        print(f"{phase}: mask available; labels = {labels}")
