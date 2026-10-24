
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT / "results" / "multi_patient_deformable_metrics.csv"
OUTPUT = PROJECT / "results" / "dice_comparison.png"

# Load results
df = pd.read_csv(CSV_FILE)

# Plot Dice scores
x = range(len(df))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    [i - width / 2 for i in x],
    df["Dice_before"],
    width,
    label="Before Registration"
)

plt.bar(
    [i + width / 2 for i in x],
    df["Dice_after"],
    width,
    label="After Registration"
)

plt.xticks(x, df["patient"], rotation=45)
plt.ylabel("Dice Score")
plt.xlabel("Patient")
plt.title("Myocardium Dice Score Before vs After Deformable STN")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT, dpi=300)
plt.show()

print(f"Graph saved to: {OUTPUT}")
