
from pathlib import Path
import json
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parent / "Task01_BrainTumour" / "Task01_BrainTumour"

with open(ROOT / "dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

print("Dataset:", dataset["name"])
print("Training cases:", dataset["numTraining"])
print("Test cases:", dataset["numTest"])
print("Modalities:", dataset["modality"])
print("Labels:", dataset["labels"])

# Inspect the first training case
case = dataset["training"][0]

image_path = ROOT / case["image"]
label_path = ROOT / case["label"]

image = nib.load(str(image_path))
label = nib.load(str(label_path))

image_data = np.asarray(image.dataobj)
label_data = np.asarray(label.dataobj)

print("\nMRI shape:", image_data.shape)
print("Mask shape:", label_data.shape)
print("Voxel spacing:", image.header.get_zooms())
print("Mask labels:", np.unique(label_data))
print("MRI contains NaN:", np.isnan(image_data).any())
print("MRI contains Inf:", np.isinf(image_data).any())

assert image_data.ndim == 4, "Expected 4D MRI"
assert image_data.shape[-1] == 4, "Expected 4 MRI modalities"
assert image_data.shape[:3] == label_data.shape[:3], "Image-mask shape mismatch"
assert np.isfinite(image_data).all(), "MRI contains NaN or Inf"
assert np.isin(np.unique(label_data), [0, 1, 2, 3]).all(), "Unexpected mask labels"

print("\nPASS: First case passed basic validation!")
