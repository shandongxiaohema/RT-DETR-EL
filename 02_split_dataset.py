"""
Dataset Preprocessing: Train/Val/Test Split
Splits dataset into training, validation, and test subsets.
"""

import random
import shutil
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Source directories
SRC_ROOT = Path(r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_demo")
SRC_IMAGES_DIR = SRC_ROOT / "JPEGImages"
SRC_ANN_DIR = SRC_ROOT / "Annotations"

# Destination root directory
DST_ROOT = Path(r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_split")

# ============================================================================
# Behavior Control
# ============================================================================

# Dry run mode: True = print only, False = execute copy
DRY_RUN = False

# Random seed for reproducible split
RANDOM_SEED = 42

# Supported image extensions (lowercase)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ============================================================================
# Dataset Split Ratios
# ============================================================================

TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.1   # 10% for testing

# ============================================================================
# Validation
# ============================================================================

if not SRC_IMAGES_DIR.exists():
    raise SystemExit(f"Error: Image directory not found: {SRC_IMAGES_DIR}")
if not SRC_ANN_DIR.exists():
    raise SystemExit(f"Error: Annotation directory not found: {SRC_ANN_DIR}")

# ============================================================================
# Collect Image-Annotation Pairs
# ============================================================================

pairs = []
for img_path in sorted(SRC_IMAGES_DIR.iterdir()):
    if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTENSIONS:
        continue

    xml_path = SRC_ANN_DIR / (img_path.stem + ".xml")
    if xml_path.exists() and xml_path.is_file():
        pairs.append((img_path, xml_path))

total_samples = len(pairs)
if total_samples == 0:
    raise SystemExit("No matching image-annotation pairs found.")

# ============================================================================
# Split Dataset
# ============================================================================

random.seed(RANDOM_SEED)
random.shuffle(pairs)

n_train = int(total_samples * TRAIN_RATIO)
n_val = int(total_samples * VAL_RATIO)
n_test = total_samples - n_train - n_val

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]

# ============================================================================
# Create Output Directories
# ============================================================================

for subset in ("train", "val", "test"):
    (DST_ROOT / subset / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / subset / "Annotations").mkdir(parents=True, exist_ok=True)

# ============================================================================
# Copy Function
# ============================================================================

def copy_subset(pair_list, subset_name):
    """Copy image-annotation pairs to destination directory."""
    items = []
    for img, xml in pair_list:
        dst_img = DST_ROOT / subset_name / "JPEGImages" / img.name
        dst_xml = DST_ROOT / subset_name / "Annotations" / xml.name
        items.append((img, dst_img, xml, dst_xml))

    if DRY_RUN:
        print(f"\n[{subset_name}] Will copy {len(items)} pairs (DRY_RUN=True, no files copied):")
        for i, (img, dst_img, xml, dst_xml) in enumerate(items[:100], 1):
            print(f"  {i}. {img.name} -> {dst_img}")
            print(f"     {xml.name} -> {dst_xml}")
        if len(items) > 100:
            print(f"  ... and {len(items) - 100} more items ...")
        return 0

    copied_count = 0
    for img, dst_img, xml, dst_xml in items:
        try:
            shutil.copy2(img, dst_img)
            shutil.copy2(xml, dst_xml)
            copied_count += 1
        except Exception as e:
            print(f"Copy failed: {img}, {xml} - {e}")

    return copied_count

# ============================================================================
# Report and Execute
# ============================================================================

print(f"Total valid samples: {total_samples}")
print(f"Split ratio: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
print(f"Split result: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

copied_train = copy_subset(train_pairs, "train")
copied_val = copy_subset(val_pairs, "val")
copied_test = copy_subset(test_pairs, "test")

if not DRY_RUN:
    print(f"\nCopy completed: train={copied_train}, val={copied_val}, test={copied_test}")
else:
    print("\nDry run mode enabled (DRY_RUN=True). No files copied.")
    print("Verify the output above, then set DRY_RUN=False and run again to execute.")

