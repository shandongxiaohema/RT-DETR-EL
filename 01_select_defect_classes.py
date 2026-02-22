"""
Dataset Preprocessing: Select Defect Classes
Filters and copies images and annotations containing target defect classes.
"""

import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# ============================================================================
# Configuration
# ============================================================================

# Source directories
SRC_IMAGES_DIR = Path(r"E:\edgedownload\mdoels\实验\EL2021\trainval\JPEGImages")
SRC_ANN_DIR = Path(r"E:\edgedownload\mdoels\实验\EL2021\trainval\Annotations")

# Destination root directory
DST_ROOT = Path(r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_demo")

# Target defect classes (case-insensitive)
TARGET_CLASSES = {
    "crack",
    "black_core",
    "finger",
    "thick_line",
    "horizontal_dislocation",
    "short_circuit",
}

# ============================================================================
# Behavior Control
# ============================================================================

# Dry run mode: True = print only, False = execute copy
DRY_RUN = False

# Preserve file metadata using copy2
COPY_METADATA = True

# Supported image extensions
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ============================================================================
# Main Processing
# ============================================================================

# Create output directories
DST_IMAGES_DIR = DST_ROOT / "JPEGImages"
DST_ANN_DIR = DST_ROOT / "Annotations"
DST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DST_ANN_DIR.mkdir(parents=True, exist_ok=True)

# Validate source directories
if not SRC_ANN_DIR.exists():
    raise SystemExit(f"Error: Annotation directory not found: {SRC_ANN_DIR}")
if not SRC_IMAGES_DIR.exists():
    raise SystemExit(f"Error: Image directory not found: {SRC_IMAGES_DIR}")

# Process annotations
matched_pairs = []
missing_images = []
parse_errors = []

for xml_path in sorted(SRC_ANN_DIR.glob("*.xml")):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        parse_errors.append((xml_path.name, str(e)))
        continue

    # Extract object class names (lowercase)
    obj_names = {
        obj.findtext("name", default="").strip().lower()
        for obj in root.findall(".//object")
    }

    if not obj_names:
        continue

    # Check if any target class is present
    if obj_names & TARGET_CLASSES:
        # Find corresponding image
        filename_tag = (root.findtext("filename") or "").strip()
        candidates = []

        if filename_tag:
            candidates.append(SRC_IMAGES_DIR / filename_tag)

        stem = xml_path.stem
        for ext in IMG_EXTENSIONS:
            candidates.append(SRC_IMAGES_DIR / (stem + ext))

        # Find existing image
        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break

        if found:
            matched_pairs.append((found, xml_path))
        else:
            missing_images.append(xml_path)

# ============================================================================
# Report and Execute
# ============================================================================

print(f"Matched annotation files: {len(matched_pairs)}")
print(f"Missing image files: {len(missing_images)}")

if parse_errors:
    print(f"XML parsing errors: {len(parse_errors)}")
    for path, error in parse_errors[:5]:
        print(f"  {path}: {error}")

if matched_pairs:
    print(f"\nSample files to copy (showing up to 100):")
    for img, xml in matched_pairs[:100]:
        print(f"  Image: {img.name} | Annotation: {xml.name}")

if missing_images:
    print(f"\nMissing image files (showing up to 20):")
    for xml in missing_images[:20]:
        print(f"  {xml.name}")

if DRY_RUN:
    print("\nDry run mode enabled (DRY_RUN=True). No files copied.")
    print("Set DRY_RUN=False to execute the copy operation.")
else:
    copied_count = 0
    for img, xml in matched_pairs:
        try:
            dst_img = DST_IMAGES_DIR / img.name
            dst_xml = DST_ANN_DIR / xml.name

            if COPY_METADATA:
                shutil.copy2(img, dst_img)
                shutil.copy2(xml, dst_xml)
            else:
                shutil.copy(img, dst_img)
                shutil.copy(xml, dst_xml)

            copied_count += 1
        except Exception as e:
            print(f"Copy failed: {img.name}, {xml.name} - {e}")

    print(f"\nSuccessfully copied {copied_count} image-annotation pairs to {DST_ROOT}")

