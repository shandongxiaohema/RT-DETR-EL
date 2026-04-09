# RTDETR - Real-Time Deformable Transformer for Object Detection

A PyTorch implementation of RT-DETR (Real-Time Deformable Transformer) for defect detection in industrial applications.

## Project Structure

```
.
├── train.py                          # Training script
├── val.py                            # Validation and evaluation script
├── 01_select_defect_classes.py       # Step 1: Select target defect classes from raw dataset
├── 02_split_dataset.py               # Step 2: Split dataset into train/val/test
├── 03_augment_and_balance.py         # Step 3: Data augmentation and class balancing
├── dataset_config.yaml               # Configuration file for dataset processing
├── requirements.txt                  # Python dependencies
├── ultralytics/                      # Core framework
│   ├── models/                       # Model architectures
│   ├── engine/                       # Training and validation engines
│   ├── data/                         # Data loading and processing
│   └── ...
└── runs/                             # Training outputs (logs, weights, etc.)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RTDETR-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for data processing:
```bash
pip install albumentations  # For data augmentation
pip install prettytable     # For result visualization
```

## Public Datasets

### PVEL-AD Dataset

The PVEL-AD (Photovoltaic Equipment Visual Inspection - Anomaly Detection) dataset is a publicly available dataset for defect detection in industrial applications.

**Download:** https://github.com/binyisu/PVEL-AD

This dataset can be used as a reference or starting point for training the RT-DETR model on defect detection tasks.

---

## Dataset Preparation

The dataset preparation pipeline consists of three steps:

### Step 1: Select Defect Classes

Extract images and annotations containing target defect classes from your raw dataset.

**Input:** Raw dataset with images and XML annotations
**Output:** Filtered dataset containing only target defect classes

```bash
python 01_select_defect_classes.py
```

**Configuration in `dataset_config.yaml`:**
```yaml
select_defect_classes:
  src_images_dir: "./data/raw/JPEGImages"
  src_ann_dir: "./data/raw/Annotations"
  dst_root: "./data/selected"
  target_classes:
    - crack
    - black_core
    - finger
    - thick_line
    - horizontal_dislocation
    - short_circuit
  dry_run: false  # Set to true for preview
```

**What it does:**
- Parses XML annotation files
- Identifies images containing target defect classes
- Copies matching image-annotation pairs to output directory
- Reports statistics (matched files, missing images, parse errors)

---

### Step 2: Split Dataset

Divide the selected dataset into training, validation, and test sets with a 7:2:1 ratio.

**Input:** Selected dataset from Step 1
**Output:** Train/Val/Test splits with 70%/20%/10% distribution

```bash
python 02_split_dataset.py
```

**Configuration in `dataset_config.yaml`:**
```yaml
split_dataset:
  src_root: "./data/selected"
  dst_root: "./data/split"
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  seed: 42  # For reproducibility
  dry_run: false
```

**What it does:**
- Collects all image-annotation pairs
- Shuffles with fixed seed for reproducibility
- Splits into train/val/test sets
- Creates directory structure for each subset
- Copies files to respective directories

**Output structure:**
```
data/split/
├── train/
│   ├── JPEGImages/
│   └── Annotations/
├── val/
│   ├── JPEGImages/
│   └── Annotations/
└── test/
    ├── JPEGImages/
    └── Annotations/
```

---

### Step 3: Augment and Balance Dataset

Apply data augmentation and class balancing to the training set to handle class imbalance and improve model robustness.

**Input:** Training set from Step 2
**Output:** Augmented and balanced training set

```bash
python 03_augment_and_balance.py
```

**Configuration in `dataset_config.yaml`:**
```yaml
augment_and_balance:
  img_dir: "./data/split/train/JPEGImages"
  xml_dir: "./data/split/train/Annotations"
  save_root: "./data/augmented/train"

  # Class balancing parameters
  base_target: 1500         # Target count per class
  target_jitter_frac: 0.10  # ±10% variation around target
  downsample_major: true    # Downsample majority classes

  # Augmentation strategies
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    brightness_contrast: 0.6
    gaussian_noise: 0.5
```

**What it does:**

1. **Reads original data:** Parses XML annotations and loads images
2. **Analyzes class distribution:** Counts instances per defect class
3. **Downsamples majority classes:** Removes excess instances of over-represented classes
4. **Copies original data:** Saves downsampled original images
5. **Augments minority classes:** Applies transformations to under-represented classes:
   - Horizontal/Vertical flips
   - Brightness and contrast adjustments
   - Gaussian noise injection
6. **Balances dataset:** Generates augmented samples until each class reaches target count
7. **Validates bounding boxes:** Ensures boxes meet minimum size requirements

**Augmentation strategies:**
- **Horizontal Flip:** 50% probability
- **Vertical Flip:** 50% probability
- **Brightness/Contrast:** 60% probability
- **Gaussian Noise:** 50% probability (variance: 10-50)

**Output:**
```
data/augmented/train/
├── JPEGImages/     # Augmented images (numbered sequentially)
└── Annotations/    # Corresponding XML annotations
```

---

## Training

### Basic Training

```bash
python train.py
```

**Configuration in `train.py`:**
```python
model = RTDETR('ultralytics/cfg/models/自定义/rtdetr_soep_biformer.yaml')
model.train(
    data='dataset/data.yaml',
    cache=False,
    imgsz=640,
    epochs=100,
    batch=4,
    workers=4,
    project='runs/train',
    name='exp_rtdetr'
)
```

**Key parameters:**
- `data`: Path to dataset YAML file
- `imgsz`: Input image size (640x640)
- `epochs`: Number of training epochs
- `batch`: Batch size
- `workers`: Number of data loading workers
- `project`: Output directory for training results
- `name`: Experiment name

### Dataset YAML Format

Create `dataset/data.yaml`:
```yaml
path: /path/to/dataset
train: data/split/train
val: data/split/val
test: data/split/test

nc: 6  # Number of classes
names: ['crack', 'black_core', 'finger', 'thick_line', 'horizontal_dislocation', 'short_circuit']
```

---

## Validation and Evaluation

### Run Validation

```bash
python val.py
```

**Configuration in `val.py`:**
```python
model = RTDETR('runs/train/exp-AIFI-EPGO/weights/best.pt')
result = model.val(
    data='/path/to/dataset/data.yaml',
    split='test',
    imgsz=640,
    batch=4,
    project='runs/val',
    name='exp'
)
```

**Output metrics:**
- **Model Info:** GFLOPs, Parameters, Inference time, FPS, Model size
- **Detection Metrics:** Precision, Recall, F1-Score, mAP50, mAP75, mAP50-95
- **Per-class results:** Individual metrics for each defect class
- **Average results:** Mean metrics across all classes

**Results saved to:** `runs/val/exp/paper_data.txt`

---

## Model Architecture

The model uses RT-DETR (Real-Time Deformable Transformer) with custom improvements:

- **Backbone:** HGNetV2 or ResNet variants
- **Neck:** Feature Pyramid Network (FPN)
- **Head:** Deformable Transformer with attention mechanisms
- **Custom modules:** GGA, MDP-Neck for enhanced feature extraction

---

## Defect Classes

The model detects 6 types of defects:

1. **Crack** - Surface cracks and fractures
2. **Black Core** - Dark spots or core defects
3. **Finger** - Finger-like protrusions
4. **Thick Line** - Thick line defects
5. **Horizontal Dislocation** - Horizontal misalignment
6. **Short Circuit** - Electrical short circuit defects

---

## Tips for Best Results

### Data Preparation
- Ensure XML annotations are in Pascal VOC format
- Verify image-annotation pairs match correctly
- Check for corrupted images before processing
- Use `dry_run=true` to preview operations before execution

### Training
- Start with smaller batch sizes if GPU memory is limited
- Use data augmentation to improve generalization
- Monitor validation metrics to detect overfitting
- Save best model weights based on mAP50-95

### Evaluation
- Use test set for final model evaluation
- Report metrics from `paper_data.txt` for publications
- Compare results across different model variants
- Analyze per-class performance to identify weak areas

---

## Troubleshooting

### Common Issues

**Issue:** "Image directory not found"
- **Solution:** Verify paths in configuration file are correct and use forward slashes or raw strings

**Issue:** "XML parsing failed"
- **Solution:** Check XML files are valid and follow Pascal VOC format

**Issue:** "No matching image-annotation pairs"
- **Solution:** Ensure image and XML files have matching names (same stem)

**Issue:** "CUDA out of memory"
- **Solution:** Reduce batch size or image size in training configuration

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rtdetr2023,
  title={RT-DETR: An Efficient Detector for Real-Time Object Detection},
  author={...},
  journal={...},
  year={2023}
}
```

---

## License

This project is provided for research and educational purposes.

---

## Contact

For questions or issues, please open an issue on the repository.
