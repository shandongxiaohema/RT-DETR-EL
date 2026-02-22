# Project Information

## Overview

This is a PyTorch implementation of **RT-DETR (Real-Time Deformable Transformer)** for industrial defect detection. The project includes:

- **Model Architecture:** RT-DETR with custom improvements (SOEP, BiFormer, CBAM)
- **Dataset Processing:** Complete pipeline for data selection, splitting, augmentation, and balancing
- **Training & Evaluation:** Full training and validation scripts with detailed metrics
- **Target Application:** Detection of 6 types of defects in industrial products

## Key Features

### 1. Complete Data Processing Pipeline
- **Step 1:** Select target defect classes from raw dataset
- **Step 2:** Split dataset into train/val/test sets (7:2:1)
- **Step 3:** Augment and balance dataset for improved model performance

### 2. Advanced Data Augmentation
- Horizontal/Vertical flips
- Brightness and contrast adjustments
- Gaussian noise injection
- Intelligent class balancing with configurable target counts

### 3. Real-Time Detection
- Efficient transformer-based architecture
- Real-time inference capability
- Optimized for edge deployment

### 4. Comprehensive Evaluation
- Per-class metrics (Precision, Recall, F1-Score)
- Multiple mAP metrics (mAP50, mAP75, mAP50-95)
- Performance metrics (FPS, inference time, model size)
- Detailed results export for paper publication

## Defect Classes

The model detects 6 types of defects:

| Class | Description |
|-------|-------------|
| Crack | Surface cracks and fractures |
| Black Core | Dark spots or core defects |
| Finger | Finger-like protrusions |
| Thick Line | Thick line defects |
| Horizontal Dislocation | Horizontal misalignment |
| Short Circuit | Electrical short circuit defects |

## Project Structure

```
RTDETR-main/
├── README.md                         # Main documentation
├── QUICKSTART.md                     # Quick start guide
├── PROJECT_INFO.md                   # This file
├── dataset_config.yaml               # Configuration for data processing
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── train.py                          # Training script
├── val.py                            # Validation script
├── test_environment.py               # Environment validation script
│
├── 01_select_defect_classes.py       # Data processing step 1
├── 02_split_dataset.py               # Data processing step 2
├── 03_augment_and_balance.py         # Data processing step 3
│
├── ultralytics/                      # Core framework
│   ├── models/                       # Model architectures
│   │   ├── rtdetr/                   # RT-DETR models
│   │   ├── fastsam/                  # FastSAM models
│   │   └── ...
│   ├── engine/                       # Training and validation engines
│   │   ├── trainer.py                # Training engine
│   │   ├── validator.py              # Validation engine
│   │   ├── predictor.py              # Inference engine
│   │   └── ...
│   ├── data/                         # Data loading and processing
│   │   ├── dataset.py                # Dataset classes
│   │   ├── loaders.py                # Data loaders
│   │   ├── augment.py                # Augmentation functions
│   │   └── ...
│   ├── cfg/                          # Configuration files
│   │   └── models/                   # Model configurations
│   └── ...
│
├── runs/                             # Training outputs (created during training)
│   ├── train/                        # Training logs and weights
│   └── val/                          # Validation results
│
└── data/                             # Dataset directory (created during processing)
    ├── raw/                          # Original dataset
    ├── selected/                     # After class selection
    ├── split/                        # After train/val/test split
    └── augmented/                    # After augmentation
```

## File Descriptions

### Core Scripts

| File | Purpose |
|------|---------|
| `train.py` | Main training script - configure model, data, and hyperparameters |
| `val.py` | Validation and evaluation script - generates detailed metrics |
| `test_environment.py` | Validates environment setup and dependencies |

### Data Processing Scripts

| File | Purpose |
|------|---------|
| `01_select_defect_classes.py` | Filters dataset to include only target defect classes |
| `02_split_dataset.py` | Splits dataset into train/val/test sets |
| `03_augment_and_balance.py` | Augments and balances training set |

### Configuration Files

| File | Purpose |
|------|---------|
| `dataset_config.yaml` | Configuration for all data processing steps |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore rules |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete project documentation |
| `QUICKSTART.md` | Quick start guide with examples |
| `PROJECT_INFO.md` | This file - project overview |

## Workflow

### 1. Environment Setup
```bash
pip install -r requirements.txt
python test_environment.py
```

### 2. Data Preparation
```bash
python 01_select_defect_classes.py
python 02_split_dataset.py
python 03_augment_and_balance.py
```

### 3. Model Training
```bash
python train.py
```

### 4. Model Evaluation
```bash
python val.py
```

## Configuration

### dataset_config.yaml
Central configuration file for data processing:
- Input/output paths
- Target defect classes
- Data augmentation parameters
- Class balancing settings

### train.py
Training configuration:
- Model architecture
- Training hyperparameters
- Dataset paths
- Output directory

### val.py
Validation configuration:
- Model weights path
- Validation dataset
- Evaluation metrics

## Dependencies

### Required
- Python 3.8+
- PyTorch 1.8+
- TorchVision 0.9+
- OpenCV 4.6+
- NumPy 1.22+
- PyYAML 5.3+
- tqdm 4.64+
- Pillow 7.1+

### Optional (for data processing)
- Albumentations (data augmentation)
- PrettyTable (result visualization)

## Performance Metrics

The model outputs comprehensive metrics:

### Model Information
- GFLOPs (computational complexity)
- Parameters (model size)
- Inference time (per image)
- FPS (frames per second)
- Model file size

### Detection Metrics
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **mAP50:** Mean average precision at IoU=0.5
- **mAP75:** Mean average precision at IoU=0.75
- **mAP50-95:** Mean average precision at IoU=0.5:0.95

## Tips for Publication

1. **Use official results:** Export metrics from `val.py` output
2. **Document configuration:** Save all hyperparameters and settings
3. **Reproducibility:** Use fixed random seeds (already configured)
4. **Comparison:** Compare with baseline models and state-of-the-art
5. **Ablation study:** Test different model variants and augmentation strategies

## Troubleshooting

### Common Issues

**Issue:** Module not found errors
- **Solution:** Run `pip install -r requirements.txt` and `python test_environment.py`

**Issue:** CUDA out of memory
- **Solution:** Reduce batch size or image size in configuration

**Issue:** Dataset not found
- **Solution:** Verify paths in configuration files and ensure data is in correct format

**Issue:** Poor model performance
- **Solution:** Check data quality, adjust augmentation parameters, increase training epochs

## Future Improvements

Potential enhancements:
- Multi-scale feature extraction
- Attention mechanisms optimization
- Knowledge distillation for model compression
- Quantization for edge deployment
- Additional defect classes
- Real-time video processing

## References

- RT-DETR: Real-Time Deformable Transformer for End-to-End Object Detection
- Vision Transformers for Object Detection
- Deformable Convolutional Networks

## License

This project is provided for research and educational purposes.

## Contact

For questions or issues, please refer to the main README.md or open an issue on the repository.

---

**Last Updated:** 2024
**Version:** 1.0
