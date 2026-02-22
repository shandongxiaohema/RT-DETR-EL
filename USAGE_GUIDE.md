# RTDETR Training & Evaluation Guide

This project provides a standardized implementation for training and evaluating RTDETR (Real-Time DETR) models using the Ultralytics framework.

## Project Structure

```
├── train.py              # Training script
├── val.py                # Validation and evaluation script
├── predict.py            # Inference script
├── export.py             # Model export script
├── requirements.txt      # Python dependencies
├── dataset/              # Dataset directory
│   └── data.yaml         # Dataset configuration
└── runs/                 # Output directory
    ├── train/            # Training outputs
    ├── val/              # Validation outputs
    └── detect/           # Detection outputs
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training

```bash
python train.py
```

Configure training parameters in `train.py`:
- `MODEL_CONFIG`: Model architecture YAML file
- `DATA_CONFIG`: Dataset configuration YAML file
- `TRAIN_PARAMS`: Training hyperparameters

### 2. Validation

```bash
python val.py
```

Generates comprehensive evaluation report including:
- Model architecture metrics (GFLOPs, parameters)
- Performance metrics (FPS, inference time)
- Per-class detection metrics (Precision, Recall, mAP)

### 3. Inference

```bash
python predict.py
```

Run detection on images or video streams. Update `SOURCE` path in the script.

### 4. Model Export

```bash
python export.py
```

Export trained model to various formats:
- ONNX
- TorchScript
- TensorFlow Lite
- CoreML

## Configuration

### Dataset Configuration (data.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['defect']  # Class names
```

### Model Configuration

Edit model YAML files in `ultralytics/cfg/models/rtdetr/` to customize architecture.

## Output Files

### Training
- `runs/train/exp_*/weights/best.pt` - Best model weights
- `runs/train/exp_*/results.csv` - Training metrics

### Validation
- `runs/val/exp/evaluation_report.txt` - Comprehensive evaluation report
- `runs/val/exp/confusion_matrix.png` - Confusion matrix visualization

### Detection
- `runs/detect/exp/` - Detection results with visualizations

## Evaluation Metrics

The validation script generates:

1. **Model Architecture Metrics**
   - GFLOPs: Floating point operations
   - Parameters: Total model parameters
   - Model Size: File size in MB

2. **Performance Metrics**
   - Preprocess/Inference/Postprocess time
   - Total FPS and Inference FPS

3. **Detection Metrics (Per-Class)**
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1-Score: Harmonic mean of precision and recall
   - mAP50: Mean average precision at IoU=0.50
   - mAP75: Mean average precision at IoU=0.75
   - mAP50-95: Mean average precision across IoU thresholds

## Advanced Usage

### Custom Training Parameters

Edit `train.py` to modify:
```python
TRAIN_PARAMS = {
    'epochs': 150,
    'batch': 4,
    'imgsz': 640,
    'workers': 4,
    # Add more parameters as needed
}
```

### Custom Inference Parameters

Edit `predict.py` to modify:
```python
results = predict(
    model_path=MODEL_PATH,
    source=SOURCE,
    conf=0.25,      # Confidence threshold
    iou=0.45,       # IoU threshold
    imgsz=640,      # Image size
)
```

## Troubleshooting

### Out of Memory
- Reduce `batch` size in training parameters
- Reduce `imgsz` (image size)
- Enable `cache=False` in training

### Low Performance
- Increase training `epochs`
- Adjust learning rate in model configuration
- Ensure dataset is properly balanced

## References

- [Ultralytics RTDETR Documentation](https://docs.ultralytics.com/models/rtdetr/)
- [RTDETR Paper](https://arxiv.org/abs/2304.08069)

## License

This project uses Ultralytics YOLOv8 framework. Please refer to the respective licenses.
