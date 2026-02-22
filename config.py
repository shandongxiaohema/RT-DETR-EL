"""
Configuration Template for RTDETR Training
Modify this file to customize training parameters.
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Path to model architecture YAML file
MODEL_CONFIG = 'ultralytics/cfg/models/rtdetr/rtdetr_GGA_MDP.yaml'

# ============================================================================
# Dataset Configuration
# ============================================================================

# Path to dataset configuration YAML file
DATA_CONFIG = 'dataset/data.yaml'

# ============================================================================
# Training Parameters
# ============================================================================

TRAIN_PARAMS = {
    # Data
    'data': DATA_CONFIG,
    'cache': False,  # Cache images for faster training

    # Model
    'imgsz': 640,  # Input image size

    # Training
    'epochs': 150,  # Number of training epochs
    'batch': 4,  # Batch size
    'workers': 4,  # Number of data loading workers

    # Optimization
    'optimizer': 'SGD',  # Optimizer (SGD, Adam, AdamW)
    'lr0': 0.01,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.0005,  # Weight decay

    # Augmentation
    'hsv_h': 0.015,  # HSV hue augmentation
    'hsv_s': 0.7,  # HSV saturation augmentation
    'hsv_v': 0.4,  # HSV value augmentation
    'degrees': 0.0,  # Rotation degrees
    'translate': 0.1,  # Translation
    'scale': 0.5,  # Scale
    'flipud': 0.0,  # Flip upside-down
    'fliplr': 0.5,  # Flip left-right
    'mosaic': 1.0,  # Mosaic augmentation
    'mixup': 0.0,  # Mixup augmentation

    # Output
    'project': 'runs/train',  # Project directory
    'name': 'exp_rtdetr_GGA_MDP',  # Experiment name
    'save': True,  # Save model checkpoints
    'save_period': -1,  # Save checkpoint every N epochs (-1 = disabled)

    # Validation
    'val': True,  # Validate during training
    'patience': 50,  # Early stopping patience

    # Device
    'device': 0,  # GPU device ID (0 for first GPU, -1 for CPU)
    'half': False,  # Use FP16 precision

    # Logging
    'verbose': True,  # Verbose output
}

# ============================================================================
# Validation Parameters
# ============================================================================

VAL_PARAMS = {
    'data': DATA_CONFIG,
    'split': 'test',  # Validation split
    'imgsz': 640,  # Image size
    'batch': 4,  # Batch size
    'project': 'runs/val',  # Project directory
    'name': 'exp',  # Experiment name
}

# ============================================================================
# Inference Parameters
# ============================================================================

PREDICT_PARAMS = {
    'conf': 0.25,  # Confidence threshold
    'iou': 0.45,  # IoU threshold for NMS
    'imgsz': 640,  # Image size
    'save': True,  # Save results
    'project': 'runs/detect',  # Project directory
    'name': 'exp',  # Experiment name
}

# ============================================================================
# Export Parameters
# ============================================================================

EXPORT_PARAMS = {
    'format': 'onnx',  # Export format (onnx, torchscript, tflite, etc.)
    'imgsz': 640,  # Image size
    'half': False,  # Use FP16 precision
    'optimize': False,  # Optimize model
}
