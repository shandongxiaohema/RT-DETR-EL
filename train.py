"""
RTDETR Training Script
Trains RTDETR model on custom dataset with specified configuration.
"""

import warnings
import os

warnings.filterwarnings('ignore')

from ultralytics import RTDETR


def main():
    """Main training function."""
    # Model configuration
    MODEL_CONFIG = 'ultralytics/cfg/models/rtdetr/rtdetr_GGA_MDP.yaml'
    DATA_CONFIG = 'dataset/data.yaml'

    # Training parameters
    TRAIN_PARAMS = {
        'data': DATA_CONFIG,
        'cache': False,
        'imgsz': 640,
        'epochs': 150,
        'batch': 4,
        'workers': 4,
        'project': 'runs/train',
        'name': 'exp_rtdetr_GGA_MDP',
    }

    # Initialize and train model
    model = RTDETR(MODEL_CONFIG)
    model.train(**TRAIN_PARAMS)


if __name__ == '__main__':
    main()