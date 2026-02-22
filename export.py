"""
RTDETR Model Export Script
Exports trained model to various formats (ONNX, TorchScript, etc.).
"""

import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from ultralytics import RTDETR


def export_model(
    model_path,
    format='onnx',
    imgsz=640,
    half=False,
    optimize=False,
    **kwargs
):
    """
    Export model to specified format.

    Args:
        model_path (str): Path to trained model weights
        format (str): Export format (onnx, torchscript, tflite, etc.)
        imgsz (int): Model input size
        half (bool): Use FP16 precision
        optimize (bool): Optimize model
        **kwargs: Additional export arguments
    """
    model = RTDETR(model_path)

    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        optimize=optimize,
        **kwargs
    )

    print(f"Model exported to: {export_path}")
    return export_path


def main():
    """Main export function."""
    # Configuration
    MODEL_PATH = 'runs/train/exp-AIFI-EPGO/weights/best.pt'

    # Export to ONNX
    print("Exporting to ONNX format...")
    export_model(
        model_path=MODEL_PATH,
        format='onnx',
        imgsz=640,
        half=False,
    )

    # Export to TorchScript
    print("\nExporting to TorchScript format...")
    export_model(
        model_path=MODEL_PATH,
        format='torchscript',
        imgsz=640,
    )


if __name__ == '__main__':
    main()
