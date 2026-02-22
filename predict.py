"""
RTDETR Inference Script
Performs object detection inference on images or video streams.
"""

import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from ultralytics import RTDETR


def predict(
    model_path,
    source,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    save=True,
    project='runs/detect',
    name='exp',
    **kwargs
):
    """
    Run inference on images or video.

    Args:
        model_path (str): Path to trained model weights
        source (str): Path to image/video or directory
        conf (float): Confidence threshold
        iou (float): IoU threshold for NMS
        imgsz (int): Inference image size
        save (bool): Save results
        project (str): Project directory
        name (str): Experiment name
        **kwargs: Additional arguments for model.predict()
    """
    model = RTDETR(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save,
        project=project,
        name=name,
        **kwargs
    )

    return results


def main():
    """Main inference function."""
    # Configuration
    MODEL_PATH = 'runs/train/exp-AIFI-EPGO/weights/best.pt'
    SOURCE = 'path/to/image_or_video'  # Change to your source

    # Run inference
    results = predict(
        model_path=MODEL_PATH,
        source=SOURCE,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=True,
        project='runs/detect',
        name='exp',
    )

    # Process results
    for result in results:
        print(f"Detected {len(result.boxes)} objects")
        if result.boxes:
            print(result.boxes.data)


if __name__ == '__main__':
    main()
