"""
RTDETR Validation Script
Evaluates model performance and generates comprehensive metrics report.
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def get_model_size(model_path):
    """Get model file size in MB."""
    stats = os.stat(model_path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def generate_metrics_report(result, model_path):
    """Generate comprehensive metrics report for model evaluation."""
    if model_path.task != 'detect':
        return

    num_classes = result.box.p.size
    class_names = list(result.names.values())

    # Extract timing metrics
    preprocess_time = result.speed['preprocess']
    inference_time = result.speed['inference']
    postprocess_time = result.speed['postprocess']
    total_time = preprocess_time + inference_time + postprocess_time

    # Extract model architecture metrics
    num_layers, num_params, num_gradients, flops = model_info(model_path.model)

    # Print header
    print('\n' + '='*70)
    print('RTDETR Model Evaluation Report'.center(70))
    print('='*70 + '\n')

    # Model Information Table
    model_info_table = PrettyTable()
    model_info_table.title = "Model Architecture & Performance Metrics"
    model_info_table.field_names = [
        "GFLOPs",
        "Parameters",
        "Preprocess (ms)",
        "Inference (ms)",
        "Postprocess (ms)",
        "Total FPS",
        "Inference FPS",
        "Model Size (MB)"
    ]
    model_info_table.add_row([
        f'{flops:.1f}',
        f'{num_params:,}',
        f'{preprocess_time / 1000:.6f}',
        f'{inference_time / 1000:.6f}',
        f'{postprocess_time / 1000:.6f}',
        f'{1000 / total_time:.2f}',
        f'{1000 / inference_time:.2f}',
        f'{get_model_size(model_path)}'
    ])
    print(model_info_table)

    # Detection Metrics Table
    metrics_table = PrettyTable()
    metrics_table.title = "Per-Class Detection Metrics"
    metrics_table.field_names = [
        "Class",
        "Precision",
        "Recall",
        "F1-Score",
        "mAP50",
        "mAP75",
        "mAP50-95"
    ]

    for idx in range(num_classes):
        metrics_table.add_row([
            class_names[idx],
            f"{result.box.p[idx]:.4f}",
            f"{result.box.r[idx]:.4f}",
            f"{result.box.f1[idx]:.4f}",
            f"{result.box.ap50[idx]:.4f}",
            f"{result.box.all_ap[idx, 5]:.4f}",
            f"{result.box.ap[idx]:.4f}"
        ])

    # Add average metrics row
    metrics_table.add_row([
        "Average",
        f"{result.results_dict['metrics/precision(B)']:.4f}",
        f"{result.results_dict['metrics/recall(B)']:.4f}",
        f"{np.mean(result.box.f1[:num_classes]):.4f}",
        f"{result.results_dict['metrics/mAP50(B)']:.4f}",
        f"{np.mean(result.box.all_ap[:num_classes, 5]):.4f}",
        f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
    ])
    print(metrics_table)

    # Save report to file
    report_path = Path(result.save_dir) / 'evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('RTDETR Model Evaluation Report\n')
        f.write('='*70 + '\n\n')
        f.write(str(model_info_table))
        f.write('\n\n')
        f.write(str(metrics_table))

    print('\n' + '='*70)
    print(f'Evaluation report saved to: {report_path}'.center(70))
    print('='*70 + '\n')


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'runs/train/exp-AIFI-EPGO/weights/best.pt'
    DATA_YAML = '/root/dataset/dataset_visdrone/data.yaml'

    # Initialize model
    model = RTDETR(MODEL_PATH)

    # Run validation
    result = model.val(
        data=DATA_YAML,
        split='test',
        imgsz=640,
        batch=4,
        project='runs/val',
        name='exp',
    )

    # Generate evaluation report
    generate_metrics_report(result, model)