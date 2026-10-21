"""
Baseline training/validation script for standard YOLOv8 (no MaxSig).

Use this to compare performance against the MaxSig-integrated model.

Example:
  python train_yolov8_baseline.py \
    --data cell_dataset.yaml \
    --variant yolov8n \
    --epochs 150 \
    --imgsz 640 \
    --batch 16
"""

import argparse
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='cell_dataset.yaml', help='Dataset YAML path')
    p.add_argument('--variant', type=str, default='yolov8n', help='YOLOv8 variant: yolov8n/s/m/l/x')
    p.add_argument('--epochs', type=int, default=150, help='Training epochs')
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('--batch', type=int, default=16, help='Batch size')
    p.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='cuda or cpu')
    p.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    p.add_argument('--project', type=str, default='cell_baseline', help='Project name')
    p.add_argument('--name', type=str, default='baseline', help='Run/experiment name')
    p.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs')
    p.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()

    print('Baseline YOLOv8 training (no MaxSig)')
    print('=' * 50)
    print({
        'variant': args.variant,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'project': args.project,
        'name': args.name,
    })

    # Load pretrained YOLOv8 model
    model = YOLO(f'{args.variant}.pt')  # auto-downloads if missing

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        save=True,
        save_period=args.save_period,
        patience=args.patience,
        seed=args.seed,
    )

    # Validate
    print('\nValidating baseline model...')
    r = model.val(data=args.data, imgsz=args.imgsz, device=args.device, workers=args.workers)
    try:
        metrics = {
            'mAP50-95': r.box.map if hasattr(r, 'box') else None,
            'mAP50':    (r.box.maps[0] if hasattr(r, 'box') and hasattr(r.box, 'maps') and len(r.box.maps) > 0 else None),
            'precision': getattr(getattr(r, 'metrics', None), 'precision', None),
            'recall':    getattr(getattr(r, 'metrics', None), 'recall', None)
        }
        print(metrics)
        # Explicit lines for easy comparison/log parsing
        print(f"mAP50-95: {metrics['mAP50-95']}")
        print(f"mAP50:    {metrics['mAP50']}")
        print(f"Precision:{metrics['precision']}")
        print(f"Recall:   {metrics['recall']}")
    except Exception as e:
        print(f'(Info) Could not print metrics summary: {e}')

    print('\n✓ Baseline training/validation completed.')
    print(f"Results saved under runs/train/{args.name} and runs/val/{args.name}")


if __name__ == '__main__':
    main()


