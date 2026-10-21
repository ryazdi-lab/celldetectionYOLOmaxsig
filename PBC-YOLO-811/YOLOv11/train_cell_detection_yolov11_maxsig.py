"""
Training Script for Cell Detection with YOLOv11 + MaxSigLayer

This script demonstrates how to train a YOLOv11 model with MaxSigLayer
integration for cell detection tasks (WBC, RBC, Platelets).

Author: Reza Yazdi
Date: 2024
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml
import os
import glob
import math
import argparse
from yolov11_maxsig_integration import modify_yolov11_backbone, MaxSigC2fPSA, MaxSigC3k2
from yolov11_maxsig_configs import create_cell_detection_config


class CellDetectionYOLOv11MaxSig:
    """
    Cell Detection with YOLOv11 and MaxSigLayer integration.
    """
    
    def __init__(self, variant='yolo11n', pretrained=True, custom_config=None):
        """
        Initialize the cell detection model.
        
        Args:
            variant (str): YOLOv11 variant ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
            pretrained (bool): Whether to use pretrained weights
            custom_config (dict): Custom MaxSigLayer configuration
        """
        
        self.variant = variant
        self.pretrained = pretrained
        
        # Get MaxSig configuration - SINGLE SOURCE OF TRUTH from configs.py
        if custom_config is None:
            # Use cell detection optimized configuration from configs.py, respecting variant
            # This is the SINGLE SOURCE OF TRUTH for all training parameters
            cell_config = create_cell_detection_config(self.variant)
            self.maxsig_config = cell_config['maxsig_config']
            self.training_config = cell_config['training_config']
        else:
            # Use custom config but still get training config from configs.py
            self.maxsig_config = custom_config
            cell_config = create_cell_detection_config(self.variant)
            self.training_config = cell_config['training_config']
        
        # Print MaxSigLayer configuration
        print(f"MaxSig Config: {self.maxsig_config}")
        
        # Initialize model
        self.model = None
        self.last_run_dir = None  # where training logged results (project/name[<n>])
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv11 model with MaxSigLayer integration."""
        
        try:
            # Convert variant name to correct Ultralytics format
            # Handle both 'yolov11n' and 'yolo11n' formats
            if self.variant.startswith('yolov11'):
                # Convert 'yolov11n' to 'yolo11n'
                ultralytics_variant = self.variant.replace('yolov11', 'yolo11')
            else:
                ultralytics_variant = self.variant
            
            # Load base model - let Ultralytics handle downloading if needed
            if self.pretrained:
                # Use the variant name with .pt extension - Ultralytics will download if needed
                self.model = YOLO(f'{ultralytics_variant}.pt')
            else:
                self.model = YOLO(f'{ultralytics_variant}.yaml')
            
            # Modify backbone with MaxSigLayer for YOLOv11 C2fPSA and C3k2 modules
            # Handle both old and new function signatures for compatibility
            try:
                # Try new signature first (with model parameter)
                self.model = modify_yolov11_backbone(
                    model=self.model,  # Pass the loaded model object
                    maxsig_config=self.maxsig_config
                )
            except TypeError as e:
                if "unexpected keyword argument 'model'" in str(e):
                    # Fallback to old signature (with model_path parameter)
                    print("Using fallback method for model modification...")
                    self.model = modify_yolov11_backbone(
                        model_path=f'{ultralytics_variant}.pt' if self.pretrained else f'{ultralytics_variant}.yaml',
                        maxsig_config=self.maxsig_config
                    )
                else:
                    raise
            
            
        except Exception as e:
            print(f"✗ Error initializing model: {e}")
            raise
    
    
    def train(self, data_path, **kwargs):
        """
        Train the model on cell detection data.
        
        Args:
            data_path (str): Path to the dataset YAML file
            **kwargs: Additional training parameters
        """
        
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model() first.")
        
        # Merge training configuration with provided kwargs
        train_config = self.training_config.copy()
        train_config.update(kwargs)
        
        try:
            # Train the model
            results = self.model.train(
                data=data_path,
                **train_config
            )

            # Capture the exact logging/run directory for later validation
            run_dir = None
            try:
                # Newer Ultralytics exposes trainer.save_dir
                run_dir = getattr(getattr(self.model, 'trainer', None), 'save_dir', None)
            except Exception:
                pass

            if run_dir is None:
                # Some versions return a result object with save_dir
                run_dir = getattr(results, 'save_dir', None)

            if run_dir is None:
                # Fallback: infer from provided project/name by choosing most recent
                project = train_config.get('project', 'runs/detect')
                name = train_config.get('name', f'{self.variant}_maxsig')
                base = os.path.join(project)
                pattern = os.path.join(base, f'{name}*')
                candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
                if candidates:
                    run_dir = max(candidates, key=lambda p: os.path.getmtime(p))

            self.last_run_dir = run_dir
            if self.last_run_dir:
                print(f"✓ Logging results to {self.last_run_dir}")
            else:
                print("⚠ Could not determine training log directory.")

            return results
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            raise
    
    def validate(self, data_path=None):
        """
        Validate the trained model.
        
        Args:
            data_path (str): Path to the dataset YAML file (optional)
        """
        
        if self.model is None:
            raise ValueError("Model not initialized.")

        # Enforce MaxSig-only: validate ONLY models trained under MaxSig runs.
        # Do NOT change the graph at validation time (no wrapping). Load and validate as saved.
        try:
            # Prefer the exact directory captured at train time
            weights_path = None
            if self.last_run_dir:
                run_best = os.path.join(self.last_run_dir, 'weights', 'best.pt')
                if os.path.exists(run_best):
                    weights_path = run_best
                    print(f"✓ Using best.pt from training log directory: {weights_path}")

            if weights_path is None:
                # Locate best checkpoint among possible MaxSig run name variants: {variant}_maxsig*
                base_dir = os.path.join('cell_detection')
                pattern = os.path.join(base_dir, f'{self.variant}_maxsig*', 'weights', 'best.pt')
                candidate_paths = [p for p in glob.glob(pattern) if os.path.exists(p)]
                # Deduplicate and choose the most recent by modification time
                candidate_paths = list(dict.fromkeys(candidate_paths))
                if not candidate_paths:
                    raise FileNotFoundError(
                        f"No MaxSig checkpoint found under '{base_dir}' matching '{self.variant}_maxsig*/weights/best.pt'."
                    )
                weights_path = max(candidate_paths, key=lambda p: os.path.getmtime(p))
                print(f"✓ Using MaxSigLayer-trained checkpoint for validation: {weights_path}")

            # Validate without modifying the graph
            base_model = YOLO(weights_path)

            if data_path:
                results = base_model.val(data=data_path)
            else:
                results = base_model.val()

            return results

        except Exception as e:
            print(f"✗ Validation failed: {e}")
            raise
    
    def predict(self, source, **kwargs):
        """
        Run inference on new images.
        
        Args:
            source: Image path, directory, or video
            **kwargs: Additional prediction parameters
        """
        
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        try:
            results = self.model(source, **kwargs)
            return results
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            raise
    
    def save_model(self, path):
        """Save the trained model."""
        
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        try:
            self.model.save(path)
            pass
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            raise


def create_dataset_yaml(data_dir, train_dir, val_dir, class_names, output_path='cell_dataset.yaml'):
    """
    Create a YAML dataset configuration file for YOLOv11.
    
    Args:
        data_dir (str): Root directory of the dataset
        train_dir (str): Training images directory (relative to data_dir)
        val_dir (str): Validation images directory (relative to data_dir)
        class_names (list): List of class names
        output_path (str): Output YAML file path
    """
    
    dataset_config = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return output_path


def parse_args():
    """Parse command line arguments. Most settings come from config file."""
    parser = argparse.ArgumentParser(description='Cell Detection Training with YOLOv11 + MaxSigLayer')
    parser.add_argument('--data', type=str, default='cell_dataset.yaml', help='Dataset YAML path')
    parser.add_argument('--variant', type=str, default='yolo11n', help='YOLOv11 variant: yolo11n/s/m/l/x')
    parser.add_argument('--epochs', type=int, help='Override training epochs (default: from config file)')
    parser.add_argument('--device', type=str, help='Override device (default: auto-detect)')
    parser.add_argument('--workers', type=int, help='Override dataloader workers (default: from config file)')
    parser.add_argument('--project', type=str, default='cell_detection', help='Project name')
    parser.add_argument('--name', type=str, default='maxsig', help='Run/experiment name')
    parser.add_argument('--save_period', type=int, help='Override save checkpoint period (default: from config file)')
    parser.add_argument('--patience', type=int, help='Override early stopping patience (default: from config file)')
    parser.add_argument('--seed', type=int, help='Override random seed (default: from config file)')
    return parser.parse_args()


def main():
    """Main training function."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Configuration from command line arguments
    YOLO_VARIANT = args.variant
    PRETRAINED = True
    DATA_PATH = args.data
    
    # Cell detection class names (WBC, RBC, Platelets)
    CLASS_NAMES = [
        'WBC',
        'RBC',
        'Platelets'
    ]
    
    print("Cell Detection (WBC, RBC, Platelets) with YOLOv11 + MaxSigLayer")
    print("=" * 60)
    print(f"Basic Configuration:")
    print(f"  Variant: {YOLO_VARIANT}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Epochs Override: {args.epochs if args.epochs is not None else 'Using config file default'}")
    print(f"  Device Override: {args.device if args.device is not None else 'Auto-detect'}")
    print("=" * 60)
    
    # Initialize model
    try:
        model = CellDetectionYOLOv11MaxSig(
            variant=YOLO_VARIANT,
            pretrained=PRETRAINED,
            custom_config=None  # Force use of configs.py
        )
        
        # Check if dataset exists
        if not os.path.exists(DATA_PATH):
            print(f"Dataset file {DATA_PATH} not found. Creating example...")
            create_dataset_yaml(
                data_dir='./cell_detection_data',
                train_dir='images/train',
                val_dir='images/val',
                class_names=CLASS_NAMES,
                output_path=DATA_PATH
            )
            print(f"Please update {DATA_PATH} with your actual dataset paths.")
            return
        

        # Override only specific settings from command line arguments (if provided)
        # All other settings come from the config file (single source of truth)
        overrides = {
            'project': args.project,
            'name': f'{YOLO_VARIANT}_{args.name}',
        }
        
        # Only override if command line argument was provided
        if args.epochs is not None:
            overrides['epochs'] = args.epochs
        if args.device is not None:
            overrides['device'] = args.device
        if args.workers is not None:
            overrides['workers'] = args.workers
        if args.save_period is not None:
            overrides['save_period'] = args.save_period
        if args.patience is not None:
            overrides['patience'] = args.patience
        if args.seed is not None:
            overrides['seed'] = args.seed
            
        model.training_config.update(overrides)
        
        # Print configuration being used
        print(f"Training Configuration (from config file + overrides):")
        for key, value in model.training_config.items():
            print(f"  {key}: {value}")
        print("=" * 60)

        # Train the model using consolidated configuration
        results = model.train(
            data_path=DATA_PATH
        )
        
        # Validate the model
        r = model.validate(DATA_PATH)
        
        # Save the model
        model_path = f'{YOLO_VARIANT}_maxsig_cell_detection.pt'
        model.save_model(model_path)
        
        print(f"✓ Training completed! Model saved as: {model_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
