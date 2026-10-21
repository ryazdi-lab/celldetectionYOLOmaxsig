"""
Training Script for Cell Detection with YOLOv8 + MaxSigLayer

This script demonstrates how to train a YOLOv8 model with MaxSigLayer
integration for cell detection tasks. All configurations are
loaded from yolov8_maxsig_configs.py to avoid duplication.

Usage:
    python train_cell_detection_yolov8_maxsig.py --variant yolov8n --data cell_dataset.yaml

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
from yolov8_maxsig_integration import modify_yolov8_backbone, MaxSigC2f
from yolov8_maxsig_configs import get_maxsig_config, create_cell_detection_config


class CellDetectionYOLOv8MaxSig:
    """
    Cell Detection with YOLOv8 and MaxSigLayer integration.
    """
    
    def __init__(self, variant, pretrained=True, custom_config=None):
        """
        Initialize the cell detection model.
        
        Args:
            variant (str): YOLOv8 variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            pretrained (bool): Whether to use pretrained weights
            custom_config (dict): Custom MaxSigLayer configuration
        """
        
        self.variant = variant
        self.pretrained = pretrained
        
        # Get MaxSig configuration - Force use of configs.py
        if custom_config is None:
            # Use cell detection optimized configuration from configs.py
            cell_config = create_cell_detection_config()
            self.maxsig_config = cell_config['maxsig_config']
            self.training_config = cell_config['training_config']
        else:
            # Use custom config but still get training config from configs.py
            self.maxsig_config = custom_config
            cell_config = create_cell_detection_config()
            self.training_config = cell_config['training_config']
        
        # Print MaxSigLayer configuration
        print(f"\nMaxSigLayer Configuration for {self.variant}:")
        for key, value in self.maxsig_config.items():
            print(f"  {key}: {value}")
        
        # Initialize model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model with MaxSigLayer integration."""
        
        try:
            # Load base model
            if self.pretrained:
                self.model = YOLO(f'{self.variant}.pt')
            else:
                self.model = YOLO(f'{self.variant}.yaml')
            
            # Modify backbone with MaxSigLayer
            self.model = modify_yolov8_backbone(
                model_path=f'{self.variant}.pt' if self.pretrained else f'{self.variant}.yaml',
                maxsig_config=self.maxsig_config
            )
            
            
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
        
        try:
            if data_path:
                results = self.model.val(data=data_path)
            else:
                results = self.model.val()
            
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
    Create a YAML dataset configuration file for YOLOv8.
    
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 with MaxSigLayer for cell detection')
    parser.add_argument('--variant', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLOv8 variant to use')
    parser.add_argument('--data', type=str, default='cell_dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                       help='Device to use for training')
    parser.add_argument('--epochs', type=int, help='Override training epochs (default: from config file)')
    return parser.parse_args()


def main():
    """Main training function - uses configurations from yolov8_maxsig_configs.py."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Get configuration from configs file
    cell_config = create_cell_detection_config()
    maxsig_config = cell_config['maxsig_config']
    training_config = cell_config['training_config']
    
    # Cell detection class names (single-class by default)
    CLASS_NAMES = ['cell']
    
    print("Cell Detection with YOLOv8 + MaxSigLayer")
    print("Using configurations from yolov8_maxsig_configs.py")
    print("=" * 60)
    print(f"Model Variant: {args.variant}")
    print(f"Device: {args.device}")
    print(f"Data Path: {args.data}")
    print("\nMaxSig Configuration:")
    for key, value in maxsig_config.items():
        print(f"  {key}: {value}")
    print("\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Initialize model
    try:
        model = CellDetectionYOLOv8MaxSig(
            variant=args.variant,
            pretrained=True,
            custom_config=None  # Force use of configs.py
        )
        
        # Check if dataset exists
        if not os.path.exists(args.data):
            print(f"Dataset file {args.data} not found. Creating example...")
            create_dataset_yaml(
                data_dir='./cell_data',
                train_dir='images/train',
                val_dir='images/val',
                class_names=CLASS_NAMES,
                output_path=args.data
            )
            print(f"Please update {args.data} with your actual dataset paths.")
            return
        
        # Override device in training config
        training_config['device'] = args.device
        
        # Train the model using configs from configs.py
        results = model.train(
            data_path=args.data,
            **training_config  # Use all training config from configs.py
        )
        
        # Validate the model
        r = model.validate(args.data)
        
        # Save the model
        model_path = f'{args.variant}_maxsig_cell_detection.pt'
        model.save_model(model_path)
        
        print(f"✓ Training completed! Model saved as: {model_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
