"""
YOLOv8 MaxSigLayer Configuration

This file contains configurations for integrating MaxSigLayer with YOLOv8
for steel defect detection tasks.

Author: Reza Yazdi
Date: 2024
"""

# Configuration constants
KERNEL_SIG = 5  # Small kernel for fine defect details

def get_maxsig_config(variant, custom_config=None):
    """
    Get MaxSigLayer configuration for a specific YOLOv8 variant.
    
    Args:
        variant (str): YOLOv8 variant
        custom_config (dict): Custom configuration to override defaults
        
    Returns:
        dict: MaxSigLayer configuration
    """
    
    # Base configuration for all variants (optimized for steel defect detection)
    base_config = {
        'kernel_sig': KERNEL_SIG,
        'use_maxsig': True,
        'maxsig_activation': 'leaky_relu',
        'use_residual': True,
        'use_bn': True
    }
    
    # Variant-specific configurations
    variant_configs = {
        'yolov8n': {'maxsig_layers': [2, 3, 4], 'wrap_backbone_c2f': 1, 'wrap_neck_c2f': 2},
        'yolov8s': {'maxsig_layers': [2, 3, 4, 5], 'wrap_backbone_c2f': 2, 'wrap_neck_c2f': 3},
        'yolov8m': {'maxsig_layers': [2, 3, 4, 5, 6], 'wrap_backbone_c2f': 3, 'wrap_neck_c2f': 3},
        'yolov8l': {'maxsig_layers': [2, 3, 4, 5, 6, 7], 'wrap_backbone_c2f': 3, 'wrap_neck_c2f': 4},
        'yolov8x': {'maxsig_layers': [2, 3, 4, 5, 6, 7, 8], 'wrap_backbone_c2f': 4, 'wrap_neck_c2f': 4}
    }
    
    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(variant_configs.keys())}")
    
    config = base_config.copy()
    config.update(variant_configs[variant])
    
    if custom_config:
        config.update(custom_config)
    
    return config



def create_cell_detection_config():
    """
    Create a specialized configuration for cell detection.

    Optimized defaults for microscopic cell detection where subtle contrast
    differences matter. You can tweak these later per dataset.
    """

    return {
        'maxsig_config': {
            'kernel_sig': KERNEL_SIG,
            'use_maxsig': True,
            'maxsig_activation': 'leaky_relu',
            'maxsig_layers': [2, 3, 4],
            'wrap_backbone_c2f': 2,
            'wrap_neck_c2f': 2,
            'use_residual': True,
            'use_bn': True
        },
        'training_config': {
            'epochs': 150,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.005,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True
        }
    }