"""
YOLOv11 MaxSigLayer Configuration

This file contains configurations for integrating MaxSigLayer with YOLOv11
for steel defect detection tasks.

Author: Reza Yazdi
Date: 2024
"""

# Configuration constants
KERNEL_SIG = 5  # Small kernel for fine defect details


def get_maxsig_config(variant='yolo11n', custom_config=None):

    """
    Get MaxSigLayer configuration for a specific YOLOv11 variant.

    Args:
        variant (str): YOLOv11 variant
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

    # Variant-specific configurations (keys are YOLOv11 variants)
    # Updated for YOLOv11 C2fPSA and C3k2 modules
    variant_configs = {
        'yolo11n': {
            'maxsig_layers': [2, 3, 4], 
            'wrap_backbone_c2fpsa': 1,  # C2fPSA modules in backbone
            'wrap_backbone_c3k2': 1,    # C3k2 modules in backbone
            'wrap_neck_c2f': 1,         # Standard C2f modules in neck
            'psa_ratio': 4,
            'psa_use_conv': True
        },
        'yolo11s': {
            'maxsig_layers': [2, 3, 4, 5], 
            'wrap_backbone_c2fpsa': 1, 
            'wrap_backbone_c3k2': 1, 
            'wrap_neck_c2f': 1,
            'psa_ratio': 4,
            'psa_use_conv': True
        },
        'yolo11m': {
            'maxsig_layers': [2, 3, 4, 5, 6], 
            'wrap_backbone_c2fpsa': 2, 
            'wrap_backbone_c3k2': 2, 
            'wrap_neck_c2f': 2,
            'psa_ratio': 4,
            'psa_use_conv': True
        },
        'yolo11l': {
            'maxsig_layers': [2, 3, 4, 5, 6, 7], 
            'wrap_backbone_c2fpsa': 3, 
            'wrap_backbone_c3k2': 3, 
            'wrap_neck_c2f': 3,
            'psa_ratio': 4,
            'psa_use_conv': True
        },
        'yolo11x': {
            'maxsig_layers': [2, 3, 4, 5, 6, 7, 8], 
            'wrap_backbone_c2fpsa': 3, 
            'wrap_backbone_c3k2': 3, 
            'wrap_neck_c2f': 3,
            'psa_ratio': 4,
            'psa_use_conv': True
        }
    }

    # Accept historical naming 'yolov11*' by normalizing to 'yolo11*'
    if variant.startswith('yolov11'):
        variant = variant.replace('yolov11', 'yolo11', 1)

    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(variant_configs.keys())}")

    config = base_config.copy()
    config.update(variant_configs[variant])

    if custom_config:
        config.update(custom_config)

    return config


def create_cell_detection_config(variant='yolo11n'):

    """
    Create a specialized configuration for cell detection (WBC, RBC, Platelets).

    This configuration is optimized for cell detection tasks where
    fine-grained feature extraction and contrast enhancement are crucial
    for distinguishing between different cell types.
    
    This is the SINGLE SOURCE OF TRUTH for all training parameters.
    Command line arguments can only override specific settings.
    """

    # Use variant-aware defaults and base settings
    maxsig_variant_config = get_maxsig_config(variant)

    return {
        'maxsig_config': maxsig_variant_config,
        'training_config': {
            'epochs': 150,  # More epochs for cell detection
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.008,  # Slightly lower learning rate for fine features
            'lrf': 0.008,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,  # Longer warmup for cell detection
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True
        }
    }


if __name__ == "__main__":
    # Test the configuration functions
    print("Testing YOLOv11 MaxSigLayer Configuration")
    print("=" * 40)

    # Select variant for testing
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else 'yolo11s'

    # Test get_maxsig_config
    config = get_maxsig_config(variant)
    print(f"{variant} Config: {config}")

    print("✓ Configuration test completed!")


