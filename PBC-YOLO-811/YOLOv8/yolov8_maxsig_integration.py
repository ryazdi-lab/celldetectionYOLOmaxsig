"""
YOLOv8 Backbone Integration with MaxSigLayer Contrast Enhancement

This module provides integration of MaxSigLayer into YOLOv8's backbone architecture.
The MaxSigLayer is inserted after early C2f modules to enhance contrast in feature maps.

Author: Reza Yazdi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF
from ultralytics.nn.tasks import DetectionModel
from MaxSigLayerNormalized import MaxSigLayernormalizedAdvanced


class MaxSigC2f(nn.Module):
    """
    C2f module with integrated MaxSigLayer for contrast enhancement.
    
    This combines the standard C2f module with MaxSigLayer to enhance
    local contrast in feature maps while maintaining the original functionality.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, 
                 kernel_sig=3, use_maxsig=True, maxsig_activation='leaky_relu'):
        super(MaxSigC2f, self).__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e
        self.use_maxsig = use_maxsig
        
        # Standard C2f components
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        # MaxSigLayer for contrast enhancement (applied on the branch with self.c channels)
        if self.use_maxsig:
            self.maxsig = MaxSigLayernormalizedAdvanced(
                ch_in=self.c,
                ch_out=self.c,
                kernel_sig=kernel_sig,
                stride_sig=1,
                use_residual=True,
                use_bn=True
            )
        else:
            self.maxsig = None
    
    @property
    def c(self):
        """Return the number of channels for the bottleneck layers."""
        return int(self.c2 * self.e)
    
    def forward(self, x):
        """Forward pass with optional contrast enhancement."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # Apply MaxSigLayer for contrast enhancement
        if self.use_maxsig and self.maxsig is not None:
            y[-1] = self.maxsig(y[-1])
        
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck module used in C2f."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class YOLOv8MaxSigBackbone(nn.Module):
    """
    Modified YOLOv8 backbone with MaxSigLayer integration.
    
    This backbone replaces standard C2f modules with MaxSigC2f modules
    in the early stages to enhance contrast in feature maps.
    """
    
    def __init__(self, model_cfg, maxsig_config=None):
        super(YOLOv8MaxSigBackbone, self).__init__()
        
        # Default MaxSig configuration
        self.maxsig_config = maxsig_config or {
            'kernel_sig': 3,
            'use_maxsig': True,
            'maxsig_activation': 'leaky_relu',
            'maxsig_layers': [2, 3, 4]  # Which backbone layers to apply MaxSig to
        }
        
        # Parse model configuration
        self.parse_model_cfg(model_cfg)
        
    def parse_model_cfg(self, cfg):
        """Parse YOLOv8 model configuration and create backbone."""
        # This is a simplified version - in practice, you'd parse the full YOLOv8 config
        # For now, we'll create a typical YOLOv8n backbone structure
        
        # YOLOv8n backbone layers (simplified)
        self.backbone_layers = nn.ModuleList()
        
        # Filter supported kwargs for MaxSigC2f
        maxsig_kwargs = {
            k: v for k, v in self.maxsig_config.items()
            if k in {"kernel_sig", "use_maxsig", "maxsig_activation"}
        }
        
        # Initial convolution
        self.backbone_layers.append(Conv(3, 64, 6, 2, 2))  # P1/2
        
        # C2f modules with MaxSig integration
        self.backbone_layers.append(MaxSigC2f(64, 128, 2, True, 1, 0.5, 
                                            **maxsig_kwargs))  # P2/4
        self.backbone_layers.append(Conv(128, 128, 3, 2))  # P2/4
        
        self.backbone_layers.append(MaxSigC2f(128, 256, 2, True, 1, 0.5,
                                            **maxsig_kwargs))  # P3/8
        self.backbone_layers.append(Conv(256, 256, 3, 2))  # P3/8
        
        self.backbone_layers.append(MaxSigC2f(256, 512, 2, True, 1, 0.5,
                                            **maxsig_kwargs))  # P4/16
        self.backbone_layers.append(Conv(512, 512, 3, 2))  # P4/16
        
        self.backbone_layers.append(MaxSigC2f(512, 1024, 2, True, 1, 0.5,
                                            **maxsig_kwargs))  # P5/32
        self.backbone_layers.append(Conv(1024, 1024, 3, 2))  # P5/32
        
        # SPPF
        self.backbone_layers.append(SPPF(1024, 1024, 5))
        
    def forward(self, x):
        """Forward pass through the backbone."""
        features = []
        
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            
            # Collect features at different scales (typical for YOLOv8)
            if i in [2, 4, 6, 8]:  # After C2f modules
                features.append(x)
        
        return features


def modify_yolov8_backbone(model_path, maxsig_config=None):
    """
    Modify an existing YOLOv8 model to include MaxSigLayer in the backbone.
    
    Args:
        model_path (str): Path to the YOLOv8 model file
        maxsig_config (dict): Configuration for MaxSigLayer integration
        
    Returns:
        Modified YOLOv8 model with MaxSigLayer integration
    """
    
    # Load the original model (Ultralytics)
    ul_model = YOLO(model_path)
    
    # Access the underlying PyTorch model graph
    core = getattr(ul_model, 'model', None)
    if core is None:
        raise ValueError("Ultralytics model core not found")
    
    # Build kwargs for MaxSigC2f/MaxSigLayerAdvanced
    maxsig_cfg = maxsig_config or {"kernel_sig": 3, "use_maxsig": True, "maxsig_activation": "leaky_relu"}
    maxsig_kwargs = {k: v for k, v in maxsig_cfg.items() if k in {"kernel_sig", "use_maxsig", "maxsig_activation"}}
    wrap_backbone_c2f = int(maxsig_cfg.get('wrap_backbone_c2f', 2))
    wrap_neck_c2f = int(maxsig_cfg.get('wrap_neck_c2f', 2))

    # Helper: insert MaxSig after C2f modules in backbone and neck (PAN/FPN)
    def _wrap_c2f_with_maxsig_panfpn(module: nn.Module, state: dict):
        for name, child in list(module.named_children()):
            # Update state when SPPF (end of backbone) is encountered
            if isinstance(child, SPPF):
                state['seen_sppf'] = True

            # Recurse first to maintain original module order in depth-first traversal
            _wrap_c2f_with_maxsig_panfpn(child, state)

            try:
                if isinstance(child, C2f):
                    # Determine whether this C2f is in backbone or neck based on SPPF flag
                    if not state['seen_sppf'] and state['wrapped_backbone'] < wrap_backbone_c2f:
                        target_scope = 'backbone'
                    elif state['seen_sppf'] and state['wrapped_neck'] < wrap_neck_c2f:
                        target_scope = 'neck'
                    else:
                        target_scope = None

                    if target_scope is not None:
                        # Infer output channels
                        if hasattr(child, 'cv2') and hasattr(child.cv2, 'conv'):
                            out_ch = child.cv2.conv.out_channels
                        else:
                            out_ch = getattr(child, 'c2', None) or getattr(child, 'ch', None)
                            if isinstance(out_ch, (list, tuple)):
                                out_ch = out_ch[-1]
                        if out_ch is None:
                            continue

                        maxsig = MaxSigLayernormalizedAdvanced(
                            ch_in=out_ch,
                            ch_out=out_ch,
                            kernel_sig=maxsig_kwargs.get('kernel_sig', 3),
                            stride_sig=1,
                            use_residual=True,
                            use_bn=True
                        )
                        wrapped = nn.Sequential(child, maxsig)
                        # Preserve Ultralytics graph attributes expected during forward/predict
                        for attr in ('f', 'i', 'type', 'np'):
                            if hasattr(child, attr):
                                try:
                                    setattr(wrapped, attr, getattr(child, attr))
                                except Exception:
                                    pass
                        setattr(module, name, wrapped)
                        if target_scope == 'backbone':
                            state['wrapped_backbone'] += 1
                        else:
                            state['wrapped_neck'] += 1
            except Exception:
                # If anything goes wrong, leave the child untouched
                pass

        return module

    # Start wrapping from the core model graph (DetectionModel or nn.Sequential)
    graph_root = core if isinstance(core, nn.Module) else core.model
    _wrap_c2f_with_maxsig_panfpn(graph_root, state={'seen_sppf': False, 'wrapped_backbone': 0, 'wrapped_neck': 0})
    
    return ul_model


def create_maxsig_yolov8_config(variant='n', maxsig_config=None):
    """
    Create a YOLOv8 configuration with MaxSigLayer integration.
    
    Args:
        variant (str): YOLOv8 variant ('n', 's', 'm', 'l', 'x')
        maxsig_config (dict): MaxSigLayer configuration
        
    Returns:
        dict: Model configuration
    """
    
    # Default MaxSig configuration
    default_maxsig_config = {
        'kernel_sig': 3,
        'use_maxsig': True,
        'maxsig_activation': 'leaky_relu',
        'maxsig_layers': [2, 3, 4]  # Which backbone layers to apply MaxSig to
    }
    
    if maxsig_config:
        default_maxsig_config.update(maxsig_config)
    
    # YOLOv8 configuration with MaxSig integration
    config = {
        'backbone': [
            # Initial convolution
            [-1, 1, Conv, [64, 6, 2, 2]],  # P1/2
            
            # C2f modules with MaxSig
            [-1, 1, MaxSigC2f, [128, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P2/4
            [-1, 1, Conv, [128, 3, 2]],  # P2/4
            
            [-1, 1, MaxSigC2f, [256, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P3/8
            [-1, 1, Conv, [256, 3, 2]],  # P3/8
            
            [-1, 1, MaxSigC2f, [512, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P4/16
            [-1, 1, Conv, [512, 3, 2]],  # P4/16
            
            [-1, 1, MaxSigC2f, [1024, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P5/32
            [-1, 1, Conv, [1024, 3, 2]],  # P5/32
            
            # SPPF
            [-1, 1, SPPF, [1024, 5]],
        ],
        'head': [
            # Detection head (simplified)
            [-1, 1, nn.Conv2d, [255, 1, 1]],  # Output layer
        ]
    }
    
    return config


def test_maxsig_integration():
    """Test the MaxSigLayer integration with YOLOv8."""
    print("Testing MaxSigLayer integration with YOLOv8...")
    
    # Test 1: MaxSigC2f module
    print("\n1. Testing MaxSigC2f module...")
    maxsig_c2f = MaxSigC2f(c1=64, c2=128, n=2, use_maxsig=True, kernel_sig=3)
    x = torch.randn(1, 64, 32, 32)
    out = maxsig_c2f(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test 2: YOLOv8MaxSigBackbone
    print("\n2. Testing YOLOv8MaxSigBackbone...")
    backbone = YOLOv8MaxSigBackbone(
        model_cfg={},  # Empty config for testing
        maxsig_config={'kernel_sig': 3, 'use_maxsig': True}
    )
    x = torch.randn(1, 3, 640, 640)  # Standard YOLOv8 input size
    features = backbone(x)
    print(f"   Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"   Feature {i}: {feat.shape}")
    
    # Test 3: MaxSigLayer standalone
    print("\n3. Testing MaxSigLayer standalone...")
    maxsig = MaxSigLayernormalizedAdvanced(ch_in=256, ch_out=256, kernel_sig=3, stride_sig=1)
    x = torch.randn(1, 256, 25, 25)
    out = maxsig(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n✓ All tests passed!")


def create_training_example():
    """
    Create an example of how to use the MaxSigLayer with YOLOv8 for training.
    """
    
    example_code = '''
# Example: Training YOLOv8 with MaxSigLayer integration

import torch
from ultralytics import YOLO
from yolov8_maxsig_integration import modify_yolov8_backbone, create_maxsig_yolov8_config

# Method 1: Modify existing YOLOv8 model
model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8n
modified_model = modify_yolov8_backbone(
    model_path='yolov8n.pt',
    maxsig_config={
        'kernel_sig': 3,
        'use_maxsig': True,
        'maxsig_activation': 'leaky_relu',
        'maxsig_layers': [2, 3, 4]
    }
)

# Method 2: Create new model with MaxSig integration
config = create_maxsig_yolov8_config(
    variant='n',
    maxsig_config={
        'kernel_sig': 3,
        'use_maxsig': True,
        'maxsig_activation': 'leaky_relu'
    }
)

# Training
results = modified_model.train(
    data='path/to/your/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Validation
results = modified_model.val()

# Inference
results = modified_model('path/to/image.jpg')
'''
    
    return example_code


if __name__ == "__main__":
    # Run tests
    test_maxsig_integration()
    
    # Print training example
    print("\n" + "="*60)
    print("TRAINING EXAMPLE:")
    print("="*60)
    print(create_training_example())
