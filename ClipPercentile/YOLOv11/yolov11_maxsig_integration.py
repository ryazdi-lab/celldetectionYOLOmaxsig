"""
YOLOv11 Integration with MaxSigLayerNormalized for C2fPSA and C3k2 Modules

This module provides integration of MaxSigLayerNormalized into YOLOv11's advanced
modules (C2fPSA and C3k2) for enhanced contrast and attention mechanisms.
Optimized for cell detection tasks (WBC, RBC, Platelets).

Author: Reza Yazdi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, C2PSA, C3k2
from ultralytics.nn.tasks import DetectionModel
from MaxSigLayerNormalized import MaxSigLayernormalizedAdvanced


class GraphCompatWrapper(nn.Module):
    """
    Ultralytics graph-compatible wrapper that preserves expected node attributes (e.g., f).
    Wraps an existing layer and applies an extra module (e.g., MaxSig) to its output.
    """
    def __init__(self, layer: nn.Module, extra: nn.Module):
        super().__init__()
        self.layer = layer
        self.extra = extra
        # Preserve attributes Ultralytics graph expects on nodes
        for attr in ("f", "i", "type", "np", "receptive"):
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

    def forward(self, x):
        y = self.layer(x)
        return self.extra(y)

class MaxSigC2fPSA(nn.Module):
    """
    C2fPSA module with integrated MaxSigLayerNormalized for enhanced spatial attention.
    
    This combines YOLOv11's C2fPSA (C2f with Position-Spatial Attention) with 
    MaxSigLayerNormalized to provide complementary contrast enhancement.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, 
                 kernel_sig=3, use_maxsig=True, maxsig_activation='leaky_relu',
                 psa_ratio=4, psa_use_conv=True):
        super(MaxSigC2fPSA, self).__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e
        self.use_maxsig = use_maxsig
        self.psa_ratio = psa_ratio
        self.psa_use_conv = psa_use_conv
        
        # Standard C2f components
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        # Position-Spatial Attention (PSA) mechanism
        self.psa = PositionSpatialAttention(
            channels=self.c, 
            ratio=psa_ratio, 
            use_conv=psa_use_conv
        )
        
        # MaxSigLayer for contrast enhancement
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
        """Forward pass with PSA attention and optional MaxSig enhancement."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # Apply PSA attention to the last feature map
        y[-1] = self.psa(y[-1])
        
        # Apply MaxSigLayer for additional contrast enhancement
        if self.use_maxsig and self.maxsig is not None:
            y[-1] = self.maxsig(y[-1])
        
        return self.cv2(torch.cat(y, 1))


class MaxSigC3k2(nn.Module):
    """
    C3k2 module with integrated MaxSigLayerNormalized for enhanced feature extraction.
    
    This combines YOLOv11's C3k2 (C3 with k=2 kernel) with MaxSigLayerNormalized
    to provide efficient feature extraction with contrast enhancement.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 kernel_sig=3, use_maxsig=True, maxsig_activation='leaky_relu'):
        super(MaxSigC3k2, self).__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e
        self.use_maxsig = use_maxsig
        
        # C3k2 components with k=2 kernel
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        # MaxSigLayer for contrast enhancement
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
        """Forward pass with optional MaxSig enhancement."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # Apply MaxSigLayer for contrast enhancement
        if self.use_maxsig and self.maxsig is not None:
            y[-1] = self.maxsig(y[-1])
        
        return self.cv2(torch.cat(y, 1))


class PositionSpatialAttention(nn.Module):
    """
    Position-Spatial Attention mechanism for YOLOv11 C2fPSA.
    
    This module provides spatial attention by focusing on important regions
    in the feature maps, complementing MaxSigLayer's contrast enhancement.
    """
    
    def __init__(self, channels, ratio=4, use_conv=True):
        super(PositionSpatialAttention, self).__init__()
        
        self.channels = channels
        self.ratio = ratio
        self.use_conv = use_conv
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Additional processing
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through position-spatial attention."""
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        
        # Apply channel attention
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        
        # Apply spatial attention
        x = x * spatial_att
        
        # Additional processing
        if self.use_conv:
            x = self.conv(x)
        
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck module used in C2fPSA and C3k2."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class YOLOv11MaxSigBackbone(nn.Module):
    """
    Modified YOLOv11 backbone with MaxSigLayerNormalized integration.
    
    This backbone uses MaxSigC2fPSA and MaxSigC3k2 modules to enhance
    contrast and attention mechanisms in YOLOv11's architecture.
    """
    
    def __init__(self, model_cfg, maxsig_config=None):
        super(YOLOv11MaxSigBackbone, self).__init__()
        
        # Default MaxSig configuration
        self.maxsig_config = maxsig_config or {
            'kernel_sig': 3,
            'use_maxsig': True,
            'maxsig_activation': 'leaky_relu',
            'maxsig_layers': [2, 3, 4],  # Which backbone layers to apply MaxSig to
            'psa_ratio': 4,
            'psa_use_conv': True
        }
        
        # Parse model configuration
        self.parse_model_cfg(model_cfg)
        
    def parse_model_cfg(self, cfg):
        """Parse YOLOv11 model configuration and create backbone."""
        # YOLOv11n backbone layers with MaxSig integration
        self.backbone_layers = nn.ModuleList()
        
        # Filter supported kwargs for MaxSig modules
        maxsig_kwargs = {
            k: v for k, v in self.maxsig_config.items()
            if k in {"kernel_sig", "use_maxsig", "maxsig_activation"}
        }
        
        # PSA-specific kwargs for C2fPSA modules
        psa_kwargs = {
            k: v for k, v in self.maxsig_config.items()
            if k in {"psa_ratio", "psa_use_conv"}
        }
        
        # Initial convolution
        self.backbone_layers.append(Conv(3, 64, 6, 2, 2))  # P1/2
        
        # C2fPSA modules with MaxSig integration
        c2fpsa_kwargs = {**maxsig_kwargs, **psa_kwargs}
        self.backbone_layers.append(MaxSigC2fPSA(64, 128, 2, True, 1, 0.5, 
                                                **c2fpsa_kwargs))  # P2/4
        self.backbone_layers.append(Conv(128, 128, 3, 2))  # P2/4
        
        self.backbone_layers.append(MaxSigC2fPSA(128, 256, 2, True, 1, 0.5,
                                                **c2fpsa_kwargs))  # P3/8
        self.backbone_layers.append(Conv(256, 256, 3, 2))  # P3/8
        
        # C3k2 modules with MaxSig integration
        self.backbone_layers.append(MaxSigC3k2(256, 512, 2, True, 1, 0.5,
                                              **maxsig_kwargs))  # P4/16
        self.backbone_layers.append(Conv(512, 512, 3, 2))  # P4/16
        
        self.backbone_layers.append(MaxSigC3k2(512, 1024, 2, True, 1, 0.5,
                                              **maxsig_kwargs))  # P5/32
        self.backbone_layers.append(Conv(1024, 1024, 3, 2))  # P5/32
        
        # SPPF
        self.backbone_layers.append(SPPF(1024, 1024, 5))
        
    def forward(self, x):
        """Forward pass through the backbone."""
        features = []
        
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            
            # Collect features at different scales (typical for YOLOv11)
            if i in [2, 4, 6, 8]:  # After C2fPSA/C3k2 modules
                features.append(x)
        
        return features


def modify_yolov11_backbone(model_path, maxsig_config=None):
    """
    Modify an existing YOLOv11 model to include MaxSigLayerNormalized in C2fPSA and C3k2 modules.
    
    Args:
        model_path (str): Path to the YOLOv11 model file
        maxsig_config (dict): Configuration for MaxSigLayer integration
        
    Returns:
        Modified YOLOv11 model with MaxSigLayer integration
    """
    
    # Load the original model (Ultralytics)
    ul_model = YOLO(model_path)
    
    # Access the underlying PyTorch model graph
    core = getattr(ul_model, 'model', None)
    if core is None:
        raise ValueError("Ultralytics model core not found")
    
    # Build kwargs for MaxSig modules
    maxsig_cfg = maxsig_config or {
        "kernel_sig": 3, 
        "use_maxsig": True, 
        "maxsig_activation": "leaky_relu",
        "psa_ratio": 4,
        "psa_use_conv": True
    }
    
    # Helper: insert MaxSig after C2fPSA and C3k2 modules
    def _wrap_yolov11_modules_with_maxsig(module: nn.Module, state: dict):
        def _infer_out_channels(layer: nn.Module):
            # Try common patterns first
            out_ch = getattr(layer, 'c2', None) or getattr(layer, 'ch', None)
            if isinstance(out_ch, (list, tuple)):
                out_ch = out_ch[-1]
            # Fallback: look for cv2.conv.out_channels
            if out_ch is None:
                try:
                    cv2 = getattr(layer, 'cv2', None)
                    conv = getattr(cv2, 'conv', None)
                    out_ch = getattr(conv, 'out_channels', None)
                except Exception:
                    out_ch = None
            return out_ch
        for name, child in list(module.named_children()):
            # Update state when SPPF (end of backbone) is encountered
            if isinstance(child, SPPF):
                state['seen_sppf'] = True

            # Recurse first to maintain original module order
            _wrap_yolov11_modules_with_maxsig(child, state)

            try:
                # Check for C2PSA modules (YOLOv11 specific)
                if isinstance(child, C2PSA):
                    if not state['seen_sppf'] and state['wrapped_backbone'] < maxsig_cfg.get('wrap_backbone_c2fpsa', 2):
                        # This is a C2PSA module in backbone
                        out_ch = _infer_out_channels(child)
                        if out_ch is None:
                            continue

                        maxsig = MaxSigLayernormalizedAdvanced(
                            ch_in=out_ch,
                            ch_out=out_ch,
                            kernel_sig=maxsig_cfg.get('kernel_sig', 3),
                            stride_sig=1,
                            use_residual=True,
                            use_bn=True
                        )
                        setattr(module, name, GraphCompatWrapper(child, maxsig))
                        state['wrapped_backbone'] += 1
                
                # Check for C3k2 modules (YOLOv11 specific)
                elif isinstance(child, C3k2):
                    if not state['seen_sppf'] and state['wrapped_backbone_c3k2'] < maxsig_cfg.get('wrap_backbone_c3k2', 2):
                        # This is a C3k2 module in backbone
                        out_ch = _infer_out_channels(child)
                        if out_ch is None:
                            continue

                        maxsig = MaxSigLayernormalizedAdvanced(
                            ch_in=out_ch,
                            ch_out=out_ch,
                            kernel_sig=maxsig_cfg.get('kernel_sig', 3),
                            stride_sig=1,
                            use_residual=True,
                            use_bn=True
                        )
                        setattr(module, name, GraphCompatWrapper(child, maxsig))
                        state['wrapped_backbone_c3k2'] += 1
                
                # Check for standard C2f modules in neck
                elif isinstance(child, C2f):
                    if state['seen_sppf'] and state['wrapped_neck'] < maxsig_cfg.get('wrap_neck_c2f', 2):
                        out_ch = _infer_out_channels(child)
                        if out_ch is None:
                            continue

                        maxsig = MaxSigLayernormalizedAdvanced(
                            ch_in=out_ch,
                            ch_out=out_ch,
                            kernel_sig=maxsig_cfg.get('kernel_sig', 3),
                            stride_sig=1,
                            use_residual=True,
                            use_bn=True
                        )
                        setattr(module, name, GraphCompatWrapper(child, maxsig))
                        state['wrapped_neck'] += 1
                        
            except Exception as e:
                # If anything goes wrong, leave the child untouched
                print(f"Warning: Could not wrap module {name}: {e}")
                pass

        return module

    # Start wrapping from the core model graph
    graph_root = core if isinstance(core, nn.Module) else core.model
    _wrap_yolov11_modules_with_maxsig(graph_root, state={
        'seen_sppf': False, 
        'wrapped_backbone': 0, 
        'wrapped_backbone_c3k2': 0,
        'wrapped_neck': 0
    })
    
    return ul_model


def create_maxsig_yolov11_config(variant='yolo11n', maxsig_config=None):
    """
    Create a YOLOv11 configuration with MaxSigLayer integration.
    
    Args:
        variant (str): YOLOv11 variant ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
        maxsig_config (dict): MaxSigLayer configuration
        
    Returns:
        dict: Model configuration
    """
    
    # Default MaxSig configuration
    default_maxsig_config = {
        'kernel_sig': 3,
        'use_maxsig': True,
        'maxsig_activation': 'leaky_relu',
        'maxsig_layers': [2, 3, 4],  # Which backbone layers to apply MaxSig to
        'psa_ratio': 4,
        'psa_use_conv': True
    }
    
    if maxsig_config:
        default_maxsig_config.update(maxsig_config)
    
    # YOLOv11 configuration with MaxSig integration
    config = {
        'backbone': [
            # Initial convolution
            [-1, 1, Conv, [64, 6, 2, 2]],  # P1/2
            
            # C2fPSA modules with MaxSig
            [-1, 1, MaxSigC2fPSA, [128, 2, True, 1, 0.5, 3, True, 'leaky_relu', 4, True]],  # P2/4
            [-1, 1, Conv, [128, 3, 2]],  # P2/4
            
            [-1, 1, MaxSigC2fPSA, [256, 2, True, 1, 0.5, 3, True, 'leaky_relu', 4, True]],  # P3/8
            [-1, 1, Conv, [256, 3, 2]],  # P3/8
            
            # C3k2 modules with MaxSig
            [-1, 1, MaxSigC3k2, [512, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P4/16
            [-1, 1, Conv, [512, 3, 2]],  # P4/16
            
            [-1, 1, MaxSigC3k2, [1024, 2, True, 1, 0.5, 3, True, 'leaky_relu']],  # P5/32
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


def test_yolov11_maxsig_integration():
    """Test the MaxSigLayer integration with YOLOv11."""
    print("Testing MaxSigLayer integration with YOLOv11...")
    
    # Test 1: MaxSigC2fPSA module
    print("\n1. Testing MaxSigC2fPSA module...")
    maxsig_c2fpsa = MaxSigC2fPSA(c1=64, c2=128, n=2, use_maxsig=True, kernel_sig=3)
    x = torch.randn(1, 64, 32, 32)
    out = maxsig_c2fpsa(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test 2: MaxSigC3k2 module
    print("\n2. Testing MaxSigC3k2 module...")
    maxsig_c3k2 = MaxSigC3k2(c1=256, c2=512, n=2, use_maxsig=True, kernel_sig=3)
    x = torch.randn(1, 256, 16, 16)
    out = maxsig_c3k2(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test 3: PositionSpatialAttention module
    print("\n3. Testing PositionSpatialAttention module...")
    psa = PositionSpatialAttention(channels=256, ratio=4, use_conv=True)
    x = torch.randn(1, 256, 16, 16)
    out = psa(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test 4: YOLOv11MaxSigBackbone
    print("\n4. Testing YOLOv11MaxSigBackbone...")
    backbone = YOLOv11MaxSigBackbone(
        model_cfg={},  # Empty config for testing
        maxsig_config={'kernel_sig': 3, 'use_maxsig': True, 'psa_ratio': 4}
    )
    x = torch.randn(1, 3, 640, 640)  # Standard YOLOv11 input size
    features = backbone(x)
    print(f"   Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"   Feature {i}: {feat.shape}")
    
    print("\n✓ All YOLOv11 MaxSig tests passed!")


if __name__ == "__main__":
    # Run tests
    test_yolov11_maxsig_integration()
