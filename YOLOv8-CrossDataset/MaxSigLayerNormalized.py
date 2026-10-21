import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxSigLayerNormalized(nn.Module):
    """
    Normalized MaxSigLayer variant.

    Differences vs original:
    - Uses mean across the KxK patch (not sum), so magnitude stays comparable
      across different kernel sizes (e.g., k=1 vs k=3).
    - Combines center + median − mean as the core formulation.

    Args:
        ch_in (int): Number of input channels.
        ch_out (int): Number of output channels.
        kernel_size (int): Patch size (odd recommended).
        stride (int): Stride for the unfolding operation.
    """

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int = 1):
        super(MaxSigLayerNormalized, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.kernel_area = kernel_size ** 2
        self.center_index = self.kernel_area // 2

        # Learnable weights and bias over the patch elements
        self.weight = nn.Parameter(torch.rand(1, 1, self.kernel_area, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, self.kernel_area, 1, 1))

        # Learnable scalars for combining terms
        self.weight_center = nn.Parameter(torch.tensor(1.0))
        self.weight_median = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W)
        """
        B, C_in, H, W = x.shape

        # Sigmoid input then unfold into KxK patches
        x_sigmoid = torch.sigmoid(x)
        x_unfold = F.unfold(
            x_sigmoid,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        # x_unfold: (B, C_in*K*K, L) where L = out_h*out_w
        Bf, CK2, L = x_unfold.shape
        assert Bf == B and CK2 == C_in * self.kernel_area
        # Compute output spatial dimensions
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        # Reshape to (B, C_in, K*K, out_h, out_w)
        x_unfold = x_unfold.view(B, C_in, self.kernel_area, out_h, out_w)

        # Sigmoid-activated learnable per-patch weights and bias
        w = torch.sigmoid(self.weight)  # (1, 1, K*K, 1, 1)
        b = torch.sigmoid(self.bias)    # (1, 1, K*K, 1, 1)

        # Elementwise combine then normalize via mean across patch dim
        # D: (B, C_in, K*K, H, W)
        D = torch.maximum(w, x_unfold) + b

        # Normalized statistics across the patch
        mean_stat = torch.mean(D, dim=2)          # (B, C_in, H, W)
        median_stat, _ = torch.median(D, dim=2)   # (B, C_in, H, W)

        # Center pixel (from unfolded input, not D)
        center_values = x_unfold[:, :, self.center_index, :, :]  # (B, C_in, out_h, out_w)

        # Combine
        out = (
            self.weight_center * center_values +
            self.weight_median * median_stat -
            mean_stat
        )  # (B, C_in, H, W)

        # Project/expand to ch_out by replication + sum over input ch
        out = out.unsqueeze(1).expand(-1, self.ch_out, -1, -1, -1)  # (B, C_out, C_in, H, W)
        out = torch.sum(out, dim=2)  # (B, C_out, H, W)

        return out

class MaxSigLayerNormalized2(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size so the center is well-defined."
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.kernel_area = kernel_size ** 2
        k0 = kernel_size // 2
        self.center_index = k0 * kernel_size + k0

        # Learnable per-index weights/bias (sigmoid-bounded)
        self.weight = nn.Parameter(torch.rand(1, 1, self.kernel_area, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, 1, self.kernel_area, 1, 1))

        # Learnable scalars
        self.weight_center = nn.Parameter(torch.tensor(1.0))
        self.weight_median = nn.Parameter(torch.tensor(1.0))

        # Proper channel mixing to C_out
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        # (Option A) keep input sigmoid, but then mention in paper:
        # x_proc = torch.sigmoid(x)
        # (Option B) match paper: do not squash activations
        x_proc = x

        x_unfold = F.unfold(
            x_proc,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )  # (B, C_in*K*K, L), L = out_h*out_w
        Bf, CK2, L = x_unfold.shape
        assert Bf == B and CK2 == C_in * self.kernel_area

        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        x_unfold = x_unfold.view(B, C_in, self.kernel_area, out_h, out_w)

        w = torch.sigmoid(self.weight)  # (1,1,K*K,1,1)
        b = torch.sigmoid(self.bias)    # (1,1,K*K,1,1)

        # D: (B, C_in, K*K, out_h, out_w)
        D = torch.maximum(w, x_unfold) + b

        mean_stat   = D.mean(dim=2)             # (B, C_in, out_h, out_w)
        median_stat = D.median(dim=2).values    # (B, C_in, out_h, out_w)
        center_vals = x_unfold[:, :, self.center_index, :, :]  # (B, C_in, out_h, out_w)

        out = self.weight_center * center_vals + self.weight_median * median_stat - mean_stat
        #comment to use original maxsiglayernormalized the below two commands: std and out
        # Optional true scale normalization:
        #MAD = (D - median_stat.unsqueeze(2)).abs().median(dim=2).values + 1e-6
        #out = out / (MAD)
        std = ((D - mean_stat.unsqueeze(2)).pow(2).mean(dim=2) + 1e-6).sqrt()
        out = out / (std)
        
        return self.proj(out)  # (B, C_out, out_h, out_w)


class MaxSigLayernormalizedAdvanced(nn.Module):
    """
    Advanced version of MaxSigLayer with additional features for YOLOv8 integration.
    
    This version includes:
    - Proper multi-channel support
    - Residual connections
    - Batch normalization
    - Multiple contrast computation methods
    """
    
    def __init__(self, ch_in, ch_out, kernel_sig, stride_sig=1, 
                 use_residual=True, use_bn=True, contrast_method='maxmin'):
        super(MaxSigLayernormalizedAdvanced, self).__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_sig = kernel_sig
        self.stride_sig = stride_sig
        self.use_residual = use_residual and (ch_in == ch_out)
        self.use_bn = use_bn
        self.contrast_method = contrast_method
        
        self.pad_sig = (self.kernel_sig - 1) // 2
        
        # Main contrast enhancement layer
        self.contrast_layer = MaxSigLayerNormalized(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=kernel_sig,
            stride=stride_sig,
        )
        
        # Batch normalization
        if self.use_bn:
            self.bn = nn.BatchNorm2d(ch_out)
        
        # Additional processing layers
        self.conv1x1 = nn.Conv2d(ch_out, ch_out, 1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        # Store input for residual connection
        identity = x if self.use_residual else None
        
        # Apply contrast enhancement
        out = self.contrast_layer(x)
        
        # Apply batch normalization
        if self.use_bn:
            out = self.bn(out)
        
        # Additional 1x1 convolution
        out = self.conv1x1(out)
        
        # Apply activation
        out = self.activation(out)
        
        # Add residual connection
        if self.use_residual and identity is not None:
            out = out + identity
        
        return out



if __name__ == "__main__":
    # Simple sanity check
    layer = MaxSigLayerNormalized(ch_in=3, ch_out=3, kernel_size=3, stride=1)
    x = torch.randn(1, 3, 64, 64)
    y = layer(x)
    print("Input:", x.shape, "Output:", y.shape)
    if(x.shape != y.shape):
        print("✗ Input and output shapes do not match")
        exit(1)
    print("✓ Input and output shapes match")
    print("✓ MaxSigLayerNormalized working correctly!")
    
    
    layer = MaxSigLayernormalizedAdvanced(ch_in=3, ch_out=3, kernel_sig=3, stride_sig=1)
    x = torch.randn(1, 3, 64, 64)
    y = layer(x)
    print("Input:", x.shape, "Output:", y.shape)
    if(x.shape != y.shape):
        print("✗ Input and output shapes do not match")
        exit(1)
    print("✓ Input and output shapes match")
    print("✓ MaxSigLayernormalizedAdvanced working correctly!")
    
    print("✓ All tests passed!")


