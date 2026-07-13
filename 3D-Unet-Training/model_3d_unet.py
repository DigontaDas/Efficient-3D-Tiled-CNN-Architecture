import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle different spatial dimensions
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net Architecture for Medical Image Segmentation
    
    Architecture Details:
    - Encoder: 4 downsampling stages with max pooling
    - Decoder: 4 upsampling stages with skip connections
    - Base filters: 32 (can be adjusted)
    - Kernel size: 3x3x3 for all convolutions
    - Activation: ReLU
    - Normalization: Batch Normalization
    """
    
    def __init__(self, n_channels=1, n_classes=1, base_filters=32, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.bilinear = bilinear

        # Encoder (downsampling path)
        self.inc = DoubleConv3D(n_channels, base_filters)
        self.down1 = Down3D(base_filters, base_filters * 2)
        self.down2 = Down3D(base_filters * 2, base_filters * 4)
        self.down3 = Down3D(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up3D(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up3D(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up3D(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up3D(base_filters * 2, base_filters, bilinear)
        
        # Output layer
        self.outc = OutConv3D(base_filters, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 32 channels
        x2 = self.down1(x1)   # 64 channels
        x3 = self.down2(x2)   # 128 channels
        x4 = self.down3(x3)   # 256 channels
        x5 = self.down4(x4)   # 512 channels (or 256 if bilinear)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)  # 256 channels
        x = self.up2(x, x3)   # 128 channels
        x = self.up3(x, x2)   # 64 channels
        x = self.up4(x, x1)   # 32 channels
        
        # Output
        logits = self.outc(x) # 1 channel for binary segmentation
        
        return logits

def get_unet3d_model(n_channels=1, n_classes=1, base_filters=32, bilinear=True):
    """
    Factory function to create 3D U-Net model
    
    Args:
        n_channels (int): Number of input channels (1 for grayscale medical images)
        n_classes (int): Number of output classes (1 for binary segmentation)
        base_filters (int): Number of base filters (controls model size)
        bilinear (bool): Whether to use bilinear upsampling
    
    Returns:
        UNet3D: 3D U-Net model
    """
    model = UNet3D(n_channels, n_classes, base_filters, bilinear)
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(weights_init)
    return model

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing 3D U-Net on device: {device}")
    
    # Create model
    model = get_unet3d_model(n_channels=1, n_classes=1, base_filters=32)
    model = model.to(device)
    
    # Test with dummy data (batch_size=2, channels=1, depth=64, height=64, width=64)
    test_input = torch.randn(2, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("3D U-Net test successful!")
