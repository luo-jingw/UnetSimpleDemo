import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolution block: Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool2d"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat

class DecoderBlock(nn.Module):
    """Decoder block: Upsampling + Concatenation + ConvBlock"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch//2 + skip_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle possible size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution processing
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """
    Standard UNet++ architecture
    
    References:
    - UNet++: A Nested U-Net Architecture for Medical Image Segmentation
      (Zhou et al., 2018)
    """
    def __init__(self, in_channels=3, num_classes=21, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Initial number of features
        filters = [32, 64, 128, 256, 512]
        
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, filters[0])
        self.encoder2 = EncoderBlock(filters[0], filters[1])
        self.encoder3 = EncoderBlock(filters[1], filters[2])
        self.encoder4 = EncoderBlock(filters[2], filters[3])
        
        # Center
        self.center = ConvBlock(filters[3], filters[4])
        
        # Nested Decoder path - First layer (L = 0)
        self.decoder0_1 = DecoderBlock(filters[4], filters[3], filters[3])  # from center to level 4
        self.decoder1_1 = DecoderBlock(filters[3], filters[2], filters[2])  # from level 4 to level 3
        self.decoder2_1 = DecoderBlock(filters[2], filters[1], filters[1])  # from level 3 to level 2
        self.decoder3_1 = DecoderBlock(filters[1], filters[0], filters[0])  # from level 2 to level 1
        
        # Nested Decoder path - Second layer (L = 1)
        self.decoder0_2 = DecoderBlock(filters[4] + filters[3], filters[3], filters[3])  # from (center + decoder0_1) to level 4
        self.decoder1_2 = DecoderBlock(filters[3] + filters[2], filters[2], filters[2])  # from (decoder0_1 + decoder1_1) to level 3
        self.decoder2_2 = DecoderBlock(filters[2] + filters[1], filters[1], filters[1])  # from (decoder1_1 + decoder2_1) to level 2
        
        # Nested Decoder path - Third layer (L = 2)
        self.decoder0_3 = DecoderBlock(filters[4] + filters[3] * 2, filters[3], filters[3])  # from (center + decoder0_1 + decoder0_2) to level 4
        self.decoder1_3 = DecoderBlock(filters[3] + filters[2] * 2, filters[2], filters[2])  # from (decoder0_2 + decoder1_1 + decoder1_2) to level 3
        
        # Nested Decoder path - Fourth layer (L = 3)
        self.decoder0_4 = DecoderBlock(filters[4] + filters[3] * 3, filters[3], filters[3])  # from (center + decoder0_1 + decoder0_2 + decoder0_3) to level 4
        
        # Output layers for deep supervision
        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)  # Output from decoder3_1
        if self.deep_supervision:
            self.final2 = nn.Conv2d(filters[1], num_classes, kernel_size=1)  # Output from decoder2_2
            self.final3 = nn.Conv2d(filters[2], num_classes, kernel_size=1)  # Output from decoder1_3
            self.final4 = nn.Conv2d(filters[3], num_classes, kernel_size=1)  # Output from decoder0_4
            
            # Learnable fusion weights
            self.weight_params = nn.Parameter(torch.ones(4) / 4)  # Initialize with equal weights
    
    def forward(self, x):
        # Store input size for later upsampling
        input_size = x.size()[2:]
        
        # Encoder Path
        x, x1_0 = self.encoder1(x)         # Level 1 features
        x, x2_0 = self.encoder2(x)         # Level 2 features
        x, x3_0 = self.encoder3(x)         # Level 3 features
        x, x4_0 = self.encoder4(x)         # Level 4 features
        
        # Center
        x5_0 = self.center(x)              # Bottleneck features
        
        # First layer of nested decoder - L=0
        x4_1 = self.decoder0_1(x5_0, x4_0)  # Up from center to level 4
        x3_1 = self.decoder1_1(x4_1, x3_0)  # Up from level 4 to level 3
        x2_1 = self.decoder2_1(x3_1, x2_0)  # Up from level 3 to level 2
        x1_1 = self.decoder3_1(x2_1, x1_0)  # Up from level 2 to level 1
        
        # Second layer of nested decoder - L=1
        # 确保要拼接的特征图具有相同的空间尺寸
        if x5_0.shape[2:] != x4_1.shape[2:]:
            x5_0_resized = F.interpolate(x5_0, size=x4_1.shape[2:], mode='bilinear', align_corners=False)
        else:
            x5_0_resized = x5_0
        x4_2 = self.decoder0_2(torch.cat([x5_0_resized, x4_1], dim=1), x4_0)
        
        if x4_1.shape[2:] != x3_1.shape[2:]:
            x4_1_resized = F.interpolate(x4_1, size=x3_1.shape[2:], mode='bilinear', align_corners=False)
        else:
            x4_1_resized = x4_1
        x3_2 = self.decoder1_2(torch.cat([x4_1_resized, x3_1], dim=1), x3_0)
        
        if x3_1.shape[2:] != x2_1.shape[2:]:
            x3_1_resized = F.interpolate(x3_1, size=x2_1.shape[2:], mode='bilinear', align_corners=False)
        else:
            x3_1_resized = x3_1
        x2_2 = self.decoder2_2(torch.cat([x3_1_resized, x2_1], dim=1), x2_0)
        
        # Third layer of nested decoder - L=2
        # 确保要拼接的所有特征图具有相同的空间尺寸
        cat_size = x4_2.shape[2:]
        x5_0_resized = F.interpolate(x5_0, size=cat_size, mode='bilinear', align_corners=False)
        x4_1_resized = F.interpolate(x4_1, size=cat_size, mode='bilinear', align_corners=False)
        x4_3 = self.decoder0_3(torch.cat([x5_0_resized, x4_1_resized, x4_2], dim=1), x4_0)
        
        cat_size = x3_2.shape[2:]
        x4_2_resized = F.interpolate(x4_2, size=cat_size, mode='bilinear', align_corners=False)
        x3_1_resized = F.interpolate(x3_1, size=cat_size, mode='bilinear', align_corners=False)
        x3_3 = self.decoder1_3(torch.cat([x4_2_resized, x3_1_resized, x3_2], dim=1), x3_0)
        
        # Fourth layer of nested decoder - L=3
        cat_size = x4_3.shape[2:]
        x5_0_resized = F.interpolate(x5_0, size=cat_size, mode='bilinear', align_corners=False)
        x4_1_resized = F.interpolate(x4_1, size=cat_size, mode='bilinear', align_corners=False)
        x4_2_resized = F.interpolate(x4_2, size=cat_size, mode='bilinear', align_corners=False)
        x4_4 = self.decoder0_4(torch.cat([x5_0_resized, x4_1_resized, x4_2_resized, x4_3], dim=1), x4_0)
        
        # Final outputs with deep supervision
        out1 = self.final1(x1_1)
        out1 = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision:
            # Get outputs from all levels
            out2 = self.final2(x2_2)
            out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=False)
            
            out3 = self.final3(x3_3)
            out3 = F.interpolate(out3, size=input_size, mode='bilinear', align_corners=False)
            
            out4 = self.final4(x4_4)
            out4 = F.interpolate(out4, size=input_size, mode='bilinear', align_corners=False)
            
            # Apply softmax to ensure weights sum to 1
            weights = F.softmax(self.weight_params, dim=0)
            
            # Linear combination of all outputs
            final_output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3 + weights[3] * out4
            
            return final_output
        
        return out1

if __name__ == "__main__":
    # Example usage
    model = UNetPlusPlus(in_channels=3, num_classes=21, deep_supervision=True)
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
