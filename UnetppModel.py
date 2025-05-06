import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolution block: Conv2d + GroupNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
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
        # Input channels here is the number of channels after upsampling plus the number of channels from skip connection
        self.conv = ConvBlock(in_ch//2 + skip_ch, out_ch)
    
    def forward(self, x, skip):
        # Upsampling operation
        x = self.up(x)
        
        # Handle possible size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution processing
        return self.conv(x)

# Define a module for aggregating multiple feature maps
class FeatureAggregator(nn.Module):
    """Aggregate multiple feature maps into a single feature map"""
    def __init__(self, in_chs, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(sum(in_chs), out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, *features):
        sizes = [f.shape[2:] for f in features]
        min_size = [min(s[i] for s in sizes) for i in range(2)]
        
        # Resample all feature maps to the same size
        resampled = []
        for f in features:
            if f.shape[2:] != min_size:
                f = F.interpolate(f, size=min_size, mode='bilinear', align_corners=False)
            resampled.append(f)
        
        # Concatenate and fuse through 1x1 convolution
        x = torch.cat(resampled, dim=1)
        return self.relu(self.norm(self.conv(x)))

class UNetPlusPlus(nn.Module):
    """UNet++ model implementation - Fixed version"""
    def __init__(self, in_channels=3, num_classes=21, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Initial number of features
        features = [64, 128, 256, 512, 1024]
        
        # Encoder part (downsampling path)
        self.enc1 = EncoderBlock(in_channels, features[0])
        self.enc2 = EncoderBlock(features[0], features[1])
        self.enc3 = EncoderBlock(features[1], features[2])
        self.enc4 = EncoderBlock(features[2], features[3])
        
        # Center block
        self.center = ConvBlock(features[3], features[4])
        
        # First row decoders
        self.dec1_0 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_0 = DecoderBlock(features[3], features[2], features[2])
        self.dec3_0 = DecoderBlock(features[2], features[1], features[1])
        self.dec4_0 = DecoderBlock(features[1], features[0], features[0])
        
        # Feature aggregation modules
        self.agg1_1 = FeatureAggregator([features[4], features[3]], features[4])
        self.agg2_1 = FeatureAggregator([features[3], features[2]], features[3])
        self.agg3_1 = FeatureAggregator([features[2], features[1]], features[2])
        
        self.agg1_2 = FeatureAggregator([features[4], features[3], features[3]], features[4])
        self.agg2_2 = FeatureAggregator([features[3], features[2], features[2]], features[3])
        
        self.agg1_3 = FeatureAggregator([features[4], features[3], features[3], features[3]], features[4])
        
        # Second row decoders
        self.dec1_1 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_1 = DecoderBlock(features[3], features[2], features[2])
        self.dec3_1 = DecoderBlock(features[2], features[1], features[1])
        
        # Third row decoders
        self.dec1_2 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_2 = DecoderBlock(features[3], features[2], features[2])
        
        # Fourth row decoders
        self.dec1_3 = DecoderBlock(features[4], features[3], features[3])
        
        # Output layers
        self.final_conv0 = nn.Conv2d(features[0], num_classes, kernel_size=1)
        if self.deep_supervision:
            self.final_conv1 = nn.Conv2d(features[1], num_classes, kernel_size=1)
            self.final_conv2 = nn.Conv2d(features[2], num_classes, kernel_size=1)
    
    def forward(self, x):
        # Get input size for later upsampling to original resolution
        input_size = x.size()[2:]
        
        # Encoding path
        x, x0_0 = self.enc1(x)      # x0_0: first layer feature
        x, x1_0 = self.enc2(x)      # x1_0: second layer feature
        x, x2_0 = self.enc3(x)      # x2_0: third layer feature
        x, x3_0 = self.enc4(x)      # x3_0: fourth layer feature
        
        # Center encoding
        x4_0 = self.center(x)       # x4_0: deepest layer feature
        
        # Decoding path - dense nested connections
        # First row decoders
        x3_1 = self.dec1_0(x4_0, x3_0)
        x2_1 = self.dec2_0(x3_1, x2_0)
        x1_1 = self.dec3_0(x2_1, x1_0)
        x0_1 = self.dec4_0(x1_1, x0_0)
        
        # Second row decoders - using feature aggregators to handle multiple inputs
        x3_2_input = self.agg1_1(x4_0, x3_1)
        x3_2 = self.dec1_1(x3_2_input, x3_0)
        
        x2_2_input = self.agg2_1(x3_1, x2_1)
        x2_2 = self.dec2_1(x2_2_input, x2_0)
        
        x1_2_input = self.agg3_1(x2_1, x1_1)
        x1_2 = self.dec3_1(x1_2_input, x1_0)
        
        # Third row decoders
        x3_3_input = self.agg1_2(x4_0, x3_1, x3_2)
        x3_3 = self.dec1_2(x3_3_input, x3_0)
        
        x2_3_input = self.agg2_2(x3_1, x2_1, x2_2)
        x2_3 = self.dec2_2(x2_3_input, x2_0)
        
        # Fourth row decoders
        x3_4_input = self.agg1_3(x4_0, x3_1, x3_2, x3_3)
        x3_4 = self.dec1_3(x3_4_input, x3_0)
        
        # Final output (upsampling to original size)
        out0 = self.final_conv0(x0_1)
        out0 = F.interpolate(out0, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision:
            # Multi-scale deep supervision outputs
            out1 = self.final_conv1(x1_2)
            out1 = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=False)
            
            out2 = self.final_conv2(x2_3)
            out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=False)
            
            return out0, out1, out2
        
        return out0