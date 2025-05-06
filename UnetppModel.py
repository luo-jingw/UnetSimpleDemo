import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积块: Conv2d + GroupNorm + ReLU"""
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
    """编码器块: ConvBlock + MaxPool2d"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat

class DecoderBlock(nn.Module):
    """解码器块: 上采样 + 拼接 + ConvBlock"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        # 这里的输入通道数是上采样后的通道数加上跳跃连接的通道数
        self.conv = ConvBlock(in_ch//2 + skip_ch, out_ch)
    
    def forward(self, x, skip):
        # 上采样操作
        x = self.up(x)
        
        # 处理可能的尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        
        # 卷积处理
        return self.conv(x)

# 定义一个用于聚合多个特征图的模块
class FeatureAggregator(nn.Module):
    """将多个特征图聚合为单一特征图"""
    def __init__(self, in_chs, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(sum(in_chs), out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, *features):
        sizes = [f.shape[2:] for f in features]
        min_size = [min(s[i] for s in sizes) for i in range(2)]
        
        # 将所有特征图重采样到相同尺寸
        resampled = []
        for f in features:
            if f.shape[2:] != min_size:
                f = F.interpolate(f, size=min_size, mode='bilinear', align_corners=False)
            resampled.append(f)
        
        # 拼接并通过1x1卷积融合
        x = torch.cat(resampled, dim=1)
        return self.relu(self.norm(self.conv(x)))

class UNetPlusPlus(nn.Module):
    """UNet++模型实现 - 修复版本"""
    def __init__(self, in_channels=3, num_classes=21, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 初始特征数量
        features = [64, 128, 256, 512, 1024]
        
        # 编码器部分（下采样路径）
        self.enc1 = EncoderBlock(in_channels, features[0])
        self.enc2 = EncoderBlock(features[0], features[1])
        self.enc3 = EncoderBlock(features[1], features[2])
        self.enc4 = EncoderBlock(features[2], features[3])
        
        # 中心块
        self.center = ConvBlock(features[3], features[4])
        
        # 第一行解码器
        self.dec1_0 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_0 = DecoderBlock(features[3], features[2], features[2])
        self.dec3_0 = DecoderBlock(features[2], features[1], features[1])
        self.dec4_0 = DecoderBlock(features[1], features[0], features[0])
        
        # 特征聚合模块
        self.agg1_1 = FeatureAggregator([features[4], features[3]], features[4])
        self.agg2_1 = FeatureAggregator([features[3], features[2]], features[3])
        self.agg3_1 = FeatureAggregator([features[2], features[1]], features[2])
        
        self.agg1_2 = FeatureAggregator([features[4], features[3], features[3]], features[4])
        self.agg2_2 = FeatureAggregator([features[3], features[2], features[2]], features[3])
        
        self.agg1_3 = FeatureAggregator([features[4], features[3], features[3], features[3]], features[4])
        
        # 第二行解码器
        self.dec1_1 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_1 = DecoderBlock(features[3], features[2], features[2])
        self.dec3_1 = DecoderBlock(features[2], features[1], features[1])
        
        # 第三行解码器
        self.dec1_2 = DecoderBlock(features[4], features[3], features[3])
        self.dec2_2 = DecoderBlock(features[3], features[2], features[2])
        
        # 第四行解码器
        self.dec1_3 = DecoderBlock(features[4], features[3], features[3])
        
        # 输出层
        self.final_conv0 = nn.Conv2d(features[0], num_classes, kernel_size=1)
        if self.deep_supervision:
            self.final_conv1 = nn.Conv2d(features[1], num_classes, kernel_size=1)
            self.final_conv2 = nn.Conv2d(features[2], num_classes, kernel_size=1)
    
    def forward(self, x):
        # 获取输入尺寸，用于后续上采样到原始分辨率
        input_size = x.size()[2:]
        
        # 编码路径
        x, x0_0 = self.enc1(x)      # x0_0: 首层特征
        x, x1_0 = self.enc2(x)      # x1_0: 第二层特征
        x, x2_0 = self.enc3(x)      # x2_0: 第三层特征
        x, x3_0 = self.enc4(x)      # x3_0: 第四层特征
        
        # 中心编码
        x4_0 = self.center(x)       # x4_0: 最深层特征
        
        # 解码路径 - 密集嵌套连接
        # 第一行解码器
        x3_1 = self.dec1_0(x4_0, x3_0)
        x2_1 = self.dec2_0(x3_1, x2_0)
        x1_1 = self.dec3_0(x2_1, x1_0)
        x0_1 = self.dec4_0(x1_1, x0_0)
        
        # 第二行解码器 - 使用特征聚合器处理多个输入
        x3_2_input = self.agg1_1(x4_0, x3_1)
        x3_2 = self.dec1_1(x3_2_input, x3_0)
        
        x2_2_input = self.agg2_1(x3_1, x2_1)
        x2_2 = self.dec2_1(x2_2_input, x2_0)
        
        x1_2_input = self.agg3_1(x2_1, x1_1)
        x1_2 = self.dec3_1(x1_2_input, x1_0)
        
        # 第三行解码器
        x3_3_input = self.agg1_2(x4_0, x3_1, x3_2)
        x3_3 = self.dec1_2(x3_3_input, x3_0)
        
        x2_3_input = self.agg2_2(x3_1, x2_1, x2_2)
        x2_3 = self.dec2_2(x2_3_input, x2_0)
        
        # 第四行解码器
        x3_4_input = self.agg1_3(x4_0, x3_1, x3_2, x3_3)
        x3_4 = self.dec1_3(x3_4_input, x3_0)
        
        # 最终输出（上采样到原始尺寸）
        out0 = self.final_conv0(x0_1)
        out0 = F.interpolate(out0, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision:
            # 多尺度深度监督输出
            out1 = self.final_conv1(x1_2)
            out1 = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=False)
            
            out2 = self.final_conv2(x2_3)
            out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=False)
            
            return out0, out1, out2
        
        return out0