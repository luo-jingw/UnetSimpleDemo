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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        # 处理可能的尺寸不匹配（确保上采样后的特征图与跳跃连接的尺寸一致）
        if x.shape != skip.shape:
            # 计算需要的padding
            diff_h = skip.size()[2] - x.size()[2]
            diff_w = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        
        # 拼接并通过卷积
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """UNet++模型实现"""
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
        
        # 嵌套的UNet++结构
        # 第一行解码器
        self.dec1_0 = DecoderBlock(features[4], features[3])
        self.dec2_0 = DecoderBlock(features[3], features[2])
        self.dec3_0 = DecoderBlock(features[2], features[1])
        self.dec4_0 = DecoderBlock(features[1], features[0])
        
        # 第二行解码器
        self.dec1_1 = DecoderBlock(features[3] + features[3], features[3])
        self.dec2_1 = DecoderBlock(features[2] + features[2], features[2])
        self.dec3_1 = DecoderBlock(features[1] + features[1], features[1])
        
        # 第三行解码器
        self.dec1_2 = DecoderBlock(features[3] + features[3] + features[3], features[3])
        self.dec2_2 = DecoderBlock(features[2] + features[2] + features[2], features[2])
        
        # 第四行解码器
        self.dec1_3 = DecoderBlock(features[3] + features[3] + features[3] + features[3], features[3])
        
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
        
        # 第二行解码器 - 确保特征图尺寸匹配后再拼接
        # 将x4_0上采样到x3_1的空间大小
        x4_0_upsampled = F.interpolate(x4_0, size=x3_1.shape[2:], mode='bilinear', align_corners=False)
        x3_2 = self.dec1_1(torch.cat([x4_0_upsampled, x3_1], 1), x3_0)
        
        # 将x3_1上采样到x2_1的空间大小
        x3_1_upsampled = F.interpolate(x3_1, size=x2_1.shape[2:], mode='bilinear', align_corners=False)
        x2_2 = self.dec2_1(torch.cat([x3_1_upsampled, x2_1], 1), x2_0)
        
        # 将x2_1上采样到x1_1的空间大小
        x2_1_upsampled = F.interpolate(x2_1, size=x1_1.shape[2:], mode='bilinear', align_corners=False)
        x1_2 = self.dec3_1(torch.cat([x2_1_upsampled, x1_1], 1), x1_0)
        
        # 第三行解码器
        # 确保所有特征图尺寸一致
        x4_0_upsampled = F.interpolate(x4_0, size=x3_2.shape[2:], mode='bilinear', align_corners=False)
        x3_1_upsampled = F.interpolate(x3_1, size=x3_2.shape[2:], mode='bilinear', align_corners=False)
        x3_3 = self.dec1_2(torch.cat([x4_0_upsampled, x3_1_upsampled, x3_2], 1), x3_0)
        
        x3_1_upsampled = F.interpolate(x3_1, size=x2_2.shape[2:], mode='bilinear', align_corners=False)
        x2_1_upsampled = F.interpolate(x2_1, size=x2_2.shape[2:], mode='bilinear', align_corners=False)
        x2_3 = self.dec2_2(torch.cat([x3_1_upsampled, x2_1_upsampled, x2_2], 1), x2_0)
        
        # 第四行解码器
        x4_0_upsampled = F.interpolate(x4_0, size=x3_3.shape[2:], mode='bilinear', align_corners=False)
        x3_1_upsampled = F.interpolate(x3_1, size=x3_3.shape[2:], mode='bilinear', align_corners=False)
        x3_2_upsampled = F.interpolate(x3_2, size=x3_3.shape[2:], mode='bilinear', align_corners=False)
        x3_4 = self.dec1_3(torch.cat([x4_0_upsampled, x3_1_upsampled, x3_2_upsampled, x3_3], 1), x3_0)
        
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