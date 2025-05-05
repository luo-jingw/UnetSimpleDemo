import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import cv2  
import matplotlib.pyplot as plt

# VOC数据集标签颜色映射（21类，包括背景）
VOC_COLORMAP = [
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # diningtable
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # pottedplant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128]      # tvmonitor
]

# VOC标签名称
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOC2012Dataset(Dataset):
    """
    VOC2012数据集加载类，移除数据增强以确保图像和掩码正确对齐
    """
    
    def __init__(self, root='voc_data', split='train', transform=None, target_transform=None, 
                 img_size=512):
        """
        初始化VOC2012数据集
        
        参数:
            root: 数据集根目录
            split: 'train', 'val' 或 'trainval'
            transform: 图像变换函数
            target_transform: 标签变换函数
            img_size: 输出图像尺寸
        """
        # 设置图像变换
        if transform is None:
            # 标准变换，无数据增强
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  # 使用固定尺寸确保一致
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transform
            
        # 设置掩码变换
        if target_transform is None:
            # 掩码变换需与图像变换保持一致
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.PILToTensor(),
            ])
        else:
            self.mask_transform = target_transform
            
        # 初始化基础VOCSegmentation数据集
        self.dataset = VOCSegmentation(
            root=root, 
            year='2012',
            image_set=split,
            download=False,
            transform=self.img_transform,
            target_transform=self.mask_transform
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img, mask = self.dataset[idx]
        
        # 处理掩码：VOC数据集中255表示忽略区域，我们可以选择处理它
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0)  # 移除通道维度 [1, H, W] -> [H, W]
            # 可选：将255（忽略区域）设为0（背景）
            # mask[mask == 255] = 0
        
        return img, mask
    
    @staticmethod
    def decode_segmap(mask):
        """
        将分割掩码转换为RGB彩色图像
        mask: [H, W] 的张量，值为0-20的类别索引
        返回: [H, W, 3] 的numpy数组，RGB彩色图
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for cls in range(21):
            rgb[mask == cls] = VOC_COLORMAP[cls]
                
        return rgb

    @staticmethod
    def get_class_names(mask):
        """
        获取掩码中存在的类别名称
        mask: [H, W] 的张量或numpy数组，值为0-20的类别索引
        返回: 存在的类别名称列表
        """
        if isinstance(mask, torch.Tensor):
            unique_classes = torch.unique(mask).numpy()
        else:
            unique_classes = np.unique(mask)
            
        return [VOC_CLASSES[cls] for cls in unique_classes if cls < 21]

# 使用示例
def create_voc_dataloaders(root='voc_data', batch_size=8, img_size=512, num_workers=2):
    """
    创建VOC2012数据集的训练和验证数据加载器
    
    参数:
        root: 数据集根目录
        batch_size: 批量大小
        img_size: 图像尺寸
        num_workers: 数据加载线程数
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 创建训练数据集（不使用数据增强）
    train_ds = VOC2012Dataset(
        root=root,
        split='train',
        img_size=img_size
    )
    
    # 创建验证数据集
    val_ds = VOC2012Dataset(
        root=root,
        split='val',
        img_size=img_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 测试代码
if __name__ == "__main__":
    
    # 创建数据集
    train_ds = VOC2012Dataset(split='train', img_size=512)
    val_ds = VOC2012Dataset(split='val', img_size=512)
    
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")
    
    # 可视化几个样本
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(3):
        img, mask = val_ds[i]
        
        # 转换图像格式用于显示
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # 转换掩码为彩色图
        color_mask = VOC2012Dataset.decode_segmap(mask)
        
        # 创建叠加效果
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)
        
        # 显示图像
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        # 显示彩色掩码
        axes[i, 1].imshow(color_mask)
        axes[i, 1].set_title("Segmentation Mask")
        axes[i, 1].axis('off')
        
        # 显示叠加效果
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
        
        # 获取并显示类别名称
        class_names = VOC2012Dataset.get_class_names(mask)
        print(f"样本 {i+1} 类别: {', '.join(class_names)}")
    
    plt.tight_layout()
    plt.savefig('aligned_samples.png')
    plt.show()