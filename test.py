# voc_dataset_visualization.py
# ---------------------------------------------------------
# VOC数据集加载与可视化示例
# ---------------------------------------------------------
import os, random, torch, numpy as np
from PIL import Image
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2  # 添加OpenCV库导入

# ------------------------------ Utils ------------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# VOC数据集标签颜色映射（21类，包括背景）
VOC_COLORMAP = [
    [0, 0, 0],        # 背景
    [128, 0, 0],      # 飞机
    [0, 128, 0],      # 自行车
    [128, 128, 0],    # 鸟
    [0, 0, 128],      # 船
    [128, 0, 128],    # 瓶子
    [0, 128, 128],    # 公共汽车
    [128, 128, 128],  # 汽车
    [64, 0, 0],       # 猫
    [192, 0, 0],      # 椅子
    [64, 128, 0],     # 牛
    [192, 128, 0],    # 餐桌
    [64, 0, 128],     # 狗
    [192, 0, 128],    # 马
    [64, 128, 128],   # 摩托车
    [192, 128, 128],  # 人
    [0, 64, 0],       # 盆栽植物
    [128, 64, 0],     # 羊
    [0, 192, 0],      # 沙发
    [128, 192, 0],    # 火车
    [0, 64, 128]      # 电视/显示器
]

# VOC标签名称
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 将分割掩码转换为RGB彩色图像
def decode_segmap(mask):
    """
    将分割掩码转换为RGB彩色图像
    mask: [H, W] 的张量，值为0-20的类别索引
    返回: [H, W, 3] 的numpy数组，RGB彩色图
    """
    mask = mask.numpy()
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(21):
        rgb[mask == cls] = VOC_COLORMAP[cls]
            
    return rgb

# ------------------------------ Main ------------------------------
def main():
    seed_all()
    root = 'voc_data'
    batch_size = 4
    num_samples = 5  # 可视化的样本数量
    
    # 简化的数据转换
    val_img_tf = transforms.Compose([
        transforms.Resize(512), 
        transforms.CenterCrop(512), 
        transforms.ToTensor()
    ])
    val_msk_tf = transforms.Compose([
        transforms.Resize(512, interpolation=Image.NEAREST),
        transforms.CenterCrop(512), 
        transforms.PILToTensor()
    ])

    # 加载数据集
    print("正在加载VOC数据集...")
    val_ds = VOCSegmentation(root, '2012', 'val', False, val_img_tf, val_msk_tf)
    print(f"数据集加载完成! 总共有 {len(val_ds)} 个样本")
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,  # 随机抽取样本
        num_workers=2
    )
    
    # 可视化一些样本
    print("可视化部分数据样本...")
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, (images, masks) in enumerate(val_loader):
        if i >= num_samples:
            break
            
        # 展示这个批次中的第一个样本
        img = images[0].permute(1, 2, 0).numpy()
        # 将图像从[0,1]转换为[0,255]
        img = (img * 255).astype(np.uint8)
        
        # 处理掩码
        mask = masks[0].squeeze(0)
        # VOC掩码中255表示忽略的区域，我们将其设为0（背景）
        mask[mask == 255] = 0
        
        # 将掩码转换为彩色图
        color_mask = decode_segmap(mask)
        
        # 创建半透明叠加效果
        overlay = img.copy()
        cv_mask = color_mask.copy()
        alpha = 0.4  # 透明度
        
        # 修正叠加效果实现
        cv_mask = cv2.addWeighted(cv_mask, alpha, img, 1 - alpha, 0)
        
        # 在同一行展示原图、分割掩码和叠加效果
        plt.subplot(num_samples, 3, i*3+1)
        plt.title("原图")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+2)
        plt.title("分割掩码")
        plt.imshow(color_mask)
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+3)
        plt.title("叠加效果")
        plt.imshow(cv_mask)
        plt.axis('off')
        
        # 统计掩码中存在的类别
        unique_classes = torch.unique(mask).numpy()
        class_names = [VOC_CLASSES[cls] for cls in unique_classes if cls < 21]
        print(f"样本 {i+1} 中的类别: {', '.join(class_names)}")
    
    plt.tight_layout()
    plt.savefig('voc_visualization.png')
    print("可视化结果已保存为 'voc_visualization.png'")
    plt.show()

if __name__ == "__main__":
    main()
