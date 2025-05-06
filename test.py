# test_train.py
# ---------------------------------------------------------
# UNet++ on PASCAL-VOC 2012语义分割 - 训练和测试
# ---------------------------------------------------------
import os, random, torch, numpy as np
import time
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchmetrics.classification import MulticlassJaccardIndex  # IoU计算

# 导入自定义网络和数据集
from UnetppModel import UNetPlusPlus
from DatasetVoc2012 import VOC2012Dataset, VOC_COLORMAP, VOC_CLASSES

# ------------------------------ Utils ------------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 将分割掩码转换为RGB彩色图像
def decode_segmap(mask):
    """
    Convert segmentation mask to RGB color image
    mask: [H, W] tensor with class indices 0-20
    returns: [H, W, 3] numpy array, RGB color image
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(21):
        rgb[mask == cls] = VOC_COLORMAP[cls]
            
    return rgb

# 计算交叉熵损失，忽略标签为255的像素
def cross_entropy_loss(pred, target):
    # 确保目标掩码是Long类型
    if target.dtype != torch.long:
        target = target.long()
    return F.cross_entropy(pred, target, ignore_index=255, reduction='mean')

# Dice Loss实现
def dice_loss(pred, target, smooth=1.0):
    # 确保target是Long类型
    if target.dtype != torch.long:
        target = target.long()
        
    pred = F.softmax(pred, dim=1)
    
    # 忽略255标签（边界区域）
    valid_mask = (target != 255)
    target = target * valid_mask
    
    # one-hot编码
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    # 计算Dice系数
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()

# ------------------------------ 训练函数 ------------------------------
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    batch_count = len(train_loader)
    start_time = time.time()
    
    # 添加mIoU计算
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 转移数据到设备
        data, target = data.to(device), target.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播 - 使用深度监督
        output0, output1, output2 = model(data)
        
        # 计算损失 - 主输出和辅助输出
        main_loss = cross_entropy_loss(output0, target) + 0.5 * dice_loss(output0, target)
        aux1_loss = cross_entropy_loss(output1, target)
        aux2_loss = cross_entropy_loss(output2, target)
        
        # 组合损失
        loss = main_loss + 0.4 * aux1_loss + 0.2 * aux2_loss
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 计算训练中的IoU（使用主输出）
        with torch.no_grad():
            pred = output0.argmax(dim=1)
            iou_score += miou(pred, target)
        
        # 打印训练信息
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{batch_count} ({100. * batch_idx / batch_count:.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和训练时间
    avg_loss = total_loss / batch_count
    avg_iou = iou_score / batch_count
    epoch_time = time.time() - start_time
    print(f'Train Epoch: {epoch}, Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}, Time: {epoch_time:.2f}s')
    
    return avg_loss, avg_iou

# ------------------------------ 验证函数 ------------------------------
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    batch_count = len(val_loader)
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output0, _, _ = model(data)
            
            # 计算损失
            loss = cross_entropy_loss(output0, target) + 0.5 * dice_loss(output0, target)
            val_loss += loss.item()
            
            # 计算IoU
            pred = output0.argmax(dim=1)
            iou_score += miou(pred, target)
    
    # 计算平均损失和IoU
    avg_loss = val_loss / batch_count
    avg_iou = iou_score / batch_count
    print(f'Validation: Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}')
    
    return avg_loss, avg_iou

# ------------------------------ 可视化预测结果 ------------------------------
def visualize_predictions(model, val_loader, device, num_samples=3):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        # 获取指定数量的样本
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # 获取一个批次中的第一个样本
            image = images[0:1].to(device)
            mask = masks[0].cpu()
            
            # 模型预测
            outputs, _, _ = model(image)
            pred = outputs[0].argmax(0).cpu()
            
            # 转换为可视化格式
            img_np = images[0].permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 在VOC掩码中，255表示忽略区域，显示时将其设为0（背景）
            mask_vis = mask.clone()
            mask_vis[mask_vis == 255] = 0
            
            # 转换为彩色图
            gt_color = decode_segmap(mask_vis)
            pred_color = decode_segmap(pred)
            
            # 显示原图
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # 显示真实掩码
            axes[i, 1].imshow(gt_color)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # 显示预测掩码
            axes[i, 2].imshow(pred_color)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            # 输出类别信息
            gt_classes = [VOC_CLASSES[cls] for cls in torch.unique(mask_vis).numpy() if cls < 21]
            pred_classes = [VOC_CLASSES[cls] for cls in torch.unique(pred).numpy() if cls < 21]
            print(f"Sample {i+1}:")
            print(f"  Ground Truth: {', '.join(gt_classes)}")
            print(f"  Prediction: {', '.join(pred_classes)}")
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    print("Prediction visualization saved as 'prediction_visualization.png'")
    plt.show()

# ------------------------------ Main ------------------------------
def main():
    # 设置随机种子
    seed_all()
    
    # 配置参数
    root = 'voc_data'
    img_size = 512
    batch_size = 4
    num_epochs = 2  # 为快速测试设置的小轮数
    learning_rate = 1e-3
    num_workers = 2
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    print("Loading datasets...")
    train_ds = VOC2012Dataset(root=root, split='train', img_size=img_size)
    val_ds = VOC2012Dataset(root=root, split='val', img_size=img_size)
    
    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    
    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 创建模型
    print("Initializing UNet++ model...")
    model = UNetPlusPlus(in_channels=3, num_classes=21, deep_supervision=True).to(device)
    
    # 打印模型结构和参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 可视化一些训练样本
    print("\nVisualizing training samples...")
    sample_batch = next(iter(train_loader))
    sample_images, sample_masks = sample_batch
    
    plt.figure(figsize=(15, 10))
    for i in range(min(4, batch_size)):
        # 转换为NumPy数组
        img = sample_images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        mask = sample_masks[i].clone()
        # 将255（忽略区域）设为0（背景）用于可视化
        mask[mask == 255] = 0
        color_mask = decode_segmap(mask)
        
        # 创建半透明叠加效果
        overlay = cv2.addWeighted(color_mask, 0.4, img, 0.6, 0)
        
        # 显示图像
        plt.subplot(2, 4, i+1)
        plt.title(f"Image {i+1}")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        plt.title(f"Mask {i+1}")
        plt.imshow(overlay)
        plt.axis('off')
    
    plt.savefig('training_samples.png')
    print("Training samples visualization saved as 'training_samples.png'")
    plt.show()
    
    # 训练模型
    print("\nStarting training...")
    train_losses = []
    train_ious = []  # 添加训练IoU记录
    val_losses = []
    val_ious = []
    best_iou = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练一个周期
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_ious.append(train_iou)  # 记录训练IoU
        
        # 验证
        val_loss, val_iou = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'checkpoints/test_best.pt')
            print(f"New best model saved! IoU: {best_iou:.6f}")
    
    print(f"Training completed! Best mIoU: {best_iou:.6f}")
    
    # 绘制训练和验证的损失与IoU曲线
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    # IoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_ious, 'b-', label='Training mIoU')
    plt.plot(range(1, num_epochs + 1), val_ious, 'r-', label='Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.title('mIoU Curves')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics visualization saved as 'training_metrics.png'")
    plt.show()
    
    # 输出每个epoch的mIoU数据
    print("\n训练过程中的每个epoch的mIoU:")
    for i, (train_iou, val_iou) in enumerate(zip(train_ious, val_ious), 1):
        print(f"Epoch {i}: Train mIoU = {train_iou:.6f}, Validation mIoU = {val_iou:.6f}")
    
    # 加载最佳模型进行可视化
    print("\nLoading best model for visualization...")
    model.load_state_dict(torch.load('checkpoints/test_best.pt'))
    
    # 可视化预测结果
    print("Visualizing predictions...")
    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()
