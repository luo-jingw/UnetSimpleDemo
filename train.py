#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------------------------------
# UNet++ on PASCAL-VOC 2012 Semantic Segmentation - Training Script
# ---------------------------------------------------------
import os
import random
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from torchmetrics.classification import MulticlassJaccardIndex  # IoU calculation
from tqdm import tqdm
import logging

# Import custom network and dataset
from UnetppModel import UNetPlusPlus
from DatasetVoc2012 import VOC2012Dataset, VOC_COLORMAP, VOC_CLASSES

# ------------------------------ Logger Configuration ------------------------------
def setup_logger(log_dir):
    """Configure logger"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# ------------------------------ Utils ------------------------------
def seed_all(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Convert segmentation mask to RGB color image
def decode_segmap(mask):
    """
    Convert segmentation mask to RGB color image
    mask: [H, W] tensor containing class indices 0-20
    returns: [H, W, 3] numpy array, RGB color image
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(21):
        rgb[mask == cls] = VOC_COLORMAP[cls]
            
    return rgb

# ------------------------------ Loss Functions ------------------------------
# Calculate cross entropy loss, ignoring pixels with label 255
def cross_entropy_loss(pred, target, weights=None):
    """Cross entropy loss with class weights"""
    # Ensure target mask is Long type
    if target.dtype != torch.long:
        target = target.long()
    return F.cross_entropy(pred, target, weight=weights, ignore_index=255, reduction='mean')

def calculate_loss(outputs, target, class_weights=None):
    """
    简化的损失计算函数，仅使用交叉熵损失
    outputs: 模型的输出元组 (output0, output1, output2)
    target: 目标分割掩码
    class_weights: 可选的类别权重
    """
    output0, output1, output2 = outputs
    
    # 主输出和辅助输出的交叉熵损失
    main_loss = cross_entropy_loss(output0, target, class_weights)
    aux1_loss = cross_entropy_loss(output1, target, class_weights)
    aux2_loss = cross_entropy_loss(output2, target, class_weights)
    
    # 按权重组合损失 (主输出权重更高)
    loss = main_loss + 0.4 * aux1_loss + 0.2 * aux2_loss
    
    return loss

# ------------------------------ Training Function ------------------------------
def train_one_epoch(model, train_loader, optimizer, device, epoch, logger, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_count = len(train_loader)
    start_time = time.time()
    
    # Calculate IoU metric
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    
    # Use tqdm to display progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass - using deep supervision
        output0, output1, output2 = model(data)
        
        # Calculate loss - main output and auxiliary outputs
        loss = calculate_loss((output0, output1, output2), target, class_weights)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Calculate IoU during training (using main output)
        with torch.no_grad():
            pred = output0.argmax(dim=1)
            iou_score += miou(pred, target)
        
        # Update progress bar info
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    # Calculate average loss and training time
    avg_loss = total_loss / batch_count
    avg_iou = iou_score / batch_count
    epoch_time = time.time() - start_time
    
    # Log information
    logger.info(f'Train Epoch: {epoch}, Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}, Time: {epoch_time:.2f}s')
    
    return avg_loss, avg_iou

# ------------------------------ Validation Function ------------------------------
def validate(model, val_loader, device, logger, class_weights=None):
    """Validate the model"""
    model.eval()
    val_loss = 0
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    batch_count = len(val_loader)
    
    # Use tqdm to display progress bar
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output0, _, _ = model(data)
            
            # Calculate loss
            loss = cross_entropy_loss(output0, target, class_weights)
            val_loss += loss.item()
            
            # Calculate IoU
            pred = output0.argmax(dim=1)
            iou_score += miou(pred, target)
            
            # Update progress bar info
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate average loss and IoU
    avg_loss = val_loss / batch_count
    avg_iou = iou_score / batch_count
    
    # Log information
    logger.info(f'Validation: Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}')
    
    return avg_loss, avg_iou

# ------------------------------ Visualize Predictions ------------------------------
def visualize_predictions(model, val_loader, device, output_dir, epoch, num_samples=3):
    """Visualize model predictions"""
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        # Get specified number of samples
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # Get the first sample in a batch
            image = images[0:1].to(device)
            mask = masks[0].cpu()
            
            # Model prediction
            outputs, _, _ = model(image)
            
            # Check original output statistics
            print(f"\nSample {i+1} prediction statistics:")
            print(f"Output tensor shape: {outputs.shape}")
            print(f"Output min value: {outputs.min().item():.4f}, max value: {outputs.max().item():.4f}")
            print(f"Output mean: {outputs.mean().item():.4f}, std: {outputs.std().item():.4f}")
            
            # Apply softmax to output
            probs = F.softmax(outputs[0], dim=0)
            
            # Get maximum probability for each class
            max_probs, _ = torch.max(probs, dim=0)
            
            # Get predicted class
            pred = outputs[0].argmax(0).cpu()
            
            # 修复：确保预测的类别索引在合法范围内（0-20）
            pred = torch.clamp(pred, 0, 20)
            
            # Print predicted class distribution
            unique_classes, counts = np.unique(pred.numpy(), return_counts=True)
            print(f"Predicted class distribution: {list(zip(unique_classes, counts))}")
            
            # Convert to visualization format
            img_np = images[0].permute(1, 2, 0).numpy()
            # 反归一化处理
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            
            # In VOC masks, 255 represents ignore regions, set to 0 (background) for display
            mask_vis = mask.clone()
            mask_vis[mask_vis == 255] = 0
            
            # Convert to color images
            gt_color = decode_segmap(mask_vis)
            pred_color = decode_segmap(pred)
            
            # Display original image
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # Display ground truth mask
            axes[i, 1].imshow(gt_color)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Display predicted mask
            axes[i, 2].imshow(pred_color)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            # Output class information
            gt_classes = [VOC_CLASSES[cls] for cls in torch.unique(mask_vis).numpy() if cls < 21]
            pred_classes = [VOC_CLASSES[cls] for cls in torch.unique(pred).numpy() if cls < 21]
            print(f"Ground Truth classes: {', '.join(gt_classes)}")
            print(f"Predicted classes: {', '.join(pred_classes)}")
            
            # Save prediction confidence map
            plt.figure(figsize=(8, 6))
            plt.imshow(max_probs.cpu().numpy(), cmap='hot')
            plt.colorbar(label='Confidence')
            plt.title(f"Sample {i+1} Prediction Confidence")
            plt.savefig(os.path.join(output_dir, f'confidence_epoch{epoch}_sample{i+1}.png'))
            plt.close()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_epoch{epoch}.png'))
    print(f"Prediction visualization saved for epoch {epoch}")
    plt.close()

# ------------------------------ Model Training Main Function ------------------------------
def train(args):
    """Main function for model training"""
    # Set random seed
    seed_all(args.seed)
    
    # Set up logger
    logger = setup_logger(args.log_dir)
    logger.info(f"Starting training with config: {args}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_ds = VOC2012Dataset(root=args.data_root, split='train', img_size=args.img_size)
    val_ds = VOC2012Dataset(root=args.data_root, split='val', img_size=args.img_size)
    
    logger.info(f"Training set: {len(train_ds)} samples")
    logger.info(f"Validation set: {len(val_ds)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Create model
    logger.info("Initializing UNet++ model...")
    model = UNetPlusPlus(
        in_channels=3, 
        num_classes=21, 
        deep_supervision=True
    ).to(device)
    
    # Print model structure and parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameter count: {total_params:,}")
    
    # Set up class weights (optional)
    if args.use_class_weights:
        # Give background lower weight
        class_weights = torch.ones(21).to(device)
        class_weights[0] = 0.5  # Lower background weight
        logger.info("Using class weights, background weight reduced to 0.5")
    else:
        class_weights = None
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        logger.info(f"Using cosine annealing scheduler, initial lr={args.lr}, min lr={args.min_lr}")
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        logger.info(f"Using step scheduler, reducing by factor {args.gamma} every {args.step_size} epochs")
    else:
        scheduler = None
        logger.info(f"Using fixed learning rate: {args.lr}")
    
    # Resume training (if checkpoint specified)
    start_epoch = 1
    best_iou = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['best_iou']
            logger.info(f"Resuming from epoch {start_epoch}, best mIoU: {best_iou:.6f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    # Visualize some training samples
    logger.info("Visualizing training samples...")
    sample_batch = next(iter(train_loader))
    sample_images, sample_masks = sample_batch
    
    plt.figure(figsize=(15, 10))
    for i in range(min(4, len(sample_images))):
        # Convert to NumPy arrays and un-normalize
        img = sample_images[i].permute(1, 2, 0).numpy()
        # 添加反归一化处理
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        mask = sample_masks[i].clone()
        # Set 255 (ignore regions) to 0 (background) for visualization
        mask[mask == 255] = 0
        color_mask = decode_segmap(mask)
        
        # Create semi-transparent overlay
        overlay = cv2.addWeighted(color_mask, 0.4, img, 0.6, 0)
        
        # Display images
        plt.subplot(2, 4, i+1)
        plt.title(f"Image {i+1}")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        plt.title(f"Mask {i+1}")
        plt.imshow(overlay)
        plt.axis('off')
    
    plt.savefig(os.path.join(args.output_dir, 'training_samples.png'))
    logger.info("Training sample visualization saved")
    plt.close()
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n===== Epoch {epoch}/{args.epochs} =====")
        
        # Train for one epoch
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger, class_weights
        )
        train_losses.append(train_loss)
        train_ious.append(train_iou.cpu().item() if isinstance(train_iou, torch.Tensor) else train_iou)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, device, logger, class_weights)
        val_losses.append(val_loss)
        val_ious.append(val_iou.cpu().item() if isinstance(val_iou, torch.Tensor) else val_iou)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # 根据save-freq参数决定是否保存检查点
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            # 保存epoch检查点
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_iou': best_iou,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou
            }, checkpoint_path)
            logger.info(f"Saved epoch {epoch} checkpoint: {checkpoint_path}")
            
            # 如果启用keep-last参数(>0)，则仅保留最近的N个检查点
            if args.keep_last > 0:
                checkpoints = [f for f in os.listdir(args.checkpoint_dir) 
                              if f.startswith('epoch') and f.endswith('.pt') and f != 'best.pt']
                # 按照epoch编号排序
                checkpoints.sort(key=lambda x: int(x.replace('epoch', '').replace('.pt', '')), reverse=True)
                # 仅保留最近的args.keep_last个检查点
                for old_ckpt in checkpoints[args.keep_last:]:
                    old_path = os.path.join(args.checkpoint_dir, old_ckpt)
                    try:
                        os.remove(old_path)
                        logger.info(f"Removed old checkpoint: {old_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")
        
        # 保存最佳模型
        current_iou = val_iou.cpu().item() if isinstance(val_iou, torch.Tensor) else val_iou
        if current_iou > best_iou:
            best_iou = current_iou
            best_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_iou': best_iou,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou
            }, best_path)
            logger.info(f"Found new best model! mIoU: {best_iou:.6f}, saved to {best_path}")
        
        # Plot current training progress
        if epoch % args.vis_freq == 0 or epoch == args.epochs:
            # Plot loss and IoU curves
            plt.figure(figsize=(15, 5))
            
            # Loss curves
            plt.subplot(1, 2, 1)
            epochs_range = list(range(start_epoch, start_epoch + len(train_losses)))
            plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            plt.grid(True)
            
            # IoU curves
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, train_ious, 'b-', label='Training mIoU')
            plt.plot(epochs_range, val_ious, 'r-', label='Validation mIoU')
            plt.xlabel('Epochs')
            plt.ylabel('mIoU')
            plt.legend()
            plt.title('mIoU Curves')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'training_metrics_epoch{epoch}.png'))
            logger.info(f"Training metrics visualization saved (epoch {epoch})")
            plt.close()
            
            # Visualize prediction results
            logger.info(f"Generating prediction visualizations for epoch {epoch}...")
            visualize_predictions(
                model, val_loader, device, args.vis_dir, epoch, num_samples=args.vis_samples
            )
    
    # Plot complete training curves
    plt.figure(figsize=(15, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    epochs_range = list(range(start_epoch, start_epoch + len(train_losses)))
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    # IoU curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_ious, 'b-', label='Training mIoU')
    plt.plot(epochs_range, val_ious, 'r-', label='Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.title('mIoU Curves')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'final_training_metrics.png'))
    logger.info("Final training metrics visualization saved")
    plt.close()
    
    # Output mIoU data for each epoch
    logger.info("\nmIoU for each epoch during training:")
    for i, (t_iou, v_iou) in enumerate(zip(train_ious, val_ious), start_epoch):
        logger.info(f"Epoch {i}: Train mIoU = {t_iou:.6f}, Validation mIoU = {v_iou:.6f}")
    
    # Save final model (not best model)
    final_model_path = os.path.join(args.output_dir, 'unetpp_voc2012_final.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training complete! Final model saved to: {final_model_path}")
    logger.info(f"Best mIoU: {best_iou:.6f}")
    
    # Load best model for final visualization
    logger.info("\nLoading best model for final visualization...")
    best_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best.pt'))
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    logger.info(f"load best model from epoch {best_epoch}")
    
    # Generate visualizations for the best model
    visualize_predictions(
        model, val_loader, device, args.vis_dir, 'best', num_samples=args.vis_samples
    )
    logger.info("Best model visualizations saved")
    
    return {
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'final_train_iou': train_ious[-1],
        'final_val_iou': val_ious[-1],
    }

# ------------------------------ Main Function ------------------------------
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='UNet++ Semantic Segmentation Training Script')
    
    # Dataset parameters
    parser.add_argument('--data-root', default='voc_data', type=str, help='Path to dataset')
    parser.add_argument('--img-size', default=512, type=int, help='Image size')
    
    # Training parameters
    parser.add_argument('--batch-size', default=8, type=int, help='Training batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='Minimum learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--num-workers', default=6, type=int, help='Number of data loading workers')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'none'], 
                        help='Learning rate scheduler type')
    parser.add_argument('--step-size', default=10, type=int, help='Step size for step scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='Decay rate for step scheduler')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    # Training control parameters
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--save-freq', default=10, type=int, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--keep-last', default=10, type=int, help='Number of most recent checkpoints to keep (0 = keep all)')
    parser.add_argument('--vis-freq', default=10, type=int, help='Visualization frequency (epoch)')
    parser.add_argument('--vis-samples', default=5, type=int, help='Number of samples to visualize each time')
    
    # Output parameters
    parser.add_argument('--output-dir', default='output', type=str, help='Output directory')
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str, help='Checkpoint save directory')
    parser.add_argument('--vis-dir', default='visualizations', type=str, help='Visualization output directory')
    parser.add_argument('--log-dir', default='logs', type=str, help='Log save directory')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    results = train(args)
    print(f"Training complete! Best mIoU: {results['best_iou']:.6f} (Epoch {results['best_epoch']})")