# test_train.py
# ---------------------------------------------------------
# UNet++ on PASCAL-VOC 2012 Semantic Segmentation - Training and Testing
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
from torchmetrics.classification import MulticlassJaccardIndex  # IoU calculation

# Import custom network and dataset
from UnetppModel import UNetPlusPlus
from DatasetVoc2012 import VOC2012Dataset, VOC_COLORMAP, VOC_CLASSES

# ------------------------------ Utils ------------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Convert segmentation mask to RGB color image
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

# Calculate cross entropy loss, ignoring pixels with label 255
def cross_entropy_loss(pred, target):
    # Ensure target mask is Long type
    if target.dtype != torch.long:
        target = target.long()
    return F.cross_entropy(pred, target, ignore_index=255, reduction='mean')

# Dice Loss implementation
def dice_loss(pred, target, smooth=1.0):
    # Ensure target is Long type
    if target.dtype != torch.long:
        target = target.long()
        
    pred = F.softmax(pred, dim=1)
    
    # Ignore 255 labels (boundary regions)
    valid_mask = (target != 255)
    target = target * valid_mask
    
    # One-hot encoding
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    # Calculate Dice coefficient
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()

# ------------------------------ Training Function ------------------------------
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    batch_count = len(train_loader)
    start_time = time.time()
    
    # Add mIoU calculation
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass - using deep supervision
        output0, output1, output2 = model(data)
        
        # Calculate loss - main output and auxiliary outputs
        main_loss = cross_entropy_loss(output0, target) + 0.5 * dice_loss(output0, target)
        aux1_loss = cross_entropy_loss(output1, target)
        aux2_loss = cross_entropy_loss(output2, target)
        
        # Combined loss
        loss = main_loss + 0.4 * aux1_loss + 0.2 * aux2_loss
        
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
        
        # Print training information
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{batch_count} ({100. * batch_idx / batch_count:.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average loss and training time
    avg_loss = total_loss / batch_count
    avg_iou = iou_score / batch_count
    epoch_time = time.time() - start_time
    print(f'Train Epoch: {epoch}, Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}, Time: {epoch_time:.2f}s')
    
    return avg_loss, avg_iou

# ------------------------------ Validation Function ------------------------------
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    iou_score = 0
    batch_count = len(val_loader)
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output0, _, _ = model(data)
            
            # Calculate loss
            loss = cross_entropy_loss(output0, target) + 0.5 * dice_loss(output0, target)
            val_loss += loss.item()
            
            # Calculate IoU
            pred = output0.argmax(dim=1)
            iou_score += miou(pred, target)
    
    # Calculate average loss and IoU
    avg_loss = val_loss / batch_count
    avg_iou = iou_score / batch_count
    print(f'Validation: Average Loss: {avg_loss:.6f}, mIoU: {avg_iou:.6f}')
    
    return avg_loss, avg_iou

# ------------------------------ Visualize Predictions ------------------------------
def visualize_predictions(model, val_loader, device, num_samples=3):
    model.eval()
    
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
            pred = outputs[0].argmax(0).cpu()
            
            # Convert to visualization format
            img_np = images[0].permute(1, 2, 0).numpy()
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
            print(f"Sample {i+1}:")
            print(f"  Ground Truth: {', '.join(gt_classes)}")
            print(f"  Prediction: {', '.join(pred_classes)}")
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    print("Prediction visualization saved as 'prediction_visualization.png'")
    plt.show()

# ------------------------------ Main ------------------------------
def main():
    # Set random seed
    seed_all()
    
    # Configuration parameters
    root = 'voc_data'
    img_size = 512
    batch_size = 4
    num_epochs = 2  # Small number of epochs for quick testing
    learning_rate = 1e-3
    num_workers = 2
    
    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_ds = VOC2012Dataset(root=root, split='train', img_size=img_size)
    val_ds = VOC2012Dataset(root=root, split='val', img_size=img_size)
    
    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Create model
    print("Initializing UNet++ model...")
    model = UNetPlusPlus(in_channels=3, num_classes=21, deep_supervision=True).to(device)
    
    # Print model structure and parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Set larger T_max value for slower learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Use longer cosine period even for just 2 epochs
    
    # Visualize some training samples
    print("\nVisualizing training samples...")
    sample_batch = next(iter(train_loader))
    sample_images, sample_masks = sample_batch
    
    plt.figure(figsize=(15, 10))
    for i in range(min(4, batch_size)):
        # Convert to NumPy arrays
        img = sample_images[i].permute(1, 2, 0).numpy()
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
    
    plt.savefig('training_samples.png')
    print("Training samples visualization saved as 'training_samples.png'")
    plt.show()
    
    # Train model
    print("\nStarting training...")
    train_losses = []
    train_ious = []  # Add training IoU record
    val_losses = []
    val_ious = []
    best_iou = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        # Ensure tensor is on CPU and convert to Python scalar
        train_ious.append(train_iou.cpu().item() if isinstance(train_iou, torch.Tensor) else train_iou)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, device)
        val_losses.append(val_loss)
        # Ensure tensor is on CPU and convert to Python scalar
        val_ious.append(val_iou.cpu().item() if isinstance(val_iou, torch.Tensor) else val_iou)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'checkpoints/test_best.pt')
            print(f"New best model saved! IoU: {best_iou:.6f}")
    
    print(f"Training completed! Best mIoU: {best_iou if not isinstance(best_iou, torch.Tensor) else best_iou.cpu().item():.6f}")
    
    # Plot training and validation loss and IoU curves
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    # IoU curves
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
    
    # Output mIoU data for each epoch
    print("\nEpoch-wise mIoU during training:")
    for i, (train_iou, val_iou) in enumerate(zip(train_ious, val_ious), 1):
        print(f"Epoch {i}: Train mIoU = {train_iou:.6f}, Validation mIoU = {val_iou:.6f}")
    
    # Load best model for visualization
    print("\nLoading best model for visualization...")
    model.load_state_dict(torch.load('checkpoints/test_best.pt'))
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()
