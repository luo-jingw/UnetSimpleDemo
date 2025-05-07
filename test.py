import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms, models
from PIL import Image
import cv2
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Import custom modules
from UnetppModel import UNetPlusPlus, ConvBlock, DecoderBlock
from DatasetVoc2012 import VOC2012Dataset, create_voc_dataloaders, VOC_CLASSES

def plot_sample(image, mask, pred=None):
    """
    Plot image, segmentation mask and prediction (if available)
    
    Args:
        image: Input image tensor [3, H, W]
        mask: Ground truth segmentation mask [H, W]
        pred: Predicted segmentation mask [H, W] (optional)
    """
    # Convert image to numpy format for display
    img_np = image.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # Handle 255 values in mask (ignore regions)
    mask_display = mask.clone()
    if torch.is_tensor(mask_display):
        mask_display[mask_display == 255] = 0
    
    # Convert mask to RGB color map
    color_mask = VOC2012Dataset.decode_segmap(mask_display)
    
    # Create overlay effect
    overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.7, color_mask, 0.3, 0)
    
    if pred is None:
        # Visualization without prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(color_mask)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Ground Truth Overlay")
        axes[2].axis('off')
    else:
        # Visualization with prediction
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(color_mask)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Handle potential 255 values in prediction 
        if torch.is_tensor(pred):
            pred_display = pred.clone()
            pred_display[pred_display == 255] = 0
        else:
            pred_display = pred.copy()
            if hasattr(pred_display, 'shape'):
                if pred_display.max() == 255:
                    pred_display[pred_display == 255] = 0
                    
        # Convert prediction to RGB color map
        pred_color = VOC2012Dataset.decode_segmap(pred_display)
        axes[2].imshow(pred_color)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        # Prediction overlay
        pred_overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.7, pred_color, 0.3, 0)
        axes[3].imshow(pred_overlay)
        axes[3].set_title("Prediction Overlay")
        axes[3].axis('off')
    
    # Get and display class names (ignoring 255 class)
    mask_for_names = mask.clone()
    if torch.is_tensor(mask_for_names):
        mask_for_names[mask_for_names == 255] = 0  # Ignore 255 class
    class_names = VOC2012Dataset.get_class_names(mask_for_names)
    plt.suptitle(f"Classes: {', '.join(class_names)}")
    plt.tight_layout()
    
    return fig

# Create a modified UNet++ model with pretrained ResNet34 encoder
class UNetPlusPlusWithResNet(nn.Module):
    """
    UNet++ architecture with a pretrained ResNet34 encoder
    """
    def __init__(self, in_channels=3, num_classes=21, deep_supervision=True):
        super(UNetPlusPlusWithResNet, self).__init__()
        self.deep_supervision = deep_supervision
        
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # Freeze first 2 layers (layer0 = conv1+bn1+relu, layer1 = maxpool+layer1)
        for name, param in resnet.named_parameters():
            if "conv1" in name or "bn1" in name or "layer1" in name:
                param.requires_grad = False
                
        print("Initialized ResNet34 encoder with first 2 layers frozen")
        
        # Extract ResNet34 layers as encoder stages
        # Initial number of features
        filters = [64, 64, 128, 256, 512]
        
        # Use ResNet layers as encoder
        self.encoder1 = nn.Sequential(
            resnet.conv1,    # 7x7 conv, 64 channels
            resnet.bn1,      # BatchNorm
            resnet.relu      # ReLU
        )  # Output: 64 channels
        
        self.pool = resnet.maxpool
        
        self.encoder2 = resnet.layer1   # Output: 64 channels
        self.encoder3 = resnet.layer2   # Output: 128 channels
        self.encoder4 = resnet.layer3   # Output: 256 channels
        
        # Center (bottleneck)
        self.center = resnet.layer4     # Output: 512 channels
        
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
        
        # Encoder Path with ResNet34
        x1_0_pre = self.encoder1(x)         # Level 1 features (before pooling)
        x = self.pool(x1_0_pre)
        
        x1_0 = x1_0_pre                     # Level 1 features
        x2_0 = self.encoder2(x)             # Output: 64 channels
        x3_0 = self.encoder3(x2_0)          # Output: 128 channels
        x4_0 = self.encoder4(x3_0)          # Output: 256 channels
        
        # Center
        x5_0 = self.center(x4_0)              # Output: 512 channels
        
        # First layer of nested decoder - L=0
        x4_1 = self.decoder0_1(x5_0, x4_0)  # Up from center to level 4
        x3_1 = self.decoder1_1(x4_1, x3_0)  # Up from level 4 to level 3
        x2_1 = self.decoder2_1(x3_1, x2_0)  # Up from level 3 to level 2
        x1_1 = self.decoder3_1(x2_1, x1_0)  # Up from level 2 to level 1
        
        # Second layer of nested decoder - L=1
        x4_2 = self.decoder0_2(torch.cat([x5_0, x4_1], dim=1), x4_0)
        x3_2 = self.decoder1_2(torch.cat([x4_1, x3_1], dim=1), x3_0)
        x2_2 = self.decoder2_2(torch.cat([x3_1, x2_1], dim=1), x2_0)
        
        # Third layer of nested decoder - L=2
        x4_3 = self.decoder0_3(torch.cat([x5_0, x4_1, x4_2], dim=1), x4_0)
        x3_3 = self.decoder1_3(torch.cat([x4_2, x3_1, x3_2], dim=1), x3_0)
        
        # Fourth layer of nested decoder - L=3
        x4_4 = self.decoder0_4(torch.cat([x5_0, x4_1, x4_2, x4_3], dim=1), x4_0)
        
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

# Calculate IoU metrics
def calculate_iou(pred, target, num_classes=21):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Ignore index 255
    mask = (target != 255)
    pred = pred[mask]
    target = target[mask]
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        
        if union == 0:
            ious.append(float('nan'))  # No instances of this class
        else:
            ious.append(intersection / union)
    
    return ious

# Function to train one epoch
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    batch_ious = []
    
    start_time = time.time()
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(torch.long).to(device)
        
        # Forward pass - new model always returns single tensor
        outputs = model(images)
        
        # Calculate loss (ignoring index 255)
        loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate IoU
        pred = torch.argmax(outputs, dim=1)
        batch_iou = calculate_iou(pred, masks)
        batch_ious.append(batch_iou)
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    avg_iou = np.nanmean([iou for batch in batch_ious for iou in batch if not np.isnan(iou)])
    
    elapsed_time = time.time() - start_time
    
    print(f'Epoch [{epoch+1}] completed in {elapsed_time:.2f}s, Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}')
    
    return avg_loss, avg_iou

# Function to validate
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    batch_ious = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1)
            batch_iou = calculate_iou(pred, masks)
            batch_ious.append(batch_iou)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = np.nanmean([iou for batch in batch_ious for iou in batch if not np.isnan(iou)])
    
    print(f'Validation - Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}')
    
    return avg_val_loss, avg_val_iou

# Function to visualize predictions during training
def visualize_predictions(model, val_loader, device, epoch, num_samples=2):
    model.eval()
    
    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(20, 10*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        fig = plot_sample(images[i].cpu(), masks[i], preds[i])
        plt.savefig(f'prediction_epoch_{epoch+1}_sample_{i+1}.png')
        plt.close(fig)

def save_checkpoint(model, optimizer, epoch, loss, iou, filename):
    """Save model checkpoint with additional information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iou': iou
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint with additional information"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    iou = checkpoint.get('iou', 0)  # Backward compatibility
    print(f"Loaded checkpoint from epoch {epoch}, Loss: {loss:.4f}, IoU: {iou:.4f}")
    return model, optimizer, epoch, loss, iou

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load datasets with data augmentation
    train_loader, val_loader = create_voc_dataloaders(batch_size=8, img_size=512)
    print(f"Training set: {len(train_loader.dataset)} samples")
    print(f"Validation set: {len(val_loader.dataset)} samples")
    
    # Initialize model - option to use standard UNetPlusPlus or version with ResNet encoder
    use_resnet = False  # Set to True to use ResNet encoder
    
    if use_resnet:
        model = UNetPlusPlusWithResNet(in_channels=3, num_classes=21, deep_supervision=True)
        print("Using UNet++ with ResNet34 encoder")
    else:
        model = UNetPlusPlus(in_channels=3, num_classes=21, deep_supervision=True)
        print("Using standard UNet++")
        
    model = model.to(device)
    
    # Define loss function - ignoring index 255
    criterion = CrossEntropyLoss(ignore_index=255)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training parameters
    num_epochs = 20
    best_val_iou = 0
    start_epoch = 0
    
    # Option to resume training from checkpoint
    resume_training = False
    checkpoint_path = 'checkpoints/best.pt'
    
    if resume_training and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _, best_val_iou = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch+1}")
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        # Save periodic checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, val_iou, f'checkpoints/epoch{epoch+1}.pt')
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(model, optimizer, epoch, val_loss, val_iou, 'checkpoints/best.pt')
            print(f"New best model saved with IoU: {best_val_iou:.4f}")
        
        # Visualize predictions every 5 epochs
        if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_predictions(model, val_loader, device, epoch, num_samples=2)
        
        # Plot and save training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_ious, label='Train IoU')
        plt.plot(val_ious, label='Val IoU')
        plt.title('IoU Curves')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'visualizations/training_curves_epoch_{epoch+1}.png')
        plt.close()
    
    print("Training completed!")
    
    # Load best model and evaluate
    model, optimizer, _, _, _ = load_checkpoint(model, optimizer, 'checkpoints/best.pt')
    val_loss, val_iou = validate(model, val_loader, criterion, device)
    print(f"Best model performance - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
    
    # Visualize some final predictions
    visualize_predictions(model, val_loader, device, num_epochs, num_samples=4)