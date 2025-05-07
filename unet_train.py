import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from DatasetVoc2012 import create_voc_dataloaders, VOC2012Dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def visualize_predictions(model, dataset, device, num_samples=5, save_path="predictions.png"):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

        # Decode ground truth and prediction
        gt_color = VOC2012Dataset.decode_segmap(mask.numpy())
        pred_color = VOC2012Dataset.decode_segmap(pred)

        # Denormalize image for display
        img_np = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)
        img_uint = (img_np * 255).astype(np.uint8)

        # Overlay prediction on image
        overlay = cv2.addWeighted(img_uint, 0.7, pred_color, 0.3, 0)

        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Device configuration
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA errors for easier debugging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders
    train_loader, val_loader = create_voc_dataloaders(
        root='voc_data',
        batch_size=8,
        img_size=512,
        num_workers=4,
        use_augmentation=True
    )

    # Initialize pre-trained UNet model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=21
    )
    model.to(device)

    # Loss and optimizer
    # Ignore the VOC 'void' label (255) to avoid out-of-range errors
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Fine-tuning
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}')

    # Save fine-tuned weights
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', 'unet_finetuned.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Saved fine-tuned model weights to {save_path}')

    # Visualize on validation set
    visualize_predictions(model, val_loader.dataset, device, num_samples=5, save_path='predictions.png')
    print('Saved visualizations to predictions.png')


if __name__ == '__main__':
    main()
