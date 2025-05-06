import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import cv2  
import matplotlib.pyplot as plt

# VOC dataset label color mapping (21 classes, including background)
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

# VOC label names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOC2012Dataset(Dataset):
    """
    VOC2012 dataset loading class, removing data augmentation to ensure proper alignment of images and masks
    """
    
    def __init__(self, root='voc_data', split='train', transform=None, target_transform=None, 
                 img_size=512):
        """
        Initialize VOC2012 dataset
        
        Parameters:
            root: Dataset root directory
            split: 'train', 'val' or 'trainval'
            transform: Image transformation function
            target_transform: Label transformation function
            img_size: Output image size
        """
        # Set image transformations
        if transform is None:
            # Standard transformation, no data augmentation
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  # Use fixed size to ensure consistency
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transform
            
        # Set mask transformations
        if target_transform is None:
            # Mask transformation should be consistent with image transformation
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.PILToTensor(),
            ])
        else:
            self.mask_transform = target_transform
            
        # Initialize base VOCSegmentation dataset
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
        """Get a sample from the dataset"""
        img, mask = self.dataset[idx]
        
        # Process mask: In VOC dataset, 255 represents ignored regions, we can choose to process it
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0)  # Remove channel dimension [1, H, W] -> [H, W]
            # Optional: Set 255 (ignore regions) to 0 (background)
            # mask[mask == 255] = 0
        
        return img, mask
    
    @staticmethod
    def decode_segmap(mask):
        """
        Convert segmentation mask to RGB color image
        mask: [H, W] tensor with class indices 0-20
        returns: [H, W, 3] numpy array, RGB color image
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
        Get class names present in the mask
        mask: [H, W] tensor or numpy array with class indices 0-20
        returns: List of present class names
        """
        if isinstance(mask, torch.Tensor):
            unique_classes = torch.unique(mask).numpy()
        else:
            unique_classes = np.unique(mask)
            
        return [VOC_CLASSES[cls] for cls in unique_classes if cls < 21]

# Usage example
def create_voc_dataloaders(root='voc_data', batch_size=8, img_size=512, num_workers=2):
    """
    Create training and validation data loaders for VOC2012 dataset
    
    Parameters:
        root: Dataset root directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading threads
        
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    # Create training dataset (without data augmentation)
    train_ds = VOC2012Dataset(
        root=root,
        split='train',
        img_size=img_size
    )
    
    # Create validation dataset
    val_ds = VOC2012Dataset(
        root=root,
        split='val',
        img_size=img_size
    )
    
    # Create data loaders
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

# Test code
if __name__ == "__main__":
    
    # Create datasets
    train_ds = VOC2012Dataset(split='train', img_size=512)
    val_ds = VOC2012Dataset(split='val', img_size=512)
    
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    
    # Visualize several samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(3):
        img, mask = val_ds[i]
        
        # Convert image format for display
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert mask to color image
        color_mask = VOC2012Dataset.decode_segmap(mask)
        
        # Create overlay effect
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)
        
        # Display image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        # Display color mask
        axes[i, 1].imshow(color_mask)
        axes[i, 1].set_title("Segmentation Mask")
        axes[i, 1].axis('off')
        
        # Display overlay effect
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
        
        # Get and display class names
        class_names = VOC2012Dataset.get_class_names(mask)
        print(f"Sample {i+1} classes: {', '.join(class_names)}")
    
    plt.tight_layout()
    plt.savefig('aligned_samples.png')
    plt.show()