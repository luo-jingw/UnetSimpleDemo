import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import cv2  
import matplotlib.pyplot as plt
import random

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
    VOC2012 dataset loading class with simple data augmentation for both images and masks using torchvision.transforms
    """
    
    def __init__(self, root='voc_data', split='train', transform=None, target_transform=None, 
                 img_size=512, use_augmentation=True, aug_mode='light'):
        """
        Initialize VOC2012 dataset
        
        Parameters:
            root: Dataset root directory
            split: 'train', 'val' or 'trainval'
            transform: Custom image transformation function (will override default transformations)
            target_transform: Custom label transformation function (will override default transformations)
            img_size: Output image size
            use_augmentation: Whether to use data augmentation (only applied for training split)
            aug_mode: Augmentation mode - 'light', 'medium', or 'heavy'
        """
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and split == 'train'
        self.aug_mode = aug_mode
        
        # Initialize custom transformations if provided
        self.custom_img_transform = transform
        self.custom_mask_transform = target_transform
        
        # Initialize base VOCSegmentation dataset
        self.dataset = VOCSegmentation(
            root=root, 
            year='2012',
            image_set=split,
            download=False,
            transform=None,  # We'll apply transforms ourselves
            target_transform=None
        )
        
        # Setup transforms for images and masks
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup the transformation pipelines using torchvision.transforms"""
        
        # Basic transforms (always applied)
        self.basic_img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.basic_mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
            transforms.PILToTensor(),
        ])
        
        # 统一简化的增强方式，仅包含基础的空间变换
        self.spatial_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((self.img_size, self.img_size))
        ])
        
        # 图像特定的后处理
        self.img_post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 掩码特定的后处理
        self.mask_post_transform = transforms.PILToTensor()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img, mask = self.dataset[idx]
        
        # Apply custom transformations if provided
        if self.custom_img_transform and self.custom_mask_transform:
            img = self.custom_img_transform(img)
            mask = self.custom_mask_transform(mask)
        else:
            # Apply data augmentation if enabled and in training mode
            if self.use_augmentation and self.split == 'train':
                # 使用相同的随机种子确保图像和掩码同步变换
                seed = torch.randint(0, 2**32, (1,)).item()
                
                # 对图像和掩码应用完全相同的空间变换
                torch.manual_seed(seed)
                random.seed(seed)
                img = self.spatial_transforms(img)
                
                torch.manual_seed(seed)
                random.seed(seed)
                mask = self.spatial_transforms(mask)
                
                # 应用各自特定的后处理
                img = self.img_post_transform(img)
                mask = self.mask_post_transform(mask)
            else:
                # Just apply basic transformations
                img = self.basic_img_transform(img)
                mask = self.basic_mask_transform(mask)
        
        # Make sure mask has the right shape (no channel dim)
        if len(mask.shape) > 2:
            mask = mask.squeeze(0)
            
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

    def visualize_augmentations(self, idx=0, n_samples=5):
        """
        Visualize the augmentations applied to a single sample
        
        Parameters:
            idx: Index of sample to visualize
            n_samples: Number of augmented versions to generate
            
        Returns:
            fig: Matplotlib figure with augmented samples
        """
        img, mask = self.dataset[idx]
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
        
        for i in range(n_samples):
            # 使用相同的随机种子确保图像和掩码同步变换
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # 对图像和掩码应用完全相同的空间变换
            torch.manual_seed(seed)
            random.seed(seed)
            aug_img = self.spatial_transforms(img.copy())
            
            torch.manual_seed(seed)
            random.seed(seed)
            aug_mask = self.spatial_transforms(mask.copy())
            
            # 应用各自特定的后处理
            aug_img = self.img_post_transform(aug_img)
            aug_mask = self.mask_post_transform(aug_mask)
            
            # 确保掩码是二维的
            if len(aug_mask.shape) > 2:
                aug_mask = aug_mask.squeeze(0)
            
            # Convert image to numpy for visualization
            aug_img_np = aug_img.permute(1, 2, 0).numpy()
            aug_img_np = np.clip(aug_img_np * np.array([0.229, 0.224, 0.225]) + 
                          np.array([0.485, 0.456, 0.406]), 0, 1)
            
            # Convert mask to color image
            color_mask = self.decode_segmap(aug_mask.numpy())
            
            # Create overlay
            overlay = cv2.addWeighted(
                (aug_img_np * 255).astype(np.uint8), 0.7, 
                color_mask, 0.3, 0
            )
            
            # Display
            axes[i, 0].imshow(aug_img_np)
            axes[i, 0].set_title(f"增强后的图像 {i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(color_mask)
            axes[i, 1].set_title(f"增强后的掩码 {i+1}")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"叠加效果 {i+1}")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        return fig

# Usage example
def create_voc_dataloaders(root='voc_data', batch_size=8, img_size=512, num_workers=2, 
                       use_augmentation=True, aug_mode='medium'):
    """
    Create training and validation data loaders for VOC2012 dataset
    
    Parameters:
        root: Dataset root directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading threads
        use_augmentation: Whether to use data augmentation for training
        aug_mode: Augmentation mode - 'light', 'medium', or 'heavy'
        
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    # Create training dataset with data augmentation
    train_ds = VOC2012Dataset(
        root=root,
        split='train',
        img_size=img_size,
        use_augmentation=use_augmentation,
        aug_mode=aug_mode
    )
    
    # Create validation dataset (no augmentation for validation)
    val_ds = VOC2012Dataset(
        root=root,
        split='val',
        img_size=img_size,
        use_augmentation=False
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
    train_ds = VOC2012Dataset(split='train', img_size=512, use_augmentation=True)
    val_ds = VOC2012Dataset(split='val', img_size=512)
    
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    
    # 1. Visualize normal samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(3):
        img, mask = val_ds[i]
        
        # Convert image format for display
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Convert mask to color image
        color_mask = VOC2012Dataset.decode_segmap(mask)
        
        # Create overlay effect
        overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.7, color_mask, 0.3, 0)
        
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
    plt.close()
    
    # 2. Visualize augmentations
    # Create a dataset with all augmentation modes for comparison
    aug_modes = ['light', 'medium', 'heavy']
    
    for mode in aug_modes:
        train_ds = VOC2012Dataset(split='train', img_size=512, use_augmentation=True, aug_mode=mode)
        fig = train_ds.visualize_augmentations(idx=0, n_samples=3)
        plt.savefig(f'augmented_samples_{mode}.png')
        plt.close()
    
    # 3. Test a batch from the data loader
    train_loader, _ = create_voc_dataloaders(batch_size=4, img_size=512, aug_mode='light')
    batch = next(iter(train_loader))
    images, masks = batch
    
    print(f"Batch shape: images={images.shape}, masks={masks.shape}")
    print("Training with data augmentation is ready!")