import os, torch
import numpy as np
from PIL import Image
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

# ------------------------------ Custom UNet++ Network ------------------------------ #
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2)
        
        # Nested skip connections
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv1_1 = ConvBlock(128 + 256, 128)
        self.conv2_1 = ConvBlock(256 + 512, 256)
        
        # Fixed channels calculation:
        # x0_2 input: x0_0(64) + x0_1(64) + up(x1_1)(128) = 256
        self.conv0_2 = ConvBlock(64 + 64 + 128, 64)  
        # x1_2 input: x1_0(128) + x1_1(128) + up(x2_1)(256) = 512
        self.conv1_2 = ConvBlock(128 + 128 + 256, 128)
        
        # x0_3 input: x0_0(64) + x0_1(64) + x0_2(64) + up(x1_2)(128) = 320
        self.conv0_3 = ConvBlock(64 + 64 + 64 + 128, 64)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final output
        self.final = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.enc1(x)
        x1_0 = self.enc2(self.pool(x0_0))
        x2_0 = self.enc3(self.pool(x1_0))
        x3_0 = self.enc4(self.pool(x2_0))
        
        # Decoder
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        # Output
        return self.final(x0_3)


def dice_loss(logits, targets, eps: float = 1e-6):
    """Soft-Dice across all classes, targets ∈ [B,H,W] int"""
    C = logits.shape[1]
    probs  = F.softmax(logits, 1)
    onehot = F.one_hot(targets.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    inter  = (probs * onehot).sum((0, 2, 3))
    union  = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def main():
    root, epochs, batch_size = 'voc_data', 100, 8
    lr, num_classes, out_dir = 5e-4, 21, 'checkpoints'
    accum_iter, use_amp = 4, True
    os.makedirs(out_dir, exist_ok=True)

    train_img_tf = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.5, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_mask_tf = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.5, 2.0), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    # Validation set: deterministic
    val_img_tf = transforms.Compose([
        transforms.Resize(544, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    val_mask_tf = transforms.Compose([
        transforms.Resize(544, interpolation=Image.NEAREST),
        transforms.CenterCrop(512),
        transforms.PILToTensor(),
    ])

    train_ds = VOCSegmentation(root=root, year='2012', image_set='train',
                               download=True,  transform=train_img_tf, target_transform=train_mask_tf)
    val_ds   = VOCSegmentation(root=root, year='2012', image_set='val',
                               download=False, transform=val_img_tf,   target_transform=val_mask_tf)
    train_loader = DataLoader(train_ds, batch_size, True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, False, num_workers=4, pin_memory=True)

    model = UNetPlusPlus(in_channels=3, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ce_loss   = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader) // accum_iter
    )
    scaler    = torch.amp.GradScaler(device='cuda', enabled=use_amp)
    miou_fn   = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)

    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for it, (imgs, masks) in enumerate(train_loader, 1):
            imgs   = imgs.to(device, non_blocking=True)
            masks  = masks.squeeze(1).long().to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(imgs)
                loss   = ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

            scaler.scale(loss).backward()
            if it % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_amp):
            for imgs, masks in val_loader:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.squeeze(1).long().to(device, non_blocking=True)

                logits = model(imgs)
                val_loss += (ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)).item() * imgs.size(0)

                preds = torch.argmax(logits, 1)
                val_iou += miou_fn(preds, masks) * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_miou  = val_iou / len(val_loader.dataset)

        print(f"[{epoch:02}/{epochs}] "
              f"train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"mIoU: {val_miou:.3f}  "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou
            }, os.path.join(out_dir, f'checkpoint_epoch{epoch}.pt'))

    torch.save(model.state_dict(), 'unetpp_voc2012_from_scratch.pt')
    print("Training completed, model saved to unetpp_voc2012_from_scratch.pt")

if __name__ == '__main__':
    main()