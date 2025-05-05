import os, torch
import numpy as np
from PIL import Image
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

# ------------------------------ utils ------------------------------ #
def dice_loss(logits, targets, eps: float = 1e-6):
    """Soft‑Dice across all classes, targets ∈ [B,H,W] int"""
    C = logits.shape[1]
    probs  = F.softmax(logits, 1)
    onehot = F.one_hot(targets.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    inter  = (probs * onehot).sum((0, 2, 3))
    union  = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()

# ---------------------------- main script -------------------------- #
def main():
    # ---------- Config ----------
    root, epochs, batch_size = 'voc_data', 100, 4
    lr, num_classes, out_dir = 1e-3, 21, 'checkpoints'
    accum_iter, use_amp = 1, True
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Transforms ----------
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
    # 验证集：确定性
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

    # ---------- Dataset ----------
    train_ds = VOCSegmentation(root=root, year='2012', image_set='train',
                               download=True,  transform=train_img_tf, target_transform=train_mask_tf)
    val_ds   = VOCSegmentation(root=root, year='2012', image_set='val',
                               download=False, transform=val_img_tf,   target_transform=val_mask_tf)
    train_loader = DataLoader(train_ds, batch_size, True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, False, num_workers=4, pin_memory=True)

    # ---------- Model ----------
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet',
                             in_channels=3, classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 冻结 encoder BN 统计
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval().requires_grad_(False)

    # ---------- Loss / Optim / LR ----------
    ce_loss   = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(device='cuda', enabled=use_amp)
    miou_fn   = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)

    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        # ======= train =======
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for it, (imgs, masks) in enumerate(train_loader, 1):
            imgs   = imgs.to(device, non_blocking=True)
            masks  = masks.squeeze(1).long().to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(imgs)
                loss   = ce_loss(logits, masks) + 0.4 * dice_loss(logits, masks)

            scaler.scale(loss).backward()
            if it % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # ======= validate =======
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_amp):
            for imgs, masks in val_loader:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.squeeze(1).long().to(device, non_blocking=True)

                logits = model(imgs)
                val_loss += (ce_loss(logits, masks) + 0.4 * dice_loss(logits, masks)).item() * imgs.size(0)

                preds = torch.argmax(logits, 1)
                val_iou += miou_fn(preds, masks) * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_miou  = val_iou / len(val_loader.dataset)

        print(f"[{epoch:02}/{epochs}] "
              f"train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"mIoU: {val_miou:.3f}  "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # ---------- checkpoint ----------
        torch.save(model, os.path.join(out_dir, f'epoch{epoch}.pt'))
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model, os.path.join(out_dir, 'best.pt'))

    # ---------- final ----------
    torch.save(model, 'unetpp_voc2012_finetuned.pt')
    print("✔ 训练完成，模型保存到 unetpp_voc2012_finetuned.pt")

if __name__ == '__main__':
    main()
