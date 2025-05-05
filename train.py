# train_unetpp_voc.py
# ---------------------------------------------------------
# UNet++ on PASCAL‑VOC 2012 语义分割 (改进版)
# 关键改动：
#   1.  LR = 1e‑2 + 5 epoch linear warm‑up → cosine
#   2.  BatchNorm → GroupNorm(8) (bs = 8 更稳定)
#   3.  CE 按 VOC 像素统计加权，DiceLoss 保留
#   4.  深监督：x0_1 / x0_2 / x0_3 各接 1×1 aux‑head
#   5.  AMP API 更新：torch.amp.*
# ---------------------------------------------------------
import os, random, torch, numpy as np
from PIL import Image
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

torch.backends.cudnn.benchmark = True

# ------------------------------ Blocks ------------------------------
def gn(ch):        # GroupNorm wrapper
    return nn.GroupNorm(8, ch)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

# ------------------------------ UNet++ ------------------------------
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        ch = [64,128,256,512]

        self.enc = nn.ModuleList([
            ConvBlock(in_channels, ch[0]),
            ConvBlock(ch[0],      ch[1]),
            ConvBlock(ch[1],      ch[2]),
            ConvBlock(ch[2],      ch[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.dec01 = ConvBlock(ch[0]+ch[1],   ch[0])
        self.dec11 = ConvBlock(ch[1]+ch[2],   ch[1])
        self.dec21 = ConvBlock(ch[2]+ch[3],   ch[2])

        self.dec02 = ConvBlock(ch[0]*2+ch[1], ch[0])
        self.dec12 = ConvBlock(ch[1]*2+ch[2], ch[1])

        self.dec03 = ConvBlock(ch[0]*3+ch[1], ch[0])

        # heads
        self.head0 = nn.Conv2d(ch[0], num_classes, 1)   # main (x0_3)
        self.head1 = nn.Conv2d(ch[0], num_classes, 1)   # aux  (x0_2)
        self.head2 = nn.Conv2d(ch[0], num_classes, 1)   # aux  (x0_1)

    def forward(self, x):
        e0 = self.enc[0](x)
        e1 = self.enc[1](self.pool(e0))
        e2 = self.enc[2](self.pool(e1))
        e3 = self.enc[3](self.pool(e2))

        d01 = self.dec01(torch.cat([e0, self.up(e1)], 1))
        d11 = self.dec11(torch.cat([e1, self.up(e2)], 1))
        d21 = self.dec21(torch.cat([e2, self.up(e3)], 1))

        d02 = self.dec02(torch.cat([e0, d01, self.up(d11)], 1))
        d12 = self.dec12(torch.cat([e1, d11, self.up(d21)], 1))

        d03 = self.dec03(torch.cat([e0, d01, d02, self.up(d12)], 1))

        out0 = self.head0(d03)         # H/1
        out1 = self.head1(d02)         # deep‑sup
        out2 = self.head2(d01)
        return out0, out1, out2

# ------------------------------ Losses ------------------------------
def dice_loss(logits, targets, eps=1e-6, weight=None):
    C = logits.shape[1]
    p = torch.softmax(logits, 1)
    y = F.one_hot(targets.clamp(0, C-1), C).permute(0,3,1,2).float()
    inter = (p*y).sum((0,2,3))
    union = p.sum((0,2,3)) + y.sum((0,2,3))
    if weight is None: weight = torch.ones_like(inter)
    return (1 - (2*inter + eps)/(union + eps) * weight).mean()

# ------------------------------ Utils ------------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------------ Main ------------------------------
def main():
    seed_all()
    root, epochs, bs = 'voc_data', 120, 8
    base_lr, n_cls, out_dir = 1e-2, 21, 'checkpoints'
    accum, clip, use_amp = 4, 1.0, True
    os.makedirs(out_dir, exist_ok=True)

    tf_common = dict(size=512, scale=(0.5,2.0))
    img_tf = transforms.Compose([
        transforms.RandomResizedCrop(**tf_common),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    msk_tf = transforms.Compose([
        transforms.RandomResizedCrop(interpolation=Image.NEAREST, **tf_common),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()])
    val_img_tf = transforms.Compose([
        transforms.Resize(544), transforms.CenterCrop(512), transforms.ToTensor()])
    val_msk_tf = transforms.Compose([
        transforms.Resize(544, interpolation=Image.NEAREST),
        transforms.CenterCrop(512), transforms.PILToTensor()])

    train_ds = VOCSegmentation(root, '2012', 'train', True, img_tf, msk_tf)
    val_ds   = VOCSegmentation(root, '2012', 'val',   False, val_img_tf, val_msk_tf)

    train_loader = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=4,               
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=4,                
    pin_memory=True,
    persistent_workers=True
    )

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetPlusPlus(num_classes=n_cls).to(dev)

    # ---------- class weights ----------
    # 像素频率统计值（预先离线算好，顺序与 VOC label 对应，最后一位背景）
    cls_freq = torch.tensor([
        0.048,0.007,0.010,0.009,0.008,0.005,0.012,0.004,0.003,0.007,
        0.002,0.009,0.005,0.004,0.002,0.003,0.003,0.004,0.002,0.002,
        0.859                             # background
    ])
    ce_weight = 1 / torch.log1p(1.02*cls_freq)

    loss_ce = nn.CrossEntropyLoss(ignore_index=255,
                                  weight=ce_weight.to(dev))

    opt = torch.optim.SGD(net.parameters(), base_lr,
                          momentum=0.9, weight_decay=1e-4)

    warm  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    cos   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs-5, eta_min=1e-5)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [5])

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    miou   = MulticlassJaccardIndex(num_classes=n_cls, ignore_index=255).to(dev)

    best = 0.
    for ep in range(1, epochs+1):
        net.train(); opt.zero_grad(set_to_none=True); run_loss = 0.
        for it,(img,mask) in enumerate(train_loader,1):
            img = img.to(dev,non_blocking=True)
            mask= mask.squeeze(1).long().to(dev,non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                o0,o1,o2 = net(img)
                main = loss_ce(o0,mask)+0.5*dice_loss(o0,mask)
                aux1 = loss_ce(o1,mask)
                aux2 = loss_ce(o2,mask)
                loss = main + 0.4*aux1 + 0.2*aux2

            scaler.scale(loss/accum).backward()
            if it%accum==0 or it==len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run_loss += loss.item()*img.size(0)

        sched.step()
        train_loss = run_loss/len(train_loader.dataset)

        # ---------------- validation ----------------
        net.eval(); v_loss=0.; v_iou=0.
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16,
                                                 enabled=use_amp):
            for img,mask in val_loader:
                img = img.to(dev,non_blocking=True)
                mask= mask.squeeze(1).long().to(dev,non_blocking=True)
                o0,_,_ = net(img)
                v_loss += (loss_ce(o0,mask)+0.5*dice_loss(o0,mask)).item()*img.size(0)
                v_iou  += miou(o0.argmax(1),mask)*img.size(0)
        v_loss/=len(val_loader.dataset); v_miou=v_iou/len(val_loader.dataset)

        print(f"[{ep:03}/{epochs}]"
              f" train {train_loss:.3f}  val {v_loss:.3f}"
              f"  mIoU {v_miou:.3f}  lr {sched.get_last_lr()[0]:.5f}")

        torch.save({'epoch':ep,'model':net.state_dict()}, os.path.join(out_dir,'latest.pt'))
        if v_miou>best:
            best=v_miou
            torch.save(net.state_dict(), os.path.join(out_dir,'best.pt'))

    torch.save(net.state_dict(), 'unetpp_voc2012_final.pt')
    print(f"Finished; best mIoU={best:.3f}")

if __name__ == "__main__":
    main()
