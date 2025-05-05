#!/usr/bin/env python
# test_unetpp_voc.py
import os, argparse, torch, importlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from pickle import UnpicklingError

from train import UNetPlusPlus   # ← 你的训练脚本

# ---------- Palette ----------
VOC_PALETTE = np.array([
    (0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),
    (128,0,128),(0,128,128),(128,128,128),(64,0,0),(192,0,0),
    (64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),
    (192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
], dtype=np.uint8)

def colorize(m): return VOC_PALETTE[m]

def save_vis(img_t, gt, pred, path):
    img = (img_t.numpy().transpose(1,2,0)*255).astype(np.uint8)
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    for ax,d,t in zip(axs,[img,colorize(gt),colorize(pred)],['Image','GT','Pred']):
        ax.imshow(d); ax.set_title(t); ax.axis('off')
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------- Loader ----------
def build_loader(split, bs, *, force_size=None):
    if force_size:
        img_tf = transforms.Compose([
            transforms.Resize(force_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(force_size),
            transforms.ToTensor()])
        mask_tf = transforms.Compose([
            transforms.Resize(force_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(force_size),
            transforms.PILToTensor()])
    else:
        img_tf, mask_tf = transforms.ToTensor(), transforms.PILToTensor()

    ds = VOCSegmentation('voc_data', year='2012', image_set=split,
                         download=(split=='train'),
                         transform=img_tf, target_transform=mask_tf)
    return DataLoader(ds, bs, shuffle=False, num_workers=4, pin_memory=True)

# ---------- Safe checkpoint load ----------
def safe_load(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        return ckpt['model'] if isinstance(ckpt,dict) and 'model' in ckpt else ckpt
    except UnpicklingError:
        smp_mod = importlib.import_module(
            'segmentation_models_pytorch.decoders.unetplusplus.model')
        torch.serialization.add_safe_globals({smp_mod.UnetPlusPlus})
        full = torch.load(path, map_location=device, weights_only=False)
        if isinstance(full, dict) and 'model' in full: return full['model']
        if isinstance(full, torch.nn.Module):          return full.state_dict()
        raise RuntimeError("Unsupported checkpoint format")

# ---------- Evaluate ----------
def evaluate(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetPlusPlus(num_classes=21).to(dev)
    net.load_state_dict(safe_load(args.weights, dev), strict=False)
    net.eval()

    miou = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(dev)
    pix  = MulticlassAccuracy(num_classes=21, average='micro', ignore_index=255).to(dev)

    loader = build_loader('val', args.batch_size, force_size=512)   # ★ FIX
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):  # ★ FIX
        for i,(img,mask) in enumerate(loader,1):
            img = img.to(dev); mask = mask.squeeze(1).long().to(dev)
            pred = net(img).argmax(1)
            miou.update(pred,mask); pix.update(pred,mask)

            if args.vis_every and i%args.vis_every==0:
                os.makedirs(args.out_dir, exist_ok=True)
                for k in range(min(args.save_vis, img.size(0))):
                    save_vis(img[k].cpu(), mask[k].cpu().numpy(),
                             pred[k].cpu().numpy(),
                             os.path.join(args.out_dir,f'vis_{i}_{k}.png'))
    print(f"\nValidation  mIoU: {miou.compute():.3f}   PixelAcc: {pix.compute():.3f}")

# ---------- Sanity ----------
def sanity_overfit(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetPlusPlus(num_classes=21).to(dev)
    img,mask = next(iter(build_loader('train',1,force_size=512)))
    img = img.to(dev); mask = mask.squeeze(1).long().to(dev)

    opt = torch.optim.Adam(net.parameters(),1e-3)
    ce  = torch.nn.CrossEntropyLoss(ignore_index=255)
    print(">>> sanity check")
    for it in range(1,101):
        net.train(); opt.zero_grad()
        logit = net(img); loss = ce(logit,mask); loss.backward(); opt.step()
        if it%10==0:
            pa = (logit.argmax(1)==mask).float().mean().item()
            print(f"[{it:03d}/100] loss={loss.item():.4f}  pixel_acc={pa:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    save_vis(img[0].cpu(), mask[0].cpu().numpy(),
             logit.argmax(1)[0].cpu().numpy(),
             os.path.join(args.out_dir,'sanity.png'))

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--vis-every", type=int, default=0)
    ap.add_argument("--save-vis", type=int, default=3)
    ap.add_argument("--out-dir", default="test_vis")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    if args.sanity: sanity_overfit(args)
    else:           evaluate(args)
