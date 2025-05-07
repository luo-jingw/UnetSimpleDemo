# backend.py
import io
import base64
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp
import cv2

# 从DatasetVoc2012导入颜色映射
from DatasetVoc2012 import VOC_COLORMAP, VOC_CLASSES

# 1. Create FastAPI application and enable CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Load custom model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用与训练时相同的模型结构
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,  # 不需要预训练权重，因为我们会加载训练好的权重
    in_channels=3,
    classes=21
)
try:
    weights = torch.load("checkpoints/unet_finetuned.pth", map_location=device)
    model.load_state_dict(weights)
    print("Model loaded successfully, weights contain the following layers:", list(weights.keys())[:5], "...")
except Exception as e:
    print(f"Model loading failed: {e}")
model.to(device)
model.eval()

# 3. 使用与训练时相同的预处理
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # 确保与训练时相同的尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 添加归一化
])

def decode_segmap(mask):
    """
    将分割掩码转换为RGB彩色图像
    mask: [H, W] tensor with class indices 0-20
    returns: [H, W, 3] numpy array, RGB color image
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # 将掩码转换为整数类型，确保没有中间值
    mask = mask.astype(np.int32)
    
    # 剪裁确保掩码值在有效范围内 (0-20)
    mask = np.clip(mask, 0, 20)
    
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(21):
        rgb[mask == cls] = VOC_COLORMAP[cls]
            
    return rgb

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3.1 Read and convert to RGB
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Received image, size: {img.size}")
    
    # Store original image dimensions
    original_width, original_height = img.size
    
    # 保存原始图像以便后续合并
    original_img_resized = img.resize((512, 512))
    original_img_np = np.array(original_img_resized)
    
    # 3.2 Preprocess + add batch dimension
    x = preprocess(img).unsqueeze(0).to(device)    # [1,3,512,512]
    print(f"Input tensor shape: {x.shape}")
    
    # 3.3 Inference
    with torch.no_grad():
        output = model(x)                          # [1,21,512,512]
        print(f"Output shape: {output.shape}")
        
        # 直接使用argmax获取预测的类别索引
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [512,512]
        
        # 获取每个类别的置信度
        probs = F.softmax(output, dim=1).squeeze(0)  # [21,512,512]
        
        print(f"Class distribution: {np.unique(mask).tolist()}")
    
    # 3.4 生成彩色掩码
    color_mask = decode_segmap(mask)
    
    # 3.5 生成叠加效果（合并）
    overlay = cv2.addWeighted(original_img_np, 0.7, color_mask, 0.3, 0)
    
    # 3.6 将图像转换为base64编码
    mask_img = Image.fromarray(color_mask)
    overlay_img = Image.fromarray(overlay)
    
    mask_buffer = io.BytesIO()
    mask_img.save(mask_buffer, format="PNG")
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
    
    overlay_buffer = io.BytesIO()
    overlay_img.save(overlay_buffer, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    # 3.7 准备返回的类别信息和置信度
    class_confidences = {}
    detected_classes = []
    
    for cls in np.unique(mask):
        if cls > 0:  # 忽略背景类
            cls_idx = int(cls)
            class_mask = (mask == cls_idx)
            if np.any(class_mask):
                class_confidence = float(probs[cls_idx][class_mask].mean().cpu())
                class_confidences[cls_idx] = class_confidence
                detected_classes.append({
                    "id": cls_idx,
                    "name": VOC_CLASSES[cls_idx],
                    "confidence": class_confidence,
                    "color": VOC_COLORMAP[cls_idx]
                })
    
    print(f"Detected classes: {len(detected_classes)}")
    
    return {
        "mask_base64": f"data:image/png;base64,{mask_base64}",
        "overlay_base64": f"data:image/png;base64,{overlay_base64}",
        "detected_classes": detected_classes,
        "width": original_width,
        "height": original_height
    }

# Add debug endpoint
@app.get("/test_model")
async def test_model():
    # Create test image (a solid color image)
    test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    x = preprocess(test_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        mask = torch.argmax(logits, dim=1).squeeze(0)
        
    # Validate if model predicts multiple classes
    unique_classes = mask.unique().tolist()
    
    return {
        "unique_classes": unique_classes,
        "logits_shape": list(logits.shape),
        "max_logit_values": torch.max(logits, dim=1)[0].mean().item()
    }

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
