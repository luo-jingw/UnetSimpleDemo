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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3.1 Read and convert to RGB
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Received image, size: {img.size}")
    
    # Store original image dimensions
    original_width, original_height = img.size
    
    # 3.2 Preprocess + add batch dimension
    x = preprocess(img).unsqueeze(0).to(device)    # [1,3,512,512]
    print(f"Input tensor shape: {x.shape}")
    
    # 3.3 Inference
    with torch.no_grad():
        logits = model(x)                          # [1,21,512,512]
        print(f"Output logits shape: {logits.shape}")
        
        # Get probabilities with softmax
        probs = F.softmax(logits, dim=1)
        
        # Get predicted class and confidence score
        confidence, mask = torch.max(probs, dim=1)
        
        # Extract as numpy arrays
        mask = mask.squeeze(0).cpu().numpy()  # [512,512]
        confidence = confidence.squeeze(0).cpu().numpy()  # [512,512]
        
        print(f"Class distribution: {np.unique(mask).tolist()}")
    
    # 3.4 Return mask data and original image dimensions
    flat_mask = mask.flatten().tolist()
    flat_confidence = confidence.flatten().tolist()
    
    # Calculate average confidence per class
    class_confidences = {}
    for cls in np.unique(mask):
        class_confidences[int(cls)] = float(np.mean(confidence[mask == cls]))
    
    print(f"Returned mask length: {len(flat_mask)}, value range: {min(flat_mask)} - {max(flat_mask)}")
    return {
        "mask": flat_mask,
        "confidence": flat_confidence,
        "class_confidences": class_confidences,
        "width": original_width,
        "height": original_height,
        "model_width": 512,
        "model_height": 512
    }

# Add debug endpoint to backend.py
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
