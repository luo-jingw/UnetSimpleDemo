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

# Define model structure to ensure compatibility when loading weights
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
        
        self.conv0_2 = ConvBlock(64 + 64 + 128, 64)
        self.conv1_2 = ConvBlock(128 + 128 + 256, 128)
        
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
model = UNetPlusPlus(in_channels=3, num_classes=21)
try:
    # Add weights_only=True to eliminate warnings
    weights = torch.load("checkpoints/best.pt", map_location=device, weights_only=True)
    model.load_state_dict(weights)
    print("Model loaded successfully, weights contain the following layers:", list(weights.keys())[:5], "...")
except Exception as e:
    print(f"Model loading failed: {e}")
model.to(device)
model.eval()

# 3. Preprocessing consistent with training
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Ensure the same size as used in training
    transforms.ToTensor(),
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
