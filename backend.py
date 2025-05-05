# backend.py
import io
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# 定义相同的模型类，确保加载时结构匹配
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

# 1. 创建 FastAPI 应用并开启 CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# 2. 加载自定义模型的权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPlusPlus(in_channels=3, num_classes=21)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
model.to(device)
model.eval()

# 3. 与训练一致的预处理
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # 确保与训练使用相同尺寸
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3.1 读取并转换为 RGB
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 3.2 预处理 + 扩 batch 维度
    x = preprocess(img).unsqueeze(0).to(device)    # [1,3,512,512]
    # 3.3 推理
    with torch.no_grad():
        logits = model(x)                          # [1,21,512,512]
        mask = torch.argmax(logits, dim=1).squeeze(0)  # [512,512]
    # 3.4 扁平化并返回
    flat = mask.cpu().numpy().flatten().tolist()
    return {"mask": flat}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
