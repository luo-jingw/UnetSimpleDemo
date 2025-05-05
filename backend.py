# backend.py
import io
import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# 1. 创建 FastAPI 应用并开启 CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# 2. 加载微调后的完整模型（启用 weights_only=False）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "unetpp_voc2012_finetuned.pt",
    map_location=device,
    weights_only=False           # ← 关键：关闭“仅权重”模式
)
model.to(device)
model.eval()

# 3. 与训练一致的预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3.1 读取并转换为 RGB
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 3.2 预处理 + 扩 batch 维度
    x = preprocess(img).unsqueeze(0).to(device)    # [1,3,256,256]
    # 3.3 推理
    with torch.no_grad():
        logits = model(x)                          # [1,21,256,256]
        mask = torch.argmax(logits, dim=1).squeeze(0)  # [256,256]
    # 3.4 扁平化并返回
    flat = mask.cpu().numpy().flatten().tolist()
    return {"mask": flat}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
