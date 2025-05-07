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

# Import color mapping from DatasetVoc2012
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
# Use the same model structure as during training
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,  # We don't need pretrained weights as we'll load our trained weights
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

# 3. Preprocessing consistent with training
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Same size as used during training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization
])

def decode_segmap(mask):
    """
    Convert segmentation mask to RGB color image
    mask: [H, W] tensor with class indices 0-20
    returns: [H, W, 3] numpy array, RGB color image
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # Convert mask to integer type, ensure no intermediate values
    mask = mask.astype(np.int32)
    
    # Clip to ensure mask values are in valid range (0-20)
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
    
    # Save original image for later merging
    original_img = np.array(img)
    
    # 3.2 Preprocess + add batch dimension
    x = preprocess(img).unsqueeze(0).to(device)    # [1,3,512,512]
    print(f"Input tensor shape: {x.shape}")
    
    # 3.3 Inference
    with torch.no_grad():
        output = model(x)                          # [1,21,512,512]
        print(f"Output shape: {output.shape}")
        
        # Get predicted classes using argmax
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [512,512]
        
        # Get class confidence scores
        probs = F.softmax(output, dim=1).squeeze(0)  # [21,512,512]
        
        print(f"Class distribution: {np.unique(mask).tolist()}")
    
    # 3.4 Generate color mask
    color_mask = decode_segmap(mask)  # [512,512,3]
    
    # 3.5 Resize mask back to original image size
    color_mask_resized = cv2.resize(color_mask, (original_width, original_height), 
                                   interpolation=cv2.INTER_NEAREST)
    
    # 3.6 Generate overlay effect
    overlay = cv2.addWeighted(original_img, 0.7, color_mask_resized, 0.3, 0)
    
    # 3.7 Convert images to base64 encoding
    original_pil = Image.fromarray(original_img)
    mask_pil = Image.fromarray(color_mask_resized)
    overlay_pil = Image.fromarray(overlay)
    
    # Create byte buffers for each image
    original_buffer = io.BytesIO()
    original_pil.save(original_buffer, format="PNG")
    original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
    
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format="PNG")
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
    
    overlay_buffer = io.BytesIO()
    overlay_pil.save(overlay_buffer, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    # 3.8 Prepare class information and confidence scores
    detected_classes = []
    
    # Resize the prediction mask to original image size for calculating correct class stats
    mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    for cls in np.unique(mask_resized):
        if cls > 0:  # Ignore background class
            cls_idx = int(cls)
            
            # We need to calculate confidence on the original prediction (before resize)
            # Create a binary mask for this class
            class_mask_512 = (mask == cls_idx)
            
            if np.any(class_mask_512):
                # Calculate confidence from the probability tensor (still at 512x512)
                class_confidence = float(probs[cls_idx][class_mask_512].mean().cpu())
                
                # Add class to results
                detected_classes.append({
                    "id": cls_idx,
                    "name": VOC_CLASSES[cls_idx],
                    "confidence": class_confidence,
                    "color": VOC_COLORMAP[cls_idx]
                })
    
    print(f"Detected classes: {len(detected_classes)}")
    
    return {
        "original_base64": f"data:image/png;base64,{original_base64}",
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
