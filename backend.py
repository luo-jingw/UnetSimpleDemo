# backend.py
import io
import base64
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import tempfile
import os
from starlette.responses import JSONResponse
import json
import subprocess

# Import color mapping from DatasetVoc2012
from DatasetVoc2012 import VOC_COLORMAP, VOC_CLASSES

# Helper function to ensure return is a Python list
def ensure_list(value):
    """Ensure the return value is a Python list, regardless of input type"""
    if hasattr(value, 'tolist'):  # Check if it's a numpy array or similar object
        return value.tolist()
    return value

# Store active WebSocket connections
active_connections = {}

# 1. Create FastAPI application and enable CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket connection
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            # Keep connection open
            data = await websocket.receive_text()
            if data == "close":
                break
    except:
        pass
    finally:
        if client_id in active_connections:
            del active_connections[client_id]
            
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
    
    # 3.6 Generate overlay effect with updated transparency (0.6 for original image, 0.4 for mask)
    overlay = cv2.addWeighted(original_img, 0.6, color_mask_resized, 0.4, 0)
    
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

# Video processing endpoint
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    # Extract client ID (if present in request headers)
    client_id = None
    request_headers = getattr(file, "headers", {})
    if "x-client-id" in request_headers:
        client_id = request_headers["x-client-id"]
        print(f"Received client ID: {client_id}")
    
    # Create temporary directory for video frames and results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded video file
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(await file.read())
        
        # Use OpenCV to read the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Cannot open video file"})
        
        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check and use FFmpeg for video transcoding (ensure browser compatibility)
        output_path = os.path.join(temp_dir, "segmented_video_raw.mp4")
        final_output_path = os.path.join(temp_dir, "segmented_video.mp4")
        
        # Use appropriate encoder
        encoders = check_encoder_availability()
        if encoders.get("libx264", False):
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif encoders.get("mpeg4", False):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            # If VideoWriter creation fails, use MJPEG format (almost always available)
            output_path = os.path.join(temp_dir, "segmented_video_raw.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                return JSONResponse(status_code=500, content={"error": "Cannot create video writer"})
        
        # Process each frame
        frames_processed = 0
        class_stats = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to PIL image for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Preprocess
            x = preprocess(pil_img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(x)
                mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                probs = F.softmax(output, dim=1).squeeze(0)
            
            # Generate color mask
            color_mask = decode_segmap(mask)
            
            # Resize back to original dimensions
            color_mask_resized = cv2.resize(color_mask, (frame_width, frame_height), 
                                           interpolation=cv2.INTER_NEAREST)
            
            # Create overlay effect with updated transparency (0.6 for original image, 0.4 for mask)
            overlay = cv2.addWeighted(frame_rgb, 0.6, color_mask_resized, 0.4, 0)
            
            # Write to output video (convert back to BGR format)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            out.write(overlay_bgr)
            
            # Accumulate class statistics
            mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            for cls in np.unique(mask_resized):
                if cls > 0:  # Ignore background class
                    cls_idx = int(cls)
                    class_mask = (mask == cls_idx)
                    
                    if np.any(class_mask):
                        # Calculate confidence for this class
                        class_confidence = float(probs[cls_idx][class_mask].mean().cpu())
                        
                        # Update class statistics
                        if cls_idx not in class_stats:
                            class_stats[cls_idx] = {
                                "id": cls_idx,
                                "name": VOC_CLASSES[cls_idx],
                                "color": ensure_list(VOC_COLORMAP[cls_idx]),
                                "frames": 0,
                                "avg_confidence": 0
                            }
                        
                        # Update frame count and average confidence
                        current = class_stats[cls_idx]
                        current["frames"] += 1
                        current["avg_confidence"] = (current["avg_confidence"] * (current["frames"] - 1) + 
                                                     class_confidence) / current["frames"]
            
            frames_processed += 1
            
            # Update progress every 5 frames
            if frames_processed % 5 == 0 or frames_processed == total_frames:
                progress = (frames_processed / total_frames) * 100
                print(f"Processed {frames_processed}/{total_frames} frames ({progress:.1f}%)")
                
                # Send progress updates via WebSocket
                if client_id and client_id in active_connections:
                    try:
                        await active_connections[client_id].send_json({
                            "type": "progress",
                            "frames_processed": frames_processed,
                            "total_frames": total_frames,
                            "progress": progress
                        })
                    except Exception as e:
                        print(f"WebSocket send failed: {e}")
        
        cap.release()
        out.release()
        
        # Use FFmpeg to re-encode video for browser compatibility
        try:
            if os.path.exists(output_path):
                # Use FFmpeg to optimize video for browser playback
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", output_path, 
                    "-c:v", "libx264", "-preset", "fast", 
                    "-profile:v", "baseline", "-level", "3.0",
                    "-pix_fmt", "yuv420p", final_output_path
                ]
                
                subprocess.run(ffmpeg_cmd, check=True)
                print("FFmpeg successfully transcoded video!")
                
                # If FFmpeg succeeded, use the transcoded video
                if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
                    output_path = final_output_path
        except Exception as e:
            print(f"FFmpeg transcoding failed: {e}")
            # If FFmpeg fails, continue with original video
            final_output_path = output_path
        
        # Read result video and convert to base64
        try:
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                
            # Sort class statistics (by frame count in descending order)
            classes_list = list(class_stats.values())
            classes_list.sort(key=lambda x: x["frames"], reverse=True)
            
            # Return video and analysis results
            return {
                "video_base64": f"data:video/mp4;base64,{video_base64}",
                "detected_classes": classes_list,
                "total_frames": total_frames,
                "fps": fps,
                "width": frame_width,
                "height": frame_height
            }
        except Exception as e:
            print(f"Video processing final stage failed: {e}")
            return JSONResponse(status_code=500, content={"error": f"Video processing failed: {str(e)}"})

# Check video encoder availability
def check_encoder_availability():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        output = result.stdout
        encoders = {
            "libx264": "libx264" in output,
            "h264": "h264" in output,
            "mpeg4": "mpeg4" in output,
            "libvpx": "libvpx" in output,
            "libvpx-vp9": "libvpx-vp9" in output,
            "libtheora": "libtheora" in output,
            "mjpeg": "mjpeg" in output
        }
        print(f"Available encoders: {[k for k, v in encoders.items() if v]}")
        return encoders
    except:
        print("Unable to check encoder availability")
        return {
            "libx264": False,
            "h264": False,
            "mpeg4": True,  # Assume MPEG4 is at least available
            "libvpx": False,
            "libvpx-vp9": False,
            "libtheora": False,
            "mjpeg": True  # MJPEG is usually available
        }
