# backend.py
import io
import base64
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, WebSocket, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
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

# Function to fix image orientation based on EXIF data using PIL's built-in function
def fix_orientation(image: Image.Image) -> Image.Image:
    """
    Fix the orientation of an image based on EXIF data using PIL's built-in function
    """
    return ImageOps.exif_transpose(image)

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
    print(f"WebSocket connected for client: {client_id}")
    print(f"Active connections: {list(active_connections.keys())}")
    
    try:
        while True:
            # Keep connection open and listen for messages
            data = await websocket.receive_text()
            
            # 简化消息处理
            try:
                message = json.loads(data)
                print(f"Received message from client {client_id}: {message}")
            except json.JSONDecodeError:
                print(f"Received invalid message from client: {data}")
                
            if data == "close":
                break
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]
            print(f"WebSocket connection closed for client: {client_id}")
            print(f"Remaining active connections: {list(active_connections.keys())}")
            
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
    
    # Fix image orientation based on EXIF data
    img = fix_orientation(img)
    
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
    
    # 重要修改：确保叠加图与原图保持一致方向，先转为PIL图像再进行方向处理
    overlay_pil = Image.fromarray(overlay.astype('uint8'))
    # 注意：对于直接从numpy生成的PIL图像，不存在EXIF数据，但为了代码一致性仍保留此调用
    # 对于某些平台，此操作仍可能必要
    overlay_pil = fix_orientation(overlay_pil)
    overlay = np.array(overlay_pil)
    
    # 3.7 Collect results - detected classes with confidences
    detected_classes = []
    for class_idx in np.unique(mask):
        if class_idx > 0:  # Skip background class
            confidence = float(probs[class_idx].max().cpu().numpy())
            detected_classes.append({
                "name": VOC_CLASSES[class_idx],
                "confidence": confidence,
                "color": ensure_list(VOC_COLORMAP[class_idx])
            })
    
    # Sort by confidence 
    detected_classes.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 3.8 Convert images to base64 strings
    pil_img = img  # Already a PIL image
    pil_mask = Image.fromarray(color_mask_resized.astype('uint8'))
    pil_overlay = Image.fromarray(overlay.astype('uint8'))
    
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    mask_byte_arr = io.BytesIO()
    pil_mask.save(mask_byte_arr, format='PNG')
    mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
    
    overlay_byte_arr = io.BytesIO()
    pil_overlay.save(overlay_byte_arr, format='PNG')
    overlay_base64 = base64.b64encode(overlay_byte_arr.getvalue()).decode('utf-8')
    
    return {
        "original_base64": f"data:image/png;base64,{img_base64}",
        "mask_base64": f"data:image/png;base64,{mask_base64}",
        "overlay_base64": f"data:image/png;base64,{overlay_base64}",
        "detected_classes": detected_classes
    }

# Video processing endpoint
@app.post("/predict_video")
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    x_client_id: str = Header(None)  # ← 使用 Header 参数接收 X-Client-ID
):
    # 1. Try to get client ID for WebSocket communication (from header or form data)
    client_id = x_client_id
    
    # Extract client ID from form data as backup if header missing
    form = await request.form()
    form_client_id = form.get("client_id")
    if not client_id and form_client_id:
        client_id = form_client_id
        
    if client_id:
        print(f"Received client ID from header: {client_id}")
        
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded video to temp file
        temp_video_path = os.path.join(temp_dir, file.filename)
        with open(temp_video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Open video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return JSONResponse(status_code=500, content={"error": "Cannot open video file"})
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Prepare video writer - use the most compatible encoder
        available_encoders = check_encoder_availability()
        print(f"Available encoders: {available_encoders}")
        
        output_path = os.path.join(temp_dir, "segmented_video_raw.mp4")
        
        # Try different encoders in order of preference
        encoder_success = False
        
        # 1. First try H.264 (most compatible)
        if 'avc1' in available_encoders:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    encoder_success = True
                    print("Using H264 encoder")
            except Exception as e:
                print(f"H264 encoder failed: {e}")
        
        # 2. Try MPEG-4 if H264 failed
        if not encoder_success and 'mp4v' in available_encoders:
            try:
                output_path = os.path.join(temp_dir, "segmented_video_raw.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    encoder_success = True
                    print("Using MPEG-4 encoder")
            except Exception as e:
                print(f"MPEG-4 encoder failed: {e}")
        
        # 3. Try MJPEG as a reliable fallback
        if not encoder_success:
            try:
                output_path = os.path.join(temp_dir, "segmented_video_raw.avi")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    encoder_success = True
                    print("Using MJPEG encoder")
                else:
                    print("MJPEG encoder failed to open")
            except Exception as e:
                print(f"MJPEG encoder failed: {e}")
        
        if not encoder_success:
            return JSONResponse(status_code=500, content={"error": "Failed to initialize any video encoder"})
        
        # Process each frame
        frames_processed = 0
        class_stats = {}
        processing_aborted = False
        
        while cap.isOpened():
            # Check for abort request
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to PIL image for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Fix any orientation issues
            # Note: Video frames usually don't have EXIF data, but we include this for completeness
            pil_img = fix_orientation(pil_img)
            
            # Update frame_rgb in case the orientation was fixed
            frame_rgb = np.array(pil_img)
            
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
            
            # Create overlay
            overlay = cv2.addWeighted(frame_rgb, 0.6, color_mask_resized, 0.4, 0)
            
            # 重要修改：确保视频中的叠加图与原图保持一致方向
            overlay_pil = Image.fromarray(overlay.astype('uint8'))
            overlay_pil = fix_orientation(overlay_pil)
            overlay = np.array(overlay_pil)
            
            # Convert RGB back to BGR for OpenCV
            output_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(output_frame)
            
            # Track classes detected in this frame
            for class_idx in np.unique(mask):
                if class_idx > 0:  # Skip background
                    confidence = float(probs[class_idx].max().cpu().numpy())
                    class_name = VOC_CLASSES[class_idx]
                    class_color = VOC_COLORMAP[class_idx]
                    
                    if class_name not in class_stats:
                        class_stats[class_name] = {
                            "frames": 1,
                            "total_confidence": confidence,
                            "color": class_color
                        }
                    else:
                        class_stats[class_name]["frames"] += 1
                        class_stats[class_name]["total_confidence"] += confidence
            
            # Update progress
            frames_processed += 1
            progress = frames_processed / total_frames
            
            # Send progress update via WebSocket
            if client_id in active_connections:
                try:
                    await active_connections[client_id].send_text(json.dumps({
                        "progress": progress,
                        "frames_processed": frames_processed,
                        "total_frames": total_frames
                    }))
                except Exception as e:
                    print(f"Error sending WebSocket update: {e}")
            
        # Close video resources
        cap.release()
        out.release()
        
        if frames_processed < 3:  # Sanity check - at least 3 frames should be processed
            return JSONResponse(status_code=500, content={"error": "Video processing failed - too few frames processed"})
        
        # Optimize output video for web viewing
        web_output_path = os.path.join(temp_dir, "segmented_video_web.mp4")
        
        try:
            # Use ffmpeg for web-optimized output if available
            optimize_cmd = [
                'ffmpeg', '-i', output_path,
                '-vcodec', 'libx264', 
                '-crf', '23',  # Lower CRF = better quality (18-28 is a good range)
                '-preset', 'medium',  # faster = quicker encoding, slower = better compression
                '-movflags', 'faststart',  # Move MOOV atom to the beginning for fast start
                '-pix_fmt', 'yuv420p',  # Standard pixel format for web
                web_output_path
            ]
            
            process = subprocess.run(
                optimize_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                print(f"FFmpeg error: {process.stderr.decode()}")
                # Use original output if optimization fails
                web_output_path = output_path
        except Exception as e:
            print(f"Failed to optimize video: {e}")
            web_output_path = output_path
        
        # Convert to base64 for web embedding
        with open(web_output_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        # Format class stats
        detected_classes = []
        for class_name, stats in class_stats.items():
            avg_confidence = stats["total_confidence"] / stats["frames"]
            detected_classes.append({
                "name": class_name,
                "frames": stats["frames"],
                "avg_confidence": avg_confidence,
                "color": ensure_list(stats["color"])
            })
        
        # Sort by frequency
        detected_classes.sort(key=lambda x: x['frames'], reverse=True)
        
        return {
            "video_base64": f"data:video/mp4;base64,{video_base64}",
            "detected_classes": detected_classes,
            "total_frames": total_frames,
            "processed_frames": frames_processed
        }

# Check video encoder availability
def check_encoder_availability():
    try:
        available_encoders = []
        
        # Check common encoders with their fourcc codes
        codecs_to_check = [
            ('mjpeg', 'MJPG'),  # Motion JPEG - widely supported
            ('avc1', 'H264'),    # H.264
            ('mp4v', 'MP4V'),    # MPEG-4
            ('divx', 'DIVX'),    # DivX
            ('xvid', 'XVID'),    # Xvid
        ]
        
        for name, code in codecs_to_check:
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Test writing a small sample
                fourcc = cv2.VideoWriter_fourcc(*code)
                test = cv2.VideoWriter(
                    temp_path, 
                    fourcc,
                    30,  # fps
                    (320, 240),  # small resolution
                    True  # isColor
                )
                
                if test.isOpened():
                    # Create a small test frame
                    test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    test.write(test_frame)
                    test.release()
                    
                    # Verify file was created and has content
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        available_encoders.append(name)
                        print(f"Encoder {name} ({code}) is available and working")
                else:
                    print(f"Encoder {name} ({code}) failed to open")
                    
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error testing encoder {name}: {e}")
                continue
        
        # Always ensure at least MJPEG is available as fallback
        if not available_encoders:
            print("No working encoders detected, defaulting to MJPEG")
            available_encoders = ['mjpeg']
            
        return available_encoders
    except Exception as e:
        print(f"Error checking encoder availability: {e}")
        return ['mjpeg']  # Fallback to MJPEG in case of errors

# 启动 web 服务器
if __name__ == "__main__":
    print("Starting the web server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
