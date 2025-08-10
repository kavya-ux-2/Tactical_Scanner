from ultralytics import YOLO
from PIL import Image
import io
import torch
import cv2
import numpy as np

# --- 1. ADVANCED AI MODEL ---
print("Loading Advanced AI Model (YOLOv8x)...")
model = YOLO('yolov8x.pt')
print("Model loaded.")

# --- 2. ADVANCED THREAT CLASSIFICATION ---
THREAT_CLASSIFICATION = {
    "airplane": {"type": "Aircraft (Hostile)", "level": "HIGH"},
    "helicopter": {"type": "Rotorcraft (Hostile)", "level": "HIGH"},
    "truck": {"type": "Military Vehicle", "level": "MEDIUM"},
    "boat": {"type": "Naval Vessel", "level": "MEDIUM"},
    "bus": {"type": "Troop Transport", "level": "MEDIUM"},
    "person": {"type": "Infantry", "level": "LOW"},
    "bicycle": {"type": "Scout", "level": "LOW"},
    "motorcycle": {"type": "Scout", "level": "LOW"},
    "car": {"type": "Civilian Vehicle", "level": "LOW"},
    "dog": {"type": "K9 Unit", "level": "INFO"},
    "cat": {"type": "Animal", "level": "INFO"},
}

# --- MiDaS Model Initialization ---
try:
    midas_model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    MIDAS_LOADED = True
    print(f"MiDaS model loaded successfully on device: {device}")
except Exception as e:
    print(f"Could not load MiDaS model. Distance calculation disabled. Error: {e}")
    MIDAS_LOADED = False

def get_object_predictions(image_data: bytes, scope: str) -> list:
    pil_image = Image.open(io.BytesIO(image_data))
    print(f"Analyzing with {scope} scope...")
    results = model(pil_image)
    predictions = []

    depth_map = None
    if MIDAS_LOADED:
        cv2_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            input_batch = transform(img).to(device)
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_name = model.names[int(box.cls[0].item())]
            distance = -1.0
            if depth_map is not None:
                x1, y1, x2, y2 = map(int, xyxy)
                object_depth_map = depth_map[y1:y2, x1:x2]
                if object_depth_map.size > 0:
                    distance = float(np.median(object_depth_map))
            
            threat_info = THREAT_CLASSIFICATION.get(class_name, {"type": "Unclassified", "level": "NONE"})
            predictions.append({
                "class": class_name,
                "score": confidence,
                "bbox": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                "distance": distance,
                "threat_type": threat_info["type"],
                "threat_level": threat_info["level"]
            })
    return predictions