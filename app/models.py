import torch
from transformers import pipeline

print("Loading Object Detection model...")

# Check if a GPU is available and use it, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1

# Load the pre-trained object detection model from Hugging Face
object_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50",
    device=device
)

print("Model loaded successfully!")
