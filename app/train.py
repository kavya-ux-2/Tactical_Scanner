from ultralytics import YOLO
import torch

def train_custom_model():
    """
    Trains a YOLOv8 model on the custom FLIR dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model = YOLO('yolov8x.pt')

    print("Starting model training on FLIR dataset...")
    results = model.train(
        data='flir_dataset/flir_data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=device
    )
    
    print("Training complete.")
    print("Model and results saved in the 'runs' directory.")
    # The best model will be at 'runs/detect/train/weights/best.pt'

if __name__ == "__main__":
    train_custom_model()
