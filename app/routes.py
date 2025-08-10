import io
from PIL import Image
from fastapi import APIRouter, File, UploadFile
from .models import object_detector # Import the model

# Create a router to organize endpoints
router = APIRouter()

@router.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Receives an image file, performs object detection, and returns the results.
    """
    # Read the image file uploaded by the user
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Perform object detection using the imported model
    predictions = object_detector(image)

    # Format the predictions to match what the frontend expects
    formatted_predictions = []
    for pred in predictions:
        box = pred['box']
        formatted_pred = {
            "class": pred['label'],
            "score": pred['score'],
            "bbox": [
                box['xmin'],
                box['ymin'],
                box['xmax'] - box['xmin'],
                box['ymax'] - box['ymin']
            ],
            "weapon_type": "Unclassified",
            "distance": 0 
        }
        formatted_predictions.append(formatted_pred)

    return {"predictions": formatted_predictions}
