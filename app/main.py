import math
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Use a relative import to correctly locate the detector module inside the app package
from . import detector

# --- 1. App Initialization ---
# Create the main FastAPI application instance
app = FastAPI(title="Tactical Scanner API")


# --- 2. CORS Middleware ---
# Allow Cross-Origin Resource Sharing for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- 3. Pydantic Model for Trajectory Input ---
# Defines the expected JSON structure for the /api/trajectory endpoint
class TrajectoryInput(BaseModel):
    target_distance: float
    velocity: float = 120.0


# --- 4. API Endpoints ---

@app.post("/api/analyze")
async def analyze_image(scope: str = Form(...), file: UploadFile = File(...)):
    """
    Receives an image and a 'scope' string.
    It passes the image data to the detector for object prediction.
    - scope: A string parameter (e.g., "all", "vehicles").
    - file: The uploaded image file.
    """
    # Read the contents of the uploaded file
    image_data = await file.read()
    
    # Call the prediction function from the detector module
    # This assumes detector.py has a function get_object_predictions
    predictions = detector.get_object_predictions(image_data, scope)
    
    return {"predictions": predictions}


@app.post("/api/trajectory")
async def get_trajectory(data: TrajectoryInput):
    """
    Calculates the launch angle and path for a projectile to hit a target.
    Receives target_distance and velocity in a JSON object.
    """
    g = 9.81  # Acceleration due to gravity (m/s^2)
    v = data.velocity
    d = data.target_distance

    # Check if the target is physically reachable with the given velocity
    # The argument of arcsin must be between -1 and 1
    val = (g * d) / (v**2)
    if not -1 <= val <= 1:
        raise HTTPException(
            status_code=400, 
            detail="Target is out of range for the given velocity."
        )

    # Calculate the launch angle for minimum energy trajectory
    angle_rad = 0.5 * math.asin(val)
    launch_angle_deg = math.degrees(angle_rad)
    time_of_flight = (2 * v * math.sin(angle_rad)) / g

    # Generate points along the trajectory path
    path_points = []
    for t in np.linspace(0, time_of_flight, num=50):
        x = v * math.cos(angle_rad) * t
        y = v * math.sin(angle_rad) * t - 0.5 * g * (t**2)
        # Only include points where the projectile is above ground
        if y >= 0:
            path_points.append({"x": round(x, 2), "y": round(y, 2)})

    return {
        "path": path_points,
        "calculated_angle": round(launch_angle_deg, 2),
        "time_of_flight": round(time_of_flight, 2)
    }

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "Tactical Scanner API is running"}
