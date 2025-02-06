from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
import io
from PIL import Image
import tempfile
import os
from pathlib import Path

from try_on import create_body_mask, create_torso_mask, create_face_hair_mask, prepare_person_representation
from posedetector import PoseDetector
from gmm import load_gmm, tps_transform

app = FastAPI(title="Virtual Try-On API")

# Initialize models
pose_detector = PoseDetector(model_path='models/graph_opt.pb')
gmm_model = load_gmm('models/gmm_final.pth')

# Create output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """Convert uploaded file to numpy array."""
    # Read image file
    contents = file.file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

@app.post("/try-on/")
async def try_on(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
):
    """
    Try on a clothing item on a person image.
    
    Parameters:
    - person_image: Image of the person
    - clothing_image: Image of the clothing item
    
    Returns:
    - Image file with the clothing item warped onto the person
    """
    try:
        # Process uploaded images
        person_img = process_uploaded_image(person_image)
        shirt_img = process_uploaded_image(clothing_image)
        
        # Get pose points
        pose_points = pose_detector.detect(person_img)
        
        if not pose_points or None in [pose_points[i] for i in [2, 5, 8, 11]]:  # Check key points
            raise HTTPException(status_code=400, detail="Could not detect pose in the person image")
        
        # Create masks
        body_mask = create_body_mask(pose_points, person_img.shape[0], person_img.shape[1])
        torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
        face_hair_mask = create_face_hair_mask(pose_points, person_img)
        
        # Prepare person representation
        person_representation = prepare_person_representation(pose_points, torso_mask, face_hair_mask)
        
        # Prepare clothing image
        shirt_img = cv2.resize(shirt_img, (192, 256))
        shirt_tensor = torch.FloatTensor(shirt_img).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Run GMM model
        with torch.no_grad():
            theta = gmm_model(person_representation, shirt_tensor)
            warped_shirt = tps_transform(theta, shirt_tensor)
            
            # Convert warped shirt to numpy
            warped_shirt_np = warped_shirt.squeeze().cpu().numpy()
            warped_shirt_np = warped_shirt_np.transpose(1, 2, 0)
            warped_shirt_np = (warped_shirt_np * 255).astype(np.uint8)
            
            # Resize warped shirt to original size
            warped_shirt_full = cv2.resize(warped_shirt_np, (person_img.shape[1], person_img.shape[0]))
            
            # Create 3-channel torso mask and blend
            torso_mask_3ch = np.stack([torso_mask] * 3, axis=-1)
            warped_shirt_masked = warped_shirt_full * torso_mask_3ch
            result = warped_shirt_masked + person_img * (1 - torso_mask_3ch)
            
            # Convert to BGR for saving
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Save result to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=output_dir) as tmp:
                cv2.imwrite(tmp.name, result_bgr)
                return FileResponse(tmp.name, media_type="image/png", filename="try_on_result.png")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Close the uploaded files
        person_image.file.close()
        clothing_image.file.close()

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Virtual Try-On API",
        "version": "1.0.0",
        "description": "API for trying on clothing items on person images",
        "endpoints": {
            "/try-on": "POST - Try on a clothing item",
            "/": "GET - This information"
        }
    }
