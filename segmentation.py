import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import logging
from utils import UNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Segmentor:
    def __init__(self, model_path="models/segmentation_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def segment(self, image):
        """Extract shirt from image using the pre-trained UNet model"""
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Get original size
            orig_size = image.size
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                mask = torch.sigmoid(output)
                mask = (mask > 0.5).float()
            
            # Convert mask back to image size
            mask = mask[0, 0].cpu().numpy()  # Take first image, first channel
            mask = cv2.resize(mask, orig_size)
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Save debug outputs
            debug_dir = Path("output")
            debug_dir.mkdir(exist_ok=True)
            
            # Convert image back to BGR for saving
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(debug_dir / "original.jpg"), image_np)
            cv2.imwrite(str(debug_dir / "segmentation_mask.jpg"), mask)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in segment: {str(e)}")
            return None
