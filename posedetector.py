import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PoseDetector:
    def __init__(self, model_path="graph_opt.pb"):
        logger.info(f"Initializing PoseDetector with model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Model parameters
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.2
        
        # COCO Output Format
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }

        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
        ]

    def load_image(self, path):
        """Load image from file path"""
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path.absolute()}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def detect(self, frame):
        """Detect poses in the input frame"""
        logger.info("Starting pose detection")
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        # Prepare input blob
        net_input = cv2.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False)
        
        # Set the prepared input
        self.net.setInput(net_input)
        
        # Make forward pass
        output = self.net.forward()
        
        H = output.shape[2]
        W = output.shape[3]
        
        # Empty list to store the detected keypoints
        points = []
        
        for i in range(len(self.BODY_PARTS)-1):  # Exclude background
            # Probability map of corresponding body part
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            
            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        return points

    def draw_landmarks(self, frame, points):
        """Draw the detected pose points and connections"""
        if frame is None or points is None:
            logger.warning("Cannot draw landmarks: frame or points is None")
            return frame
            
        # Draw points
        for i, p in enumerate(points):
            if p is not None:
                cv2.circle(frame, p, 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, f"{list(self.BODY_PARTS.keys())[i]}", p,
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        
        # Draw skeleton
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                
        return frame
