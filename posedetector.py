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
        """Draw the detected pose landmarks and connections."""
        if frame is None or points is None:
            logger.warning("Cannot draw landmarks: frame or points is None")
            return frame
        
        # Draw skeleton first (behind points)
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = list(self.BODY_PARTS.keys()).index(partFrom)
            idTo = list(self.BODY_PARTS.keys()).index(partTo)
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
        
        # Define face region points
        face_points = ["Nose", "Neck", "REye", "LEye", "REar", "LEar"]
        face_indices = [list(self.BODY_PARTS.keys()).index(p) for p in face_points]
        
        # Group points by vertical position
        valid_points = [(i, p) for i, p in enumerate(points) if p is not None]
        
        # Separate face points and body points
        face_valid_points = [(i, p) for i, p in valid_points if i in face_indices]
        body_valid_points = [(i, p) for i, p in valid_points if i not in face_indices]
        
        # Process face points with special spacing
        if face_valid_points:
            # Sort face points by y coordinate
            face_valid_points.sort(key=lambda x: x[1][1])
            
            for idx, (i, p) in enumerate(face_valid_points):
                # Draw point
                cv2.circle(frame, p, 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                
                # Get label
                label = list(self.BODY_PARTS.keys())[i]
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                
                # Alternate between left and right sides for face labels
                if idx % 2 == 0:
                    text_x = p[0] + 25  # Place on right
                else:
                    text_x = p[0] - text_size[0] - 25  # Place on left
                
                # Stagger vertical positions more aggressively for face
                text_y = p[1] + (15 * (idx % 3) - 15)
                
                # Add background rectangle
                rect_pad = 2
                rect_x = text_x - rect_pad
                rect_y = text_y - text_size[1] - rect_pad
                rect_w = text_size[0] + 2 * rect_pad
                rect_h = text_size[1] + 2 * rect_pad
                
                cv2.rectangle(frame, 
                             (rect_x, rect_y), 
                             (rect_x + rect_w, rect_y + rect_h),
                             (255, 255, 255), 
                             -1)
                
                cv2.putText(frame, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        
        # Process body points
        vertical_groups = {}
        
        # Group points that are within 40 pixels vertically (increased from 30)
        for i, p in body_valid_points:
            grouped = False
            for y in vertical_groups:
                if abs(p[1] - y) < 40:
                    vertical_groups[y].append((i, p))
                    grouped = True
                    break
            if not grouped:
                vertical_groups[p[1]] = [(i, p)]
        
        # Draw body points and labels
        for y, group in vertical_groups.items():
            # Sort points in group by x-coordinate
            group.sort(key=lambda x: x[1][0])
            
            # Calculate vertical offset for this group
            base_y_offset = 25 if y < frame.shape[0] // 2 else -15
            
            for idx, (i, p) in enumerate(group):
                # Draw point
                cv2.circle(frame, p, 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                
                # Get label
                label = list(self.BODY_PARTS.keys())[i]
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                
                # Calculate horizontal position with increased offset
                if p[0] < frame.shape[1] // 2:
                    text_x = p[0] + 15
                else:
                    text_x = p[0] - text_size[0] - 15
                
                # Calculate vertical position with increased stagger
                if len(group) > 1:
                    y_offset = base_y_offset + (15 * (idx % 2))  # Increased stagger
                else:
                    y_offset = base_y_offset
                
                text_y = p[1] + y_offset
                
                # Add background rectangle
                rect_pad = 2
                rect_x = text_x - rect_pad
                rect_y = text_y - text_size[1] - rect_pad
                rect_w = text_size[0] + 2 * rect_pad
                rect_h = text_size[1] + 2 * rect_pad
                
                cv2.rectangle(frame, 
                             (rect_x, rect_y), 
                             (rect_x + rect_w, rect_y + rect_h),
                             (255, 255, 255), 
                             -1)
                
                cv2.putText(frame, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        
        return frame
