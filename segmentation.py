import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Segmentor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def segment(self, img_rgb, pose_points):
        """Segment the shirt region using color and pose information."""
        h, w = img_rgb.shape[:2]
        
        # Convert to HSV for better color segmentation
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Create torso region from pose points
        if pose_points is not None:
            # Get key points
            neck = pose_points[1]
            shoulders = [pose_points[2], pose_points[5]]  # Left and right shoulder
            hips = [pose_points[8], pose_points[11]]      # Left and right hip
            
            # Calculate dimensions
            shoulder_width = abs(shoulders[1][0] - shoulders[0][0])
            hip_width = abs(hips[1][0] - hips[0][0])
            torso_height = abs(hips[0][1] - neck[1])
            
            # Create base mask from pose points
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Calculate control points for natural curve
            center_x = (shoulders[0][0] + shoulders[1][0]) / 2
            
            # Create natural shoulder line with slight upward curve
            shoulder_curve = int(shoulder_width * 0.1)
            shoulder_points = []
            for t in np.linspace(0, 1, 10):
                x = shoulders[0][0] + t * (shoulders[1][0] - shoulders[0][0])
                y = neck[1] - shoulder_curve * np.sin(np.pi * t)
                shoulder_points.append([int(x), int(y)])
            
            # Create side curves
            left_points = []
            right_points = []
            for t in np.linspace(0, 1, 10):
                # Cubic bezier curve for natural waist
                t2 = t * t
                t3 = t2 * t
                bezier = lambda p0, p1, p2, p3: (
                    p0 * (1-t3) + 3*p1*t*(1-t2) + 3*p2*t2*(1-t) + p3*t3
                )
                
                # Control points for waist curve
                left_ctrl1 = shoulders[0][0] - shoulder_width * 0.1
                right_ctrl1 = shoulders[1][0] + shoulder_width * 0.1
                
                # Left curve
                x_left = bezier(shoulders[0][0], left_ctrl1, 
                              hips[0][0] - hip_width * 0.1, hips[0][0])
                y_left = neck[1] + t * torso_height
                left_points.append([int(x_left), int(y_left)])
                
                # Right curve
                x_right = bezier(shoulders[1][0], right_ctrl1,
                               hips[1][0] + hip_width * 0.1, hips[1][0])
                y_right = neck[1] + t * torso_height
                right_points.append([int(x_right), int(y_right)])
            
            # Combine all points
            points = np.array(shoulder_points + right_points + 
                            [[int(hips[1][0]), int(hips[1][1])]] +
                            [[int(hips[0][0]), int(hips[0][1])]] +
                            left_points[::-1], dtype=np.int32)
            
            # Fill the polygon
            cv2.fillPoly(mask, [points], 255)
            
            # Add neck region
            neck_radius = int(shoulder_width * 0.15)
            cv2.circle(mask, (int(neck[0]), int(neck[1])), neck_radius, 255, -1)
            
            # Exclude face
            if pose_points[0][0] > 0 and pose_points[0][1] > 0:  # If nose is detected
                face_radius = int(shoulder_width * 0.4)
                cv2.circle(mask, (int(pose_points[0][0]), int(pose_points[0][1])),
                          face_radius, 0, -1)
            
            # Smooth edges
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Create color mask using the pose mask as a guide
            color_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Sample colors from the center of the torso
            center_y = int((neck[1] + hips[0][1]) / 2)
            center_x = int((shoulders[0][0] + shoulders[1][0]) / 2)
            sample_region = img_hsv[center_y-20:center_y+20, center_x-20:center_x+20]
            
            if sample_region.size > 0:
                # Calculate average color in sample region
                avg_hue = np.median(sample_region[:,:,0])
                avg_sat = np.median(sample_region[:,:,1])
                avg_val = np.median(sample_region[:,:,2])
                
                # Create color thresholds
                hue_range = 20
                sat_range = 50
                val_range = 50
                
                # Create color mask
                color_mask = cv2.inRange(img_hsv,
                    (max(0, avg_hue - hue_range), 
                     max(0, avg_sat - sat_range),
                     max(0, avg_val - val_range)),
                    (min(180, avg_hue + hue_range),
                     min(255, avg_sat + sat_range),
                     min(255, avg_val + val_range)))
                
                # Combine pose and color masks
                combined_mask = cv2.bitwise_and(mask, color_mask)
                
                # Clean up the mask
                kernel = np.ones((5,5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                
                # Final smoothing
                combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
                
                return combined_mask
            
        return mask

    def segment_old(self, image, pose_points=None):
        """Segment clothing using color and pose information"""
        try:
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = image
            else:
                img_rgb = np.array(image)
            
            # Convert to HSV for better color segmentation
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # Create initial mask using color thresholding
            # Exclude very dark (black) and very light (white) colors
            # Use multiple HSV ranges to better capture different clothing colors
            mask1 = cv2.inRange(img_hsv, np.array([0, 15, 40]), np.array([180, 255, 240]))  # Most colors
            mask2 = cv2.inRange(img_hsv, np.array([0, 0, 50]), np.array([180, 30, 240]))    # Grays
            color_mask = cv2.bitwise_or(mask1, mask2)
            
            # Create torso region from pose points
            if pose_points is not None:
                h, w = img_rgb.shape[:2]
                torso_region = np.zeros((h, w), dtype=np.uint8)
                face_region = np.zeros((h, w), dtype=np.uint8)
                
                # Get relevant pose points
                shoulders = [pose_points[2], pose_points[5]]  # Left and right shoulder
                hips = [pose_points[8], pose_points[11]]      # Left and right hip
                neck = pose_points[1]                         # Neck
                nose = pose_points[0]                         # Nose
                
                # Calculate dimensions
                shoulder_width = abs(shoulders[1][0] - shoulders[0][0])
                padding = int(shoulder_width * 0.2)  # 20% of shoulder width
                
                # Create smooth curve for torso
                curve_points = []
                num_points = 20  # Number of points for smooth curve
                
                # Generate points for curved top (collar)
                center_x = (shoulders[0][0] + shoulders[1][0]) / 2
                collar_height = int(shoulder_width * 0.1)  # Slight curve up at collar
                
                for i in range(num_points):
                    t = i / (num_points - 1)
                    x = shoulders[0][0] + t * (shoulders[1][0] - shoulders[0][0])
                    # Quadratic curve for collar
                    y = neck[1] - collar_height * 4 * (t - 0.5) ** 2
                    curve_points.append([x, y])
                
                # Generate points for sides with natural curve
                left_points = []
                right_points = []
                torso_height = hips[0][1] - neck[1]
                
                for i in range(num_points):
                    t = i / (num_points - 1)
                    # Cubic bezier for natural waist curve
                    ctrl1 = 0.3  # Control point 1 at 30% down
                    ctrl2 = 0.7  # Control point 2 at 70% down
                    
                    # Left side curve
                    x_left = shoulders[0][0] - padding + (hips[0][0] - shoulders[0][0]) * (
                        (1-t)**3 + 3*ctrl1*t*(1-t)**2 + 3*ctrl2*t**2*(1-t) + t**3
                    )
                    y = neck[1] + t * torso_height
                    left_points.append([x_left, y])
                    
                    # Right side curve
                    x_right = shoulders[1][0] + padding + (hips[1][0] - shoulders[1][0]) * (
                        (1-t)**3 + 3*ctrl1*t*(1-t)**2 + 3*ctrl2*t**2*(1-t) + t**3
                    )
                    right_points.append([x_right, y])
                
                # Generate points for curved bottom
                bottom_points = []
                bottom_curve = int(shoulder_width * 0.05)  # Slight curve at bottom
                
                for i in range(num_points):
                    t = i / (num_points - 1)
                    x = hips[0][0] + t * (hips[1][0] - hips[0][0])
                    # Quadratic curve for bottom
                    y = hips[0][1] + bottom_curve * 4 * (t - 0.5) ** 2
                    bottom_points.append([x, y])
                
                # Combine all points into smooth contour
                all_points = (curve_points + right_points + bottom_points[::-1] + left_points[::-1])
                points_array = np.array(all_points, dtype=np.int32)
                
                # Fill torso region with smooth contour
                cv2.fillPoly(torso_region, [points_array], 255)
                
                # Add neck region with smooth transition
                neck_radius = int(shoulder_width * 0.15)  # 15% of shoulder width
                neck_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(neck_mask, (int(neck[0]), int(neck[1])), neck_radius, 255, -1)
                
                # Smooth the neck mask
                neck_mask = cv2.GaussianBlur(neck_mask, (15, 15), 0)
                
                # Combine torso and neck masks
                torso_region = cv2.bitwise_or(torso_region, neck_mask)
                
                # Create and exclude face region
                if nose[0] > 0 and nose[1] > 0:
                    face_radius = int(shoulder_width * 0.4)
                    cv2.circle(face_region, (int(nose[0]), int(nose[1])), face_radius, 255, -1)
                    face_region = cv2.GaussianBlur(face_region, (31, 31), 0)
                    torso_region = cv2.bitwise_and(torso_region, cv2.bitwise_not(face_region))
                
                # Smooth the entire mask
                torso_region = cv2.GaussianBlur(torso_region, (15, 15), 0)
                
                # Combine color mask with torso region using weighted blend
                color_mask_float = color_mask.astype(float) / 255
                torso_region_float = torso_region.astype(float) / 255
                
                # Weight the masks (give more weight to color mask in the center)
                center_weight = np.zeros((h, w), dtype=float)
                center_y = (neck[1] + hips[0][1]) // 2
                center_x = (shoulders[0][0] + shoulders[1][0]) // 2
                
                for y in range(h):
                    for x in range(w):
                        dist = np.sqrt(((x - center_x) / shoulder_width) ** 2 + 
                                     ((y - center_y) / torso_height) ** 2)
                        center_weight[y, x] = np.clip(1 - dist, 0, 1)
                
                # Combine masks with center-weighted blend
                combined_mask = (color_mask_float * center_weight + 
                               torso_region_float * (1 - center_weight))
                
                # Threshold and clean up
                mask = (combined_mask > 0.3).astype(np.uint8) * 255
                
                # Final smoothing
                kernel_size = max(3, int(shoulder_width * 0.02))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
                
            else:
                mask = color_mask
            
            # Save debug outputs
            debug_dir = Path("output")
            debug_dir.mkdir(exist_ok=True)
            
            # Save original image
            cv2.imwrite(str(debug_dir / "original.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            # Save intermediate masks
            cv2.imwrite(str(debug_dir / "color_mask.jpg"), color_mask)
            if pose_points is not None:
                cv2.imwrite(str(debug_dir / "torso_region.jpg"), torso_region)
            cv2.imwrite(str(debug_dir / "final_mask.jpg"), mask)
            
            # Create visualization
            overlay = img_rgb.copy()
            overlay[mask > 0] = [0, 255, 0]  # Highlight segmented region in green
            alpha = 0.5
            segmentation_viz = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
            cv2.imwrite(str(debug_dir / "segmentation_result.png"), cv2.cvtColor(segmentation_viz, cv2.COLOR_RGB2BGR))
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in segment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
