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
        if pose_points is None:
            return None
            
        h, w = img_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get key pose points
        nose = pose_points[0]
        neck = pose_points[1]
        l_shoulder = pose_points[2]
        r_shoulder = pose_points[5]
        l_hip = pose_points[8]
        r_hip = pose_points[11]
        
        if neck is None or l_shoulder is None or r_shoulder is None or l_hip is None or r_hip is None:
            return None
            
        # Calculate key measurements
        shoulder_width = np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder))
        hip_width = np.linalg.norm(np.array(r_hip) - np.array(l_hip))
        torso_height = np.linalg.norm(np.array(l_hip) - np.array(l_shoulder))
        
        # Add padding
        padding_x = int(shoulder_width * 0.2)
        padding_y = int(torso_height * 0.1)
        
        def create_bezier_curve(p0, p1, p2, p3, num_points=20):
            points = []
            for t in np.linspace(0, 1, num_points):
                t2 = t * t
                t3 = t2 * t
                x = p0[0] * (1-t3) + 3*p1[0]*t*(1-t2) + 3*p2[0]*t2*(1-t) + p3[0]*t3
                y = p0[1] * (1-t3) + 3*p1[1]*t*(1-t2) + 3*p2[1]*t2*(1-t) + p3[1]*t3
                points.append([int(x), int(y)])
            return points
        
        # Create natural curves for the sides of the torso
        # Left side control points
        l_ctrl1 = [l_shoulder[0] - padding_x, l_shoulder[1] + torso_height * 0.2]
        l_ctrl2 = [l_hip[0] - padding_x, l_hip[1] - torso_height * 0.2]
        
        # Right side control points
        r_ctrl1 = [r_shoulder[0] + padding_x, r_shoulder[1] + torso_height * 0.2]
        r_ctrl2 = [r_hip[0] + padding_x, r_hip[1] - torso_height * 0.2]
        
        # Generate curve points
        left_curve = create_bezier_curve(
            [l_shoulder[0] - padding_x, l_shoulder[1]],
            l_ctrl1,
            l_ctrl2,
            [l_hip[0] - padding_x, l_hip[1] + padding_y]
        )
        
        right_curve = create_bezier_curve(
            [r_shoulder[0] + padding_x, r_shoulder[1]],
            r_ctrl1,
            r_ctrl2,
            [r_hip[0] + padding_x, r_hip[1] + padding_y]
        )
        
        # Create shoulder line with slight curve
        shoulder_points = []
        shoulder_height = min(l_shoulder[1], r_shoulder[1]) - padding_y
        for t in np.linspace(0, 1, 20):
            x = l_shoulder[0] - padding_x + t * (r_shoulder[0] - l_shoulder[0] + 2*padding_x)
            y = shoulder_height - np.sin(np.pi * t) * padding_y
            shoulder_points.append([int(x), int(y)])
        
        # Create bottom curve
        bottom_points = []
        for t in np.linspace(0, 1, 20):
            x = l_hip[0] - padding_x + t * (r_hip[0] - l_hip[0] + 2*padding_x)
            y = max(l_hip[1], r_hip[1]) + padding_y + np.sin(np.pi * t) * padding_y
            bottom_points.append([int(x), int(y)])
        
        # Combine all points
        all_points = np.array(
            shoulder_points + 
            right_curve + 
            bottom_points[::-1] + 
            left_curve[::-1], 
            dtype=np.int32
        )
        
        # Fill the main torso
        cv2.fillPoly(mask, [all_points], 255)
        
        # Add neck region
        neck_top = [neck[0], neck[1] - int(shoulder_width * 0.3)]
        neck_width = int(shoulder_width * 0.3)
        neck_points = np.array([
            [neck_top[0] - neck_width, neck_top[1]],
            [neck_top[0] + neck_width, neck_top[1]],
            [r_shoulder[0], r_shoulder[1]],
            [l_shoulder[0], l_shoulder[1]]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [neck_points], 255)
        
        # Smooth the edges
        kernel_size = max(5, int(shoulder_width * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Exclude face region if nose is detected
        if nose[0] > 0 and nose[1] > 0:
            face_radius = int(shoulder_width * 0.4)
            cv2.circle(mask, (int(nose[0]), int(nose[1])), 
                      face_radius, 0, -1)
        
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
