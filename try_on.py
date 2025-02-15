import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
from posedetector import PoseDetector
from gmm import load_gmm, prepare_person_representation, tps_transform
from utils import UNet
from segmentation import Segmentor
import traceback

def load_segmentation_model(model_path='models/segmentation_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def segment_shirt(model, image_path, threshold=0.5):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Remove white background by creating an alpha mask
    is_white = np.all(img_np > 240, axis=2)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output[0, 0]) > threshold
        mask = mask.cpu().numpy().astype(np.uint8)
        
        # Remove any white background from the mask
        mask_full = cv2.resize(mask, (image.size[0], image.size[1]))
        mask_full[is_white] = 0
        mask = cv2.resize(mask_full, (128, 128))
    
    return mask

def create_body_mask(pose_points, height, width):
    """Create a full body mask using pose points."""
    # Get all relevant points
    nose = pose_points[0]
    neck = pose_points[1]
    r_shoulder = pose_points[2]
    r_elbow = pose_points[3]
    r_wrist = pose_points[4]
    l_shoulder = pose_points[5]
    l_elbow = pose_points[6]
    l_wrist = pose_points[7]
    r_hip = pose_points[8]
    r_knee = pose_points[9]
    r_ankle = pose_points[10]
    l_hip = pose_points[11]
    l_knee = pose_points[12]
    l_ankle = pose_points[13]
    
    # Convert valid points to numpy arrays
    points = []
    for p in [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
              r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]:
        if p is not None:
            points.append(np.array(p))
        else:
            return np.zeros((height, width), dtype=np.float32)
    
    [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
     r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle] = points
    
    # Calculate key measurements
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
    hip_width = np.linalg.norm(r_hip - l_hip)
    
    # Create body mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Helper function to draw a filled polygon
    def draw_polygon(points):
        if len(points) > 0:
            pts = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(mask, pts, 1.0)
    
    # Draw head and neck
    head_top = nose + (nose - neck) * 1.2
    head_width = shoulder_width * 0.4
    head_points = [
        tuple(map(int, head_top + np.array([-head_width/2, 0]))),
        tuple(map(int, head_top + np.array([head_width/2, 0]))),
        tuple(map(int, r_shoulder)),
        tuple(map(int, l_shoulder))
    ]
    draw_polygon(head_points)
    
    # Draw torso with curved sides
    num_points = 8
    r_side_points = []
    l_side_points = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        # Interpolate between shoulder and hip
        r_point = r_shoulder * (1-t) + r_hip * t
        l_point = l_shoulder * (1-t) + l_hip * t
        # Add curve
        curve = np.sin(t * np.pi) * shoulder_width * 0.15
        r_point = r_point + np.array([curve, 0])
        l_point = l_point + np.array([-curve, 0])
        r_side_points.append(tuple(map(int, r_point)))
        l_side_points.append(tuple(map(int, l_point)))
    
    # Draw torso
    torso_points = r_side_points + l_side_points[::-1]
    draw_polygon(torso_points)
    
    # Draw arms
    arm_width = shoulder_width * 0.15
    
    def draw_limb(p1, p2, width):
        if p1 is not None and p2 is not None:
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length > 0:
                norm = vec / length
                perp = np.array([-norm[1], norm[0]]) * width
                points = [
                    tuple(map(int, p1 + perp)),
                    tuple(map(int, p2 + perp)),
                    tuple(map(int, p2 - perp)),
                    tuple(map(int, p1 - perp))
                ]
                draw_polygon(points)
    
    # Draw arms
    draw_limb(r_shoulder, r_elbow, arm_width)
    draw_limb(r_elbow, r_wrist, arm_width * 0.8)
    draw_limb(l_shoulder, l_elbow, arm_width)
    draw_limb(l_elbow, l_wrist, arm_width * 0.8)
    
    # Draw legs
    leg_width = hip_width * 0.25
    draw_limb(r_hip, r_knee, leg_width)
    draw_limb(r_knee, r_ankle, leg_width * 0.8)
    draw_limb(l_hip, l_knee, leg_width)
    draw_limb(l_knee, l_ankle, leg_width * 0.8)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    mask = np.clip(mask, 0, 1)
    
    return mask

def create_face_hair_mask(pose_points, img):
    """Create a mask for face and hair regions"""
    mask = np.zeros_like(img)
    
    # Use nose and eyes to estimate face region
    face_points = [0, 14, 15, 16, 17]  # nose, eyes, ears
    face_coords = []
    
    for i in face_points:
        if pose_points[i] is not None:
            face_coords.append([int(pose_points[i][0]), int(pose_points[i][1])])
    
    if len(face_coords) >= 3:
        face_coords = np.array(face_coords, dtype=np.int32)
        # Create an enlarged face region
        rect = cv2.boundingRect(face_coords)
        x, y, w, h = rect
        # Enlarge the rectangle
        x = max(0, x - w//2)
        y = max(0, y - h)
        w = min(img.shape[1] - x, w * 2)
        h = min(img.shape[0] - y, h * 2)
        
        # Copy the face region
        mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
    
    return mask

def create_torso_mask(pose_points, height, width):
    """Create a binary mask for the torso region."""
    # Get key points
    neck = pose_points[1]
    r_shoulder = pose_points[2]
    l_shoulder = pose_points[5]
    r_hip = pose_points[8]
    l_hip = pose_points[11]
    r_elbow = pose_points[3]
    l_elbow = pose_points[6]
    
    if any(p is None for p in [neck, r_shoulder, l_shoulder, r_hip, l_hip]):
        return np.zeros((height, width), dtype=np.float32)
    
    # Convert points to numpy arrays
    neck = np.array(neck)
    r_shoulder = np.array(r_shoulder)
    l_shoulder = np.array(l_shoulder)
    r_hip = np.array(r_hip)
    l_hip = np.array(l_hip)
    r_elbow = np.array(r_elbow) if r_elbow is not None else None
    l_elbow = np.array(l_elbow) if l_elbow is not None else None
    
    # Calculate key measurements
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
    hip_width = np.linalg.norm(r_hip - l_hip)
    torso_height = np.linalg.norm((r_hip + l_hip)/2 - neck)
    
    # Create collar points
    collar_width = shoulder_width * 0.3
    collar_height = collar_width * 0.2
    collar_center = neck + np.array([0, -collar_height])
    collar_right = collar_center + np.array([collar_width/2, 0])
    collar_left = collar_center + np.array([-collar_width/2, 0])
    
    # Create shoulder points with natural padding
    shoulder_pad = shoulder_width * 0.2  # Slightly reduced
    r_shoulder_out = r_shoulder + np.array([shoulder_pad, 0])
    l_shoulder_out = l_shoulder + np.array([-shoulder_pad, 0])
    
    # Add hip padding - make it proportional to shoulder padding
    hip_pad = shoulder_pad * 0.9  # Slightly less than shoulder pad
    r_hip_out = r_hip + np.array([hip_pad, 0])
    l_hip_out = l_hip + np.array([-hip_pad, 0])
    
    # Create curved sides using multiple control points
    num_points = 16  # Increased for even smoother curves
    r_side_points = []
    l_side_points = []
    
    # Helper function to get sleeve direction
    def get_sleeve_direction(shoulder, elbow):
        if elbow is not None:
            return (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        return np.array([1, 0])  # Default horizontal direction
    
    r_sleeve_dir = get_sleeve_direction(r_shoulder, r_elbow)
    l_sleeve_dir = get_sleeve_direction(l_shoulder, l_elbow)
    
    # Calculate waist position (approximately)
    waist_height = (r_hip[1] + l_hip[1]) / 2 - (r_shoulder[1] + l_shoulder[1]) / 2
    waist_y = r_shoulder[1] + waist_height * 0.6  # Waist at 60% down from shoulder
    
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # Base points - interpolate between padded points
        r_base = r_shoulder_out * (1-t) + r_hip_out * t
        l_base = l_shoulder_out * (1-t) + l_hip_out * t
        
        # Calculate distance from waist
        dist_from_waist = abs(r_base[1] - waist_y) / waist_height
        
        # Curve inward more at waist, less at shoulders and hips
        curve_factor = 0.3 * (1 - np.exp(-dist_from_waist * 2))  # Exponential falloff
        
        # Add natural curves that taper in at waist
        r_curve = shoulder_width * curve_factor
        l_curve = -shoulder_width * curve_factor
        
        # Add sleeve influence only in upper body
        if t < 0.3:
            sleeve_influence = (0.3 - t) / 0.3
            r_base = r_base + r_sleeve_dir * shoulder_width * 0.15 * sleeve_influence
            l_base = l_base + l_sleeve_dir * shoulder_width * 0.15 * sleeve_influence
        
        # Calculate final points
        r_point = r_base + np.array([r_curve, 0])
        l_point = l_base + np.array([l_curve, 0])
        
        # Add slight forward curve throughout torso
        forward_curve = np.sin(t * np.pi) * shoulder_width * 0.05
        r_point = r_point + np.array([0, forward_curve])
        l_point = l_point + np.array([0, forward_curve])
        
        r_side_points.append(tuple(map(int, r_point)))
        l_side_points.append(tuple(map(int, l_point)))
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Draw the torso shape
    points = ([tuple(map(int, p)) for p in [collar_center, collar_right, r_shoulder_out]] + 
             r_side_points + 
             [tuple(map(int, r_hip_out)), tuple(map(int, l_hip_out))] + 
             l_side_points[::-1] + 
             [tuple(map(int, l_shoulder_out)), tuple(map(int, collar_left))])
    
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1.0)
    
    # Apply minimal smoothing
    mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
    
    # Ensure binary mask
    mask = np.clip(mask, 0, 1)
    
    return mask

def try_on_shirt(person_path, shirt_path, pose_model_path='models/graph_opt.pb', gmm_model_path='models/gmm_final.pth'):
    try:
        # Initialize models
        pose_detector = PoseDetector(model_path=pose_model_path)
        segmentor = Segmentor()  # Initialize segmentation model
        gmm_model = load_gmm(gmm_model_path)
        
        # Load person image and detect pose
        person_img = cv2.imread(person_path)
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        pose_points = pose_detector.detect(person_img)
        
        # Create person representation for GMM
        body_mask = create_body_mask(pose_points, person_img.shape[0], person_img.shape[1])
        
        # Get torso segmentation once and reuse it
        segmentor = Segmentor()
        torso_mask = segmentor.segment(person_img, pose_points)
        if torso_mask is None:
            # Fallback to pose-based mask if segmentation fails
            torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
        else:
            # Normalize mask to 0-1 range
            torso_mask = torso_mask.astype(np.float32) / 255.0
            
        face_hair_mask = create_face_hair_mask(pose_points, person_img)
        person_repr = prepare_person_representation(pose_points, torso_mask, face_hair_mask)
        
        # Segment and prepare shirt
        shirt_img = cv2.imread(shirt_path)
        shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Remove white background from shirt with smoother transition
        is_white = np.all(shirt_img > 240, axis=2)
        alpha = np.ones((shirt_img.shape[0], shirt_img.shape[1]), dtype=np.float32)
        alpha[is_white] = 0.0
        
        # Apply Gaussian blur to alpha for smoother edges
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Resize shirt while maintaining aspect ratio
        target_height = person_img.shape[0]
        aspect_ratio = shirt_img.shape[1] / shirt_img.shape[0]
        target_width = int(target_height * aspect_ratio)
        shirt_img = cv2.resize(shirt_img, (target_width, target_height))
        alpha = cv2.resize(alpha, (target_width, target_height))
        
        # Center the shirt horizontally
        if target_width < person_img.shape[1]:
            pad_left = (person_img.shape[1] - target_width) // 2
            pad_right = person_img.shape[1] - target_width - pad_left
            shirt_img = cv2.copyMakeBorder(shirt_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            alpha = cv2.copyMakeBorder(alpha, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0])
        else:
            # Crop from center if shirt is too wide
            start_x = (target_width - person_img.shape[1]) // 2
            shirt_img = shirt_img[:, start_x:start_x + person_img.shape[1]]
            alpha = alpha[:, start_x:start_x + person_img.shape[1]]
        
        # Warp shirt using pose points and TPS
        if pose_points is not None and torso_mask is not None:
            # Use the already computed torso mask
            final_mask = (torso_mask * 255).astype(np.uint8)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return person_img
                
            # Get the largest contour
            target_contour = max(contours, key=cv2.contourArea)
            
            # Create shirt mask from alpha channel
            shirt_mask = (alpha * 255).astype(np.uint8)
            shirt_contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
            if not shirt_contours:
                return person_img
                
            shirt_contour = max(shirt_contours, key=cv2.contourArea)
            
            # Sample points along contours with more points for better detail
            num_points = 200  # Increased from 100
            
            # Get evenly spaced points along contours
            def get_evenly_spaced_points(contour, num_points):
                total_length = cv2.arcLength(contour, True)
                points = []
                for i in range(num_points):
                    index = int((i * len(contour)) / num_points)
                    points.append(contour[index][0])
                return np.array(points, dtype=np.float32)
            
            target_points = get_evenly_spaced_points(target_contour, num_points)
            shirt_points = get_evenly_spaced_points(shirt_contour, num_points)
            
            # Add additional control points for better shape preservation
            if pose_points[1] is not None:  # neck
                neck = pose_points[1]
                shoulders = [pose_points[2], pose_points[5]]  # left and right shoulder
                chest = [pose_points[8], pose_points[11]]     # left and right hip
                
                # Calculate key measurements
                shoulder_width = np.linalg.norm(np.array(shoulders[1]) - np.array(shoulders[0]))
                chest_width = np.linalg.norm(np.array(chest[1]) - np.array(chest[0]))
                torso_height = np.linalg.norm(np.array(chest[0]) - np.array(shoulders[0]))
                
                # Add shoulder points
                target_points = np.vstack([target_points, shoulders[0], shoulders[1]])
                shirt_shoulder_left = [shirt_img.shape[1]//4, shirt_img.shape[0]//4]
                shirt_shoulder_right = [3*shirt_img.shape[1]//4, shirt_img.shape[0]//4]
                shirt_points = np.vstack([shirt_points, shirt_shoulder_left, shirt_shoulder_right])
                
                # Add chest points
                target_points = np.vstack([target_points, chest[0], chest[1]])
                shirt_chest_left = [shirt_img.shape[1]//4, 3*shirt_img.shape[0]//4]
                shirt_chest_right = [3*shirt_img.shape[1]//4, 3*shirt_img.shape[0]//4]
                shirt_points = np.vstack([shirt_points, shirt_chest_left, shirt_chest_right])
                
                # Add center line points
                center_line_target = np.array([
                    [neck[0], neck[1]],
                    [(shoulders[0][0] + shoulders[1][0])/2, (shoulders[0][1] + shoulders[1][1])/2],
                    [(chest[0][0] + chest[1][0])/2, (chest[0][1] + chest[1][1])/2]
                ])
                center_line_shirt = np.array([
                    [shirt_img.shape[1]//2, 0],
                    [shirt_img.shape[1]//2, shirt_img.shape[0]//2],
                    [shirt_img.shape[1]//2, shirt_img.shape[0]]
                ])
                target_points = np.vstack([target_points, center_line_target])
                shirt_points = np.vstack([shirt_points, center_line_shirt])
            
            # Create triangulation for piece-wise affine warping
            rect = (0, 0, shirt_img.shape[1], shirt_img.shape[0])
            subdiv = cv2.Subdiv2D(rect)
            
            # Add points to subdivision
            for point in shirt_points:
                if 0 <= point[0] < shirt_img.shape[1] and 0 <= point[1] < shirt_img.shape[0]:
                    subdiv.insert(tuple(map(float, point)))
            
            # Get triangles
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)
            
            # Convert triangles to point indices with shape preservation
            src_triangles = []
            dst_triangles = []
            
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                
                # Find indices in shirt_points
                idx1 = np.argmin(np.sum((shirt_points - pt1) ** 2, axis=1))
                idx2 = np.argmin(np.sum((shirt_points - pt2) ** 2, axis=1))
                idx3 = np.argmin(np.sum((shirt_points - pt3) ** 2, axis=1))
                
                if idx1 != idx2 and idx2 != idx3 and idx3 != idx1:
                    # Calculate triangle properties for shape preservation
                    src_tri = np.float32([shirt_points[idx1], shirt_points[idx2], shirt_points[idx3]])
                    dst_tri = np.float32([target_points[idx1], target_points[idx2], target_points[idx3]])
                    
                    # Calculate and preserve aspect ratio
                    src_width = np.linalg.norm(src_tri[1] - src_tri[0])
                    src_height = np.linalg.norm(src_tri[2] - src_tri[0])
                    dst_width = np.linalg.norm(dst_tri[1] - dst_tri[0])
                    dst_height = np.linalg.norm(dst_tri[2] - dst_tri[0])
                    
                    if src_width > 0 and src_height > 0 and dst_width > 0 and dst_height > 0:
                        # Adjust destination points to preserve aspect ratio
                        scale = min(dst_width/src_width, dst_height/src_height)
                        center = np.mean(dst_tri, axis=0)
                        dst_tri = center + (dst_tri - center) * scale
                        
                        src_triangles.append(src_tri)
                        dst_triangles.append(dst_tri)
            
            # Create output image
            warped_shirt = np.zeros_like(shirt_img, dtype=np.float32)
            warped_alpha = np.zeros((person_img.shape[0], person_img.shape[1]), dtype=np.float32)
            
            print("Warping triangles...")
            # Create initial triangulation mask
            tri_mask = np.zeros((person_img.shape[0], person_img.shape[1]), dtype=np.float32)
            
            # Warp each triangle
            for src_tri, dst_tri in zip(src_triangles, dst_triangles):
                # Get bounding rectangle for dest triangle
                rect = cv2.boundingRect(np.int32(dst_tri))
                x, y, w, h = rect
                
                # Create mask for current triangle
                mask = np.zeros((h, w), dtype=np.float32)
                dst_tri_shifted = dst_tri - np.float32([[x, y]])
                cv2.fillConvexPoly(mask, np.int32(dst_tri_shifted), 1)
                
                # Add to triangulation mask
                tri_roi = tri_mask[y:y+h, x:x+w]
                if tri_roi.shape == mask.shape:
                    tri_mask[y:y+h, x:x+w] = np.maximum(tri_roi, mask)
                
                # Get affine transform
                M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri_shifted))
                
                # Warp shirt piece
                warped_piece = cv2.warpAffine(shirt_img, M, (w, h), flags=cv2.INTER_LINEAR)
                
                # Apply mask
                mask_3ch = np.stack([mask] * 3, axis=2)
                warped_piece = warped_piece * mask_3ch
                
                # Update output image in valid region
                roi = warped_shirt[y:y+h, x:x+w]
                if roi.shape == warped_piece.shape:
                    warped_shirt[y:y+h, x:x+w] = roi * (1 - mask_3ch) + warped_piece
                
                # Warp alpha channel
                alpha_piece = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR)
                alpha_piece = alpha_piece * mask
                
                # Update alpha in valid region
                alpha_roi = warped_alpha[y:y+h, x:x+w]
                if alpha_roi.shape == alpha_piece.shape:
                    warped_alpha[y:y+h, x:x+w] = alpha_roi * (1 - mask) + alpha_piece
            
            print("Cleaning up alpha...")
            # Clean up warped alpha
            warped_alpha = np.clip(warped_alpha, 0, 1)
            warped_alpha = cv2.GaussianBlur(warped_alpha, (5, 5), 0)
            
            # Save triangulation mask for debugging
            cv2.imwrite('output/triangulation_mask.png', (tri_mask * 255).astype(np.uint8))
            
            print("Creating combined mask...")
            # Convert torso mask to float32 and normalize
            torso_mask = torso_mask.astype(np.float32) / 255.0
            
            # Debug: Save intermediate masks
            cv2.imwrite('output/warped_alpha.png', (warped_alpha * 255).astype(np.uint8))
            cv2.imwrite('output/torso_mask.png', (torso_mask * 255).astype(np.uint8))
            
            # Create initial combined mask
            combined_mask = warped_alpha * torso_mask
            
            # Get pose points for torso mask
            if pose_points[1] is not None:  # If we have pose points
                neck = pose_points[1]
                shoulders = [pose_points[2], pose_points[5]]  # Left and right shoulder
                hips = [pose_points[8], pose_points[11]]     # Left and right hip
                
                # Calculate dimensions
                shoulder_width = np.linalg.norm(np.array(shoulders[1]) - np.array(shoulders[0]))
                hip_width = np.linalg.norm(np.array(hips[1]) - np.array(hips[0]))
                torso_height = np.linalg.norm(np.array(hips[0]) - np.array(shoulders[0]))
                
                # Create a base torso mask from pose points with more padding
                base_torso = np.zeros_like(torso_mask)
                points = np.array([
                    [shoulders[0][0] - int(shoulder_width * 0.3), shoulders[0][1] - int(shoulder_width * 0.1)],  # Left shoulder
                    [shoulders[1][0] + int(shoulder_width * 0.3), shoulders[1][1] - int(shoulder_width * 0.1)],  # Right shoulder
                    [hips[1][0] + int(hip_width * 0.3), hips[1][1] + int(torso_height * 0.2)],                 # Right hip
                    [hips[0][0] - int(hip_width * 0.3), hips[0][1] + int(torso_height * 0.2)]                  # Left hip
                ], dtype=np.int32)
                cv2.fillConvexPoly(base_torso, points, 1.0)
                
                # Add neck region
                neck_width = int(shoulder_width * 0.2)
                neck_height = int(shoulder_width * 0.3)
                neck_points = np.array([
                    [neck[0] - neck_width, neck[1] - neck_height],
                    [neck[0] + neck_width, neck[1] - neck_height],
                    [shoulders[1][0] + int(shoulder_width * 0.2), shoulders[1][1]],
                    [shoulders[0][0] - int(shoulder_width * 0.2), shoulders[0][1]]
                ], dtype=np.int32)
                cv2.fillConvexPoly(base_torso, neck_points, 1.0)
                
                # Save base torso mask for debugging
                cv2.imwrite('output/base_torso_mask.png', (base_torso * 255).astype(np.uint8))
                
                # Combine with base torso mask
                combined_mask = cv2.bitwise_or(
                    combined_mask.astype(np.uint8),
                    (base_torso * warped_alpha).astype(np.uint8)
                ).astype(np.float32)
            
            # Ensure full coverage
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel).astype(np.float32)
            
            # Smooth the mask
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            
            # Save combined mask for debugging
            cv2.imwrite('output/combined_mask.png', (combined_mask * 255).astype(np.uint8))
            
            print("Blending images...")
            # Expand to 3 channels for blending
            combined_mask_3ch = np.stack([combined_mask] * 3, axis=2)
            
            # Save warped shirt for debugging
            cv2.imwrite('output/warped_shirt.png', cv2.convertScaleAbs(warped_shirt))
            
            # Perform blending with explicit type conversion and range checking
            result = person_img.astype(np.float32) * (1 - combined_mask_3ch) + warped_shirt * combined_mask_3ch
            
            # Print debug info
            print(f"Warped shirt shape: {warped_shirt.shape}")
            print(f"Combined mask shape: {combined_mask_3ch.shape}")
            print(f"Person image shape: {person_img.shape}")
            print(f"Combined mask range: {combined_mask.min():.2f} to {combined_mask.max():.2f}")
            
            # Clean up result
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for saving
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('output/try_on_result.png', result_bgr)
            print("Result saved to output/try_on_result.png")
            print("Pose detection saved to output/pose_result.png")
            print("Segmentation visualization saved to output/segmentation_result.png")
            
            # Visualize keypoints on result
            result_with_points = result.copy()
            pose_detector.draw_landmarks(result_with_points, pose_points)
            cv2.imwrite('output/try_on_result_with_points.png', cv2.cvtColor(result_with_points, cv2.COLOR_RGB2BGR))
            print("Result with keypoints saved to output/try_on_result_with_points.png")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

def main(person_path, shirt_path):
    """Main function to process virtual try-on."""
    # Load person image
    person_img = cv2.imread(person_path)
    person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    
    # Initialize models
    pose_detector = PoseDetector(model_path='models/graph_opt.pb')
    segmentor = Segmentor()
    
    # Get pose points
    pose_points = pose_detector.detect(person_img)
    
    if not pose_points or None in [pose_points[i] for i in [2, 5, 8, 11]]:
        print("Could not detect pose in the person image")
        return
    
    # Create pose visualization
    pose_vis = person_img.copy()
    pose_detector.draw_landmarks(pose_vis, pose_points)
    pose_vis_bgr = cv2.cvtColor(pose_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/pose_result.png', pose_vis_bgr)
    
    # Create masks using segmentation model
    torso_mask = segmentor.segment(person_img, pose_points)
    if torso_mask is None:
        # Fallback to pose-based mask if segmentation fails
        print("Segmentation failed, falling back to pose-based mask")
        torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
        torso_mask = (torso_mask * 255).astype(np.uint8)
    
    # Create body mask and face mask
    body_mask = create_body_mask(pose_points, person_img.shape[0], person_img.shape[1])
    face_hair_mask = create_face_hair_mask(pose_points, person_img)
    
    # Create segmentation visualization
    seg_vis = person_img.copy().astype(np.float32)
    
    # Create colored overlays
    body_mask_3ch = np.stack([body_mask] * 3, axis=-1)
    torso_mask_3ch = np.stack([torso_mask.astype(np.float32) / 255.0] * 3, axis=-1)
    face_mask_3ch = np.stack([face_hair_mask[:,:,0]] * 3, axis=-1)
    
    # Add colored overlays
    seg_vis = np.where(body_mask_3ch > 0, 
                      seg_vis * 0.7 + np.array([0, 0, 255]) * 0.3,  # Blue tint for body
                      seg_vis)
    seg_vis = np.where(torso_mask_3ch > 0,
                      seg_vis * 0.7 + np.array([255, 0, 0]) * 0.3,  # Red tint for torso
                      seg_vis)
    seg_vis = np.where(face_mask_3ch > 0,
                      seg_vis * 0.7 + np.array([0, 255, 0]) * 0.3,  # Green tint for face
                      seg_vis)
    
    # Save segmentation visualization
    seg_vis = np.clip(seg_vis, 0, 255).astype(np.uint8)
    seg_vis = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/segmentation_result.png', seg_vis)
    
    # Try on the shirt
    try_on_shirt(person_path, shirt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Virtual Try-On Demo')
    parser.add_argument('person_image', help='Path to person image')
    parser.add_argument('shirt_image', help='Path to shirt image')
    args = parser.parse_args()
    main(args.person_image, args.shirt_image)
