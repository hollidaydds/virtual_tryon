import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from posedetector import PoseDetector
from gmm import load_gmm, prepare_person_representation, tps_transform
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
    """Create a rough body mask using pose points"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Connect points to create a rough body shape
    body_parts = [
        # Torso
        [2, 5, 11, 8, 2],  # shoulders to hips
        # Arms
        [2, 3, 4],  # right arm
        [5, 6, 7],  # left arm
        # Legs
        [8, 9, 10],  # right leg
        [11, 12, 13],  # left leg
    ]
    
    for part in body_parts:
        points = []
        for i in part:
            if pose_points[i] is not None:
                points.append([int(pose_points[i][0]), int(pose_points[i][1])])
        if len(points) >= 2:
            points = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(mask, points, 1)
    
    # Dilate to create a fuller body shape
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
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
    """Create a binary mask for the torso region using pose points."""
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Convert pose points to numpy arrays
    left_shoulder = np.array(pose_points[5])
    right_shoulder = np.array(pose_points[2])
    left_hip = np.array(pose_points[11])
    right_hip = np.array(pose_points[8])
    neck = np.array(pose_points[1])
    
    # Calculate additional points for natural shirt shape
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    
    # Create sleeve points (reduced outward extension)
    left_sleeve = left_shoulder + np.array([-shoulder_width*0.15, 0])  # Reduced from 0.3
    right_sleeve = right_shoulder + np.array([shoulder_width*0.15, 0])  # Reduced from 0.3
    
    # Create points for natural shirt curve (reduced side extension)
    left_side = (left_shoulder + left_hip) / 2 + np.array([-hip_width*0.1, 0])  # Reduced from 0.2
    right_side = (right_shoulder + right_hip) / 2 + np.array([hip_width*0.1, 0])  # Reduced from 0.2
    
    # Create collar points (adjusted for better fit)
    collar_width = shoulder_width * 0.15  # Reduced from 0.2
    collar_height = collar_width * 0.3  # Reduced vertical offset
    left_collar = neck + np.array([-collar_width, -collar_height])
    right_collar = neck + np.array([collar_width, -collar_height])
    
    # Add intermediate points for smoother curves
    left_mid = (left_shoulder + left_side) / 2
    right_mid = (right_shoulder + right_side) / 2
    
    # Create polygon points with natural curves
    points = np.array([
        left_sleeve,
        left_shoulder,
        left_mid,
        left_side,
        left_hip,
        right_hip,
        right_side,
        right_mid,
        right_shoulder,
        right_sleeve,
        right_collar,
        neck,
        left_collar,
    ], dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [points], 1.0)
    
    # Apply smaller Gaussian blur for sharper edges
    mask = cv2.GaussianBlur(mask, (7, 7), 3)  # Reduced kernel size and sigma
    
    # Normalize the mask
    mask = np.clip(mask, 0, 1)
    
    return mask

def try_on_shirt(person_path, shirt_path, pose_model_path='models/graph_opt.pb', gmm_model_path='models/gmm_final.pth'):
    try:
        # Initialize models
        pose_detector = PoseDetector(model_path=pose_model_path)
        seg_model = load_segmentation_model()
        gmm_model = load_gmm(gmm_model_path)
        
        # Load person image and detect pose
        person_img = pose_detector.load_image(person_path)
        pose_points = pose_detector.detect(person_img)
        
        # Create person representation for GMM
        body_mask = create_body_mask(pose_points, person_img.shape[0], person_img.shape[1])
        torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
        face_hair_mask = create_face_hair_mask(pose_points, person_img)
        person_repr = prepare_person_representation(pose_points, torso_mask, face_hair_mask)
        
        # Segment and prepare shirt
        shirt_mask = segment_shirt(seg_model, shirt_path)
        shirt_img = cv2.imread(shirt_path)
        # Resize to VITON-HD standard size
        shirt_img = cv2.resize(shirt_img, (192, 256))
        
        # Convert shirt to tensor
        shirt_tensor = torch.FloatTensor(shirt_img.transpose(2, 0, 1)).unsqueeze(0) / 255.0
        
        # Load GMM state dict to check expected shapes
        state_dict = torch.load(gmm_model_path)
        
        # Warp shirt using GMM
        warped_shirt = gmm_model(person_repr, shirt_tensor)
        
        # Convert warped shirt back to numpy
        warped_shirt = warped_shirt.squeeze().cpu().numpy().transpose(1, 2, 0)
        warped_shirt = (warped_shirt * 255).astype(np.uint8)
        
        # Resize back to original size
        warped_shirt = cv2.resize(warped_shirt, (person_img.shape[1], person_img.shape[0]))
        
        # Apply torso mask
        torso_mask_resized = cv2.resize(torso_mask, (warped_shirt.shape[1], warped_shirt.shape[0]))
        torso_mask_resized = torso_mask_resized[..., np.newaxis]  # Add channel dimension
        warped_shirt_masked = warped_shirt * torso_mask_resized
        
        # Blend with original image
        alpha = torso_mask_resized.astype(float)
        result = warped_shirt_masked * alpha + person_img * (1 - alpha)
        
        # Save results
        cv2.imwrite('output/try_on_result.png', result)
        print("Result saved to output/try_on_result.png")
        
        # Visualize keypoints on result
        result_with_points = result.copy()
        pose_detector.draw_landmarks(result_with_points, pose_points)
        cv2.imwrite('output/try_on_result_with_points.png', result_with_points)
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
    
    # Get pose points
    pose_detector = PoseDetector(model_path='models/graph_opt.pb')
    pose_points = pose_detector.detect(person_img)
    
    if not pose_points or None in [pose_points[i] for i in [2, 5, 8, 11]]:
        print("Could not detect pose in the person image")
        return
    
    # Create pose visualization
    pose_vis = person_img.copy()
    pose_detector.draw_landmarks(pose_vis, pose_points)
    pose_vis_bgr = cv2.cvtColor(pose_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/pose_result.png', pose_vis_bgr)
    
    # Create masks
    torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
    face_hair_mask = create_face_hair_mask(pose_points, person_img)
    
    # Create segmentation visualization
    seg_vis = person_img.copy().astype(np.float32)
    
    # Create colored overlays
    torso_mask_3ch = np.stack([torso_mask] * 3, axis=-1)
    face_mask_3ch = np.stack([face_hair_mask[:,:,0]] * 3, axis=-1)
    
    # Add colored overlays
    seg_vis = np.where(torso_mask_3ch > 0, 
                      seg_vis * 0.7 + np.array([255, 0, 0]) * 0.3,  # Red tint for torso
                      seg_vis)
    seg_vis = np.where(face_mask_3ch > 0,
                      seg_vis * 0.7 + np.array([0, 255, 0]) * 0.3,  # Green tint for face
                      seg_vis)
    
    # Save segmentation visualization
    cv2.imwrite('output/segmentation_result.png', cv2.cvtColor(seg_vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Prepare person representation
    person_repr = prepare_person_representation(pose_points, torso_mask, face_hair_mask)
    
    # Load and prepare shirt
    shirt_img = cv2.imread(shirt_path)
    shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)
    shirt_img = cv2.resize(shirt_img, (192, 256))
    shirt_tensor = torch.FloatTensor(shirt_img).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Load GMM model
    gmm_model = load_gmm('models/gmm_final.pth')
    
    # Run GMM model
    with torch.no_grad():
        theta = gmm_model(person_repr, shirt_tensor)
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
        
        # Save result
        result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/try_on_result.png', result_bgr)
        print("Result saved to output/try_on_result.png")
        print("Pose detection saved to output/pose_result.png")
        print("Segmentation visualization saved to output/segmentation_result.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Virtual Try-On Demo')
    parser.add_argument('person_image', help='Path to person image')
    parser.add_argument('shirt_image', help='Path to shirt image')
    args = parser.parse_args()
    main(args.person_image, args.shirt_image)
