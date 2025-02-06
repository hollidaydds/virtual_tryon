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
        
        print("Person representation shape:", person_repr.shape)
        
        # Segment and prepare shirt
        shirt_mask = segment_shirt(seg_model, shirt_path)
        shirt_img = cv2.imread(shirt_path)
        # Resize to VITON-HD standard size
        shirt_img = cv2.resize(shirt_img, (192, 256))
        
        # Convert shirt to tensor
        shirt_tensor = torch.FloatTensor(shirt_img.transpose(2, 0, 1)).unsqueeze(0) / 255.0
        print("Shirt tensor shape:", shirt_tensor.shape)
        
        # Load GMM state dict to check expected shapes
        state_dict = torch.load(gmm_model_path)
        print("\nGMM model expected shapes:")
        for name, param in state_dict.items():
            print(f"{name}: {param.shape}")
        
        # Warp shirt using GMM
        warped_shirt = gmm_model(person_repr, shirt_tensor)
        print("Warped shirt shape:", warped_shirt.shape)
        
        # Convert warped shirt back to numpy
        warped_shirt = warped_shirt.squeeze().cpu().numpy().transpose(1, 2, 0)
        warped_shirt = (warped_shirt * 255).astype(np.uint8)
        
        # Save intermediate results for debugging
        cv2.imwrite('debug_warped_cloth.png', cv2.cvtColor(warped_shirt, cv2.COLOR_RGB2BGR))
        print("\nSaved warped cloth to debug_warped_cloth.png")
        
        # Save original images for comparison
        cv2.imwrite('debug_pose.png', cv2.imread(person_path))
        cv2.imwrite('debug_clothing.png', cv2.imread(shirt_path))
        print("Saved original images for comparison")
        
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Virtual Try-On Demo')
    parser.add_argument('person_image', help='Path to person image')
    parser.add_argument('shirt_image', help='Path to shirt image')
    args = parser.parse_args()

    try:
        # Load person image
        person_img = cv2.imread(args.person_image)
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Initialize pose detector and get pose points
        pose_detector = PoseDetector(model_path='models/graph_opt.pb')
        pose_points = pose_detector.detect(person_img)
        
        # Create body mask focusing on torso region
        body_mask = create_body_mask(pose_points, person_img.shape[0], person_img.shape[1])
        torso_mask = create_torso_mask(pose_points, person_img.shape[0], person_img.shape[1])
        face_hair_mask = create_face_hair_mask(pose_points, person_img)
        
        # Save masks for debugging
        cv2.imwrite('debug_body_mask.png', body_mask * 255)
        cv2.imwrite('debug_torso_mask.png', torso_mask * 255)
        cv2.imwrite('debug_face_hair_mask.png', face_hair_mask * 255)
        
        # Prepare person representation (includes pose heatmap, body mask, and face/hair mask)
        person_representation = prepare_person_representation(pose_points, torso_mask, face_hair_mask)
        print(f"\nPerson representation shape: {person_representation.shape}")

        # Load and prepare clothing image
        shirt_img = cv2.imread(args.shirt_image)
        shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)
        shirt_img = cv2.resize(shirt_img, (192, 256))
        shirt_tensor = torch.FloatTensor(shirt_img).permute(2, 0, 1).unsqueeze(0) / 255.0
        print(f"Shirt tensor shape: {shirt_tensor.shape}")

        # Save original images for comparison
        cv2.imwrite('debug_pose.png', cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('debug_clothing.png', cv2.cvtColor(shirt_img, cv2.COLOR_RGB2BGR))
        print("\nSaved original images for comparison")

        # Load GMM model
        gmm_model = load_gmm('models/gmm_final.pth')
        print("\nGMM model expected shapes:")
        for name, param in gmm_model.named_parameters():
            print(f"{name}: {param.shape}")

        # Run GMM model
        print("\nRunning GMM model...")
        with torch.no_grad():
            try:
                theta = gmm_model(person_representation, shirt_tensor)
                print(f"GMM output theta shape: {theta.shape}")
                
                # Save theta values for debugging
                theta_np = theta.cpu().numpy()
                print("\nTheta values:")
                print(f"Mean: {theta_np.mean():.3f}")
                print(f"Std: {theta_np.std():.3f}")
                print(f"Min: {theta_np.min():.3f}")
                print(f"Max: {theta_np.max():.3f}")
                
                # Apply TPS transformation
                warped_shirt = tps_transform(theta, shirt_tensor)
                print(f"\nWarped shirt tensor shape: {warped_shirt.shape}")
                
                # Convert warped shirt to numpy
                warped_shirt_np = warped_shirt.squeeze().cpu().numpy()
                print(f"Warped shirt numpy shape: {warped_shirt_np.shape}")
                
                # Save raw warped shirt for debugging
                np.save('debug_warped_shirt.npy', warped_shirt_np)
                print("Saved raw warped shirt to debug_warped_shirt.npy")
                
                # Convert to image format and apply torso mask
                warped_shirt_np = warped_shirt_np.transpose(1, 2, 0)
                print(f"After transpose shape: {warped_shirt_np.shape}")
                
                # Convert to uint8 and apply mask
                warped_shirt_np = (warped_shirt_np * 255).astype(np.uint8)
                
                # Resize warped shirt and mask to original size
                warped_shirt_full = cv2.resize(warped_shirt_np, (person_img.shape[1], person_img.shape[0]))
                
                # Create 3-channel torso mask
                torso_mask_3ch = np.stack([torso_mask] * 3, axis=-1)
                
                # Apply mask and blend
                warped_shirt_masked = warped_shirt_full * torso_mask_3ch
                result = warped_shirt_masked + person_img * (1 - torso_mask_3ch)
                
                # Save intermediate and final results
                cv2.imwrite('debug_warped_cloth.png', cv2.cvtColor(warped_shirt_full, cv2.COLOR_RGB2BGR))
                cv2.imwrite('debug_warped_masked.png', cv2.cvtColor(warped_shirt_masked, cv2.COLOR_RGB2BGR))
                cv2.imwrite('output/try_on_result.png', cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print("Result saved to output/try_on_result.png")
                
                # Visualize keypoints on result
                result_with_points = result.copy()
                pose_detector.draw_landmarks(result_with_points, pose_points)
                cv2.imwrite('output/try_on_result_with_points.png', cv2.cvtColor(result_with_points.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print("Result with keypoints saved to output/try_on_result_with_points.png")

            except Exception as e:
                print(f"\nError in GMM forward pass: {str(e)}")
                import traceback
                traceback.print_exc()
                return

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
