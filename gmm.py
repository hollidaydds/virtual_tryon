import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3):
        super(FeatureExtraction, self).__init__()
        
        # Create sequential model with exact layer indices to match pre-trained model
        model = []
        
        # Layer 0: Initial convolution
        model.append(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1))
        
        # Layer 1-2: First normalization block
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(ngf))
        
        # Layer 3: Second convolution
        model.append(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1))
        
        # Layer 4-5: Second normalization block
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(ngf * 2))
        
        # Layer 6: Third convolution
        model.append(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1))
        
        # Layer 7-8: Third normalization block
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(ngf * 4))
        
        # Layer 9: Fourth convolution
        model.append(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1))
        
        # Layer 10-11: Fourth normalization block
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(ngf * 8))
        
        # Layer 12: Fifth convolution
        model.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1))
        
        # Layer 13-14: Fifth normalization block
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(ngf * 8))
        
        # Layer 15: Final convolution
        model.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1))
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        print(f"\nFeature Extraction Forward Pass:")
        print(f"Input shape: {x.shape}")
        
        try:
            # Track intermediate shapes
            for i, layer in enumerate(self.model):
                x = layer(x)
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    print(f"Layer {i} ({layer.__class__.__name__}) output shape: {x.shape}")
            
            print(f"Final output shape: {x.shape}")
            return x
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            print(f"Error occurred at layer {i}: {layer.__class__.__name__}")
            raise

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
        
    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        
        # Normalize features
        feature_A = F.normalize(feature_A, p=2, dim=1)
        feature_B = F.normalize(feature_B, p=2, dim=1)
        
        # Reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, -1)  # b x c x hw
        feature_B = feature_B.view(b, c, -1).transpose(1, 2)  # b x hw x c
        
        # Compute correlation
        correlation = torch.bmm(feature_B, feature_A)  # b x hw x hw
        
        # Reshape to match regression network input
        correlation = correlation.view(b, h * w, h, w)  # Reshape to match pre-trained model
        
        return correlation

class FeatureRegression(nn.Module):
    def __init__(self):
        super(FeatureRegression, self).__init__()
        
        # Create sequential model to match pre-trained layer names
        self.conv = nn.Sequential(
            # First block - reduce spatial dimensions
            nn.Conv2d(192, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Second block - further reduce
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Third block - maintain spatial size
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Fourth block - final features
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        # Final linear layer to predict TPS parameters
        self.linear = nn.Linear(768, 50)  # Match pre-trained model shape
        
    def forward(self, x):
        # Apply convolutions
        x = self.conv(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (12, 1))  # Adjust pooling size to match pre-trained model
        
        # Reshape and apply linear layer
        x = x.reshape(x.size(0), -1)  # Flatten to vector
        x = self.linear(x)
        
        return x

class GMM(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(GMM, self).__init__()
        
        # Feature extraction for person and clothing images
        self.extractionA = FeatureExtraction(input_nc, ngf=64, n_layers=3)  # 7 channels for person (3 RGB + 4 pose)
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3)  # 3 channels for clothing
        
        # Feature correlation and regression
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression()
        
    def forward(self, person, clothing):
        try:
            # Write debug info to file
            with open('debug_gmm.log', 'w') as f:
                f.write("\nGMM Forward Pass:\n")
                f.write(f"Input person shape: {person.shape}\n")
                f.write(f"Input clothing shape: {clothing.shape}\n")
                f.write(f"person.dtype: {person.dtype}\n")
                f.write(f"clothing.dtype: {clothing.dtype}\n")
                f.write(f"person min/max: {person.min():.2f}/{person.max():.2f}\n")
                f.write(f"clothing min/max: {clothing.min():.2f}/{clothing.max():.2f}\n")
            
            # Extract features
            person_features = self.extractionA(person)
            clothing_features = self.extractionB(clothing)
            
            with open('debug_gmm.log', 'a') as f:
                f.write(f"\nExtracted person features shape: {person_features.shape}\n")
                f.write(f"Extracted clothing features shape: {clothing_features.shape}\n")
                f.write(f"person_features min/max: {person_features.min():.2f}/{person_features.max():.2f}\n")
                f.write(f"clothing_features min/max: {clothing_features.min():.2f}/{clothing_features.max():.2f}\n")
            
            # Compute correlation between features
            correlation = self.correlation(person_features, clothing_features)
            
            with open('debug_gmm.log', 'a') as f:
                f.write(f"\nCorrelation output shape: {correlation.shape}\n")
                f.write(f"correlation min/max: {correlation.min():.2f}/{correlation.max():.2f}\n")
            
            # Predict TPS parameters
            theta = self.regression(correlation)
            
            with open('debug_gmm.log', 'a') as f:
                f.write(f"\nFinal theta shape: {theta.shape}\n")
                f.write(f"theta min/max: {theta.min():.2f}/{theta.max():.2f}\n")
            
            return theta
            
        except Exception as e:
            with open('debug_gmm.log', 'a') as f:
                f.write("\nError in GMM forward pass:\n")
                import traceback
                traceback.print_exc(file=f)
            raise

def load_gmm(model_path):
    """Load a pre-trained GMM model"""
    model = GMM(7)
    print("\nLoading GMM model from:", model_path)
    state_dict = torch.load(model_path)
    
    # Compare keys
    pretrained_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    
    print("\nKeys analysis:")
    missing_in_model = pretrained_keys - model_keys
    missing_in_pretrained = model_keys - pretrained_keys
    
    if missing_in_model:
        print("Keys in pre-trained but missing in model:")
        for key in sorted(missing_in_model):
            print(f"  {key}")
    
    if missing_in_pretrained:
        print("\nKeys in model but missing in pre-trained:")
        for key in sorted(missing_in_pretrained):
            print(f"  {key}")
    
    # Check for shape mismatches
    print("\nShape mismatches:")
    common_keys = pretrained_keys.intersection(model_keys)
    mismatches = []
    for key in common_keys:
        pretrained_shape = state_dict[key].shape
        model_shape = model.state_dict()[key].shape
        if pretrained_shape != model_shape:
            mismatches.append((key, pretrained_shape, model_shape))
    
    if mismatches:
        for key, pre_shape, mod_shape in sorted(mismatches):
            print(f"  {key}:")
            print(f"    Pre-trained: {pre_shape}")
            print(f"    Model: {mod_shape}")
    else:
        print("  No shape mismatches found")
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print("\nSuccessfully loaded state dict")
    except Exception as e:
        print("\nError loading state dict:", str(e))
        raise
    
    model.eval()
    return model

def pose_points_to_heatmap(pose_points, h=256, w=192, sigma=6):
    """
    Convert pose points to heatmap representation.
    Only uses shoulder and neck points (indices 2, 5, 1).
    """
    # Create pose heatmaps (only for shoulders and neck)
    pose_map = np.zeros((h, w, 3), dtype=np.float32)
    shoulder_neck_indices = [2, 5, 1]  # Left shoulder, right shoulder, neck
    
    # Create coordinate grids
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    for i, idx in enumerate(shoulder_neck_indices):
        if pose_points[idx] is not None:
            x, y = pose_points[idx]
            if x is not None and y is not None:
                # Generate gaussian heatmap
                heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
                pose_map[..., i] = heatmap
    
    return pose_map

def prepare_person_representation(pose_points, body_mask, face_hair_mask):
    """
    Create a person representation for the GMM model.
    Input:
        pose_points: numpy array of pose keypoints
        body_mask: binary mask of body
        face_hair_mask: binary mask of face and hair
    Output:
        person_repr: tensor of shape (1, 7, H, W)
    """
    # Get target size
    h, w = body_mask.shape[:2]
    
    # Convert pose points to heatmap
    heatmap = pose_points_to_heatmap(pose_points, h=h, w=w)
    
    # Ensure body_mask is 3D
    if len(body_mask.shape) == 2:
        body_mask = body_mask[..., np.newaxis]
    
    # Ensure face_hair_mask is the right shape
    if len(face_hair_mask.shape) == 2:
        face_hair_mask = face_hair_mask[..., np.newaxis]
    elif face_hair_mask.shape[-1] != 3:
        face_hair_mask = np.repeat(face_hair_mask[..., np.newaxis], 3, axis=-1)
    
    # Print shapes for debugging
    print(f"\nShape check:")
    print(f"heatmap shape: {heatmap.shape}")
    print(f"body_mask shape: {body_mask.shape}")
    print(f"face_hair_mask shape: {face_hair_mask.shape}")
    
    # Stack inputs
    person_repr = np.concatenate([
        heatmap,  # 3 channels for pose heatmap
        body_mask,  # 1 channel for body mask
        face_hair_mask  # 3 channels for face/hair mask
    ], axis=-1)
    
    print(f"person_repr shape after concat: {person_repr.shape}")
    
    # Convert to tensor and add batch dimension
    person_repr = torch.FloatTensor(person_repr.transpose(2, 0, 1)).unsqueeze(0)
    
    print(f"person_repr shape after transpose: {person_repr.shape}")
    
    # Normalize to [0, 1] range
    person_repr = person_repr / 255.0
    
    # Ensure final size is correct
    person_repr = F.interpolate(person_repr, size=(256, 192), mode='bilinear', align_corners=False)
    
    print(f"Final person_repr shape: {person_repr.shape}")
    
    return person_repr

def warp_clothing(gmm_model, person_repr, clothing_img):
    """
    Warp clothing image using GMM prediction
    Args:
        gmm_model: Loaded GMM model
        person_repr: Person representation tensor
        clothing_img: RGB clothing image tensor
    Returns:
        Warped clothing image
    """
    with torch.no_grad():
        theta = gmm_model(person_repr, clothing_img)
        warped_cloth = tps_transform(theta, clothing_img)
        return warped_cloth

def tps_transform(theta, clothing, target_height=256, target_width=192):
    """
    Apply Thin Plate Spline transformation to warp the clothing image.
    Args:
        theta: Tensor of shape (batch_size, 2 * num_control_points)
        clothing: Tensor of shape (batch_size, channels, height, width)
    Returns:
        Warped clothing tensor of shape (batch_size, channels, height, width)
    """
    batch_size = theta.size(0)
    num_control_points = theta.size(1) // 2
    
    # Create control points grid
    grid_size = int(np.sqrt(num_control_points))
    target_control_points = []
    
    # Create uniform grid of control points with more padding to reduce edge distortion
    padding = 0.15  # Increased padding
    for i in range(grid_size):
        for j in range(grid_size):
            x = 2 * ((j / (grid_size - 1)) * (1 - 2*padding) + padding - 0.5)
            y = 2 * ((i / (grid_size - 1)) * (1 - 2*padding) + padding - 0.5)
            target_control_points.append([x, y])
    
    target_control_points = torch.FloatTensor(target_control_points)
    target_control_points = target_control_points.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Reshape and normalize source control points from theta with reduced range
    source_control_points = theta.view(batch_size, num_control_points, 2)
    source_control_points = torch.tanh(source_control_points) * 0.8  # Reduced range to [-0.8, 0.8]
    
    # Create sampling grid
    x = torch.linspace(-1, 1, target_width)
    y = torch.linspace(-1, 1, target_height)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    grid_points = grid_points.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Compute TPS weights with increased regularization
    D = torch.cdist(target_control_points, target_control_points)
    K = D * torch.log(D + 1e-6)
    P = torch.cat([torch.ones(batch_size, num_control_points, 1), target_control_points], dim=2)
    
    # Increase regularization to prevent extreme deformations
    reg_lambda = 0.001  # Increased from 0.0005
    L = torch.zeros(batch_size, num_control_points + 3, num_control_points + 3)
    L[:, :num_control_points, :num_control_points] = K + reg_lambda * torch.eye(num_control_points)
    L[:, :num_control_points, num_control_points:] = P
    L[:, num_control_points:, :num_control_points] = P.transpose(1, 2)
    L[:, num_control_points:, num_control_points:] = torch.zeros(batch_size, 3, 3)
    
    Y = torch.cat([source_control_points, torch.zeros(batch_size, 3, 2)], dim=1)
    weights = torch.linalg.solve(L, Y)
    
    # Apply transformation
    D = torch.cdist(grid_points, target_control_points)
    K = D * torch.log(D + 1e-6)
    P = torch.cat([torch.ones(batch_size, target_height * target_width, 1), grid_points], dim=2)
    
    transformed_points = torch.bmm(K, weights[:, :num_control_points, :]) + torch.bmm(P, weights[:, num_control_points:, :])
    transformed_points = transformed_points.view(batch_size, target_height, target_width, 2)
    
    # Constrain transformed points to prevent extreme distortions
    transformed_points = torch.tanh(transformed_points) * 0.9  # Further limit the range
    
    # Use grid_sample with bicubic interpolation for better detail preservation
    warped_cloth = F.grid_sample(
        clothing, 
        transformed_points, 
        mode='bicubic',
        padding_mode='border',
        align_corners=True
    )
    
    return warped_cloth
