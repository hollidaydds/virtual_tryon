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
        try:
            # Track intermediate shapes
            for i, layer in enumerate(self.model):
                x = layer(x)
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    pass  # Removed print statement
            
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
    """Load the GMM model from a checkpoint file."""
    model = GMM(7)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
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

def prepare_person_representation(pose_points, torso_mask, face_hair_mask, height=256, width=192):
    """
    Prepare the person representation for the GMM model.
    Args:
        pose_points: List of pose keypoints
        torso_mask: Binary mask for torso region
        face_hair_mask: Binary mask for face and hair regions
        height: Target height (default: 256)
        width: Target width (default: 192)
    Returns:
        Tensor of shape (1, 7, height, width)
    """
    # Create heatmap from pose points
    heatmap = pose_points_to_heatmap(pose_points, torso_mask.shape[0], torso_mask.shape[1])
    
    # Stack inputs
    person_repr = np.concatenate([heatmap, torso_mask[..., np.newaxis], face_hair_mask], axis=2)
    
    # Convert to tensor format (B, C, H, W)
    person_repr = torch.FloatTensor(person_repr).permute(2, 0, 1).unsqueeze(0)
    
    # Resize to target size
    person_repr = F.interpolate(person_repr, size=(height, width), mode='bilinear', align_corners=True)
    
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

def tps_transform(theta, image):
    """Apply Thin Plate Spline transformation to the image."""
    batch_size = theta.size(0)
    
    # Grid size for TPS
    grid_size = 5
    
    # Create regular grid
    x = torch.linspace(-0.9, 0.9, grid_size)
    y = torch.linspace(-0.9, 0.9, grid_size)
    grid_y, grid_x = torch.meshgrid(y, x)
    target_coords = torch.stack([grid_x, grid_y], dim=-1)
    
    # Reshape target coordinates
    Y, X = target_coords.shape[:2]
    target_coords = target_coords.reshape(-1, 2)
    target_coords = target_coords.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Apply transformation
    source_coords = target_coords + theta.view(batch_size, -1, 2)
    
    # Ensure source coordinates stay within bounds
    source_coords = torch.clamp(source_coords, -0.9, 0.9)
    
    # Create sampling grid
    source_coords = source_coords.view(batch_size, Y, X, 2)
    
    # Apply grid sample with bilinear interpolation
    transformed_image = F.grid_sample(image, source_coords,
                                    mode='bilinear',
                                    padding_mode='border',
                                    align_corners=True)
    
    return transformed_image
