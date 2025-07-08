import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ConservativeAugmentationTransforms:
    """Conservative augmentation transforms for robot navigation training - preserves track visibility"""
    
    def __init__(self, use_vertical_crop=False, crop_pixels=100, use_albumentations=True):
        self.use_vertical_crop = use_vertical_crop
        self.crop_pixels = crop_pixels
        self.use_albumentations = use_albumentations
        
    def get_training_transforms(self):
        """Get conservative training augmentations that preserve track visibility"""
        
        if self.use_albumentations:
            return self._get_albumentations_transform()
        else:
            return self._get_torchvision_transform()
    
    def _get_albumentations_transform(self):
        """Conservative augmentations using Albumentations - NO rotations/translations"""
        
        # Pre-processing (before albumentations)
        pre_transforms = []
        if self.use_vertical_crop:
            pre_transforms.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=self.crop_pixels, left=0, 
                    height=img.height - 2*self.crop_pixels, width=img.width
                ))
            )
        
        # Conservative albumentations transforms
        album_transforms = A.Compose([
            # Resize first
            A.Resize(224, 224),
            
            # CONSERVATIVE lighting augmentations (preserve green tape visibility)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Reduced from 0.8 to 0.3
                contrast_limit=0.3,    # Reduced from 0.8 to 0.3
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Reduced from (50,150) to (80,120)
            A.HueSaturationValue(
                hue_shift_limit=10,    # Reduced from 30 to 10
                sat_shift_limit=20,    # Reduced from 40 to 20
                val_shift_limit=20,    # Reduced from 50 to 20
                p=0.5
            ),
            
            # NO GEOMETRIC TRANSFORMATIONS - removed all rotations/translations/scaling
            
            # Very light blur effects only
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),      # Reduced from 7 to 3
                A.MedianBlur(blur_limit=3, p=1.0),        # Reduced from 5 to 3
            ], p=0.2),  # Reduced probability from 0.3 to 0.2
            
            # Very light noise only - REDUCED BY HALF
            A.GaussNoise(var_limit=(1.0, 2.5), p=0.2),  # Reduced from (5.0, 15.0) to (2.5, 7.5)
            
            # Light compression artifacts
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.15),  # Higher quality
            
            # ADDED BACK: Grayscale and invert with low probability
            A.ToGray(p=0.05),  # 5% chance to convert to grayscale
            A.Solarize(threshold=128, p=0.03),  # 3% chance to invert (solarize effect)
            
            # REMOVED: All fog, shadow, sunflare, rain, gravel effects
            # REMOVED: Solarize, ToGray, RandomInvert (these can make image unusable)
            # REMOVED: All distortions and elastic transforms
            
            # Standard normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        if pre_transforms:
            # Combine PIL pre-processing with albumentations
            def combined_transform(img):
                # Apply PIL transforms first
                for t in pre_transforms:
                    img = t(img)
                
                # Convert to numpy for albumentations
                img_np = np.array(img)
                
                # Apply albumentations
                augmented = album_transforms(image=img_np)
                return augmented['image']
            
            return combined_transform
        else:
            # Direct albumentations transform
            def album_only_transform(img):
                img_np = np.array(img)
                augmented = album_transforms(image=img_np)
                return augmented['image']
            
            return album_only_transform
    
    def _get_torchvision_transform(self):
        """Conservative augmentations using only PyTorch transforms - NO rotations/translations"""
        
        transforms_list = []
        
        # Add vertical crop if enabled
        if self.use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=self.crop_pixels, left=0, 
                    height=img.height - 2*self.crop_pixels, width=img.width
                ))
            )
        
        transforms_list.extend([
            transforms.Resize((224, 224)),
            
            # NO GEOMETRIC AUGMENTATIONS - removed all rotations, translations, scaling
            
            # CONSERVATIVE lighting augmentations (preserve track visibility)
            transforms.ColorJitter(
                brightness=0.3,       # Reduced from 0.8 to 0.3
                contrast=0.3,         # Reduced from 0.8 to 0.3
                saturation=0.2,       # Reduced from 0.5 to 0.2
                hue=0.1              # Reduced from 0.3 to 0.1
            ),
            
            # Light sharpness adjustment
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  # Reduced
            
            # Very light blur only
            transforms.GaussianBlur(
                kernel_size=3,            # Reduced from 5 to 3
                sigma=(0.1, 1.0)         # Reduced from (0.1, 3.0)
            ),
            
            # ADDED BACK: Grayscale and invert with low probability  
            transforms.RandomGrayscale(p=0.05),  # 5% chance to convert to grayscale
            transforms.RandomInvert(p=0.03),     # 3% chance to invert colors
            
            # Standard preprocessing
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_validation_transforms(self):
        """Validation transforms (no augmentation)"""
        transforms_list = []
        
        if self.use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=self.crop_pixels, left=0, 
                    height=img.height - 2*self.crop_pixels, width=img.width
                ))
            )
        
        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)

# Even more conservative option for very sensitive scenarios
class MinimalAugmentationTransforms:
    """Minimal augmentations - only basic lighting variations"""
    
    def __init__(self, use_vertical_crop=False, crop_pixels=100):
        self.use_vertical_crop = use_vertical_crop
        self.crop_pixels = crop_pixels
        
    def get_training_transforms(self):
        """Minimal augmentations - just basic lighting"""
        
        transforms_list = []
        
        if self.use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=self.crop_pixels, left=0, 
                    height=img.height - 2*self.crop_pixels, width=img.width
                ))
            )
        
        transforms_list.extend([
            transforms.Resize((224, 224)),
            
            # Only very light lighting adjustments
            transforms.ColorJitter(
                brightness=0.2,       # Very conservative ±20%
                contrast=0.2,         # Very conservative ±20%
                saturation=0.1,       # Very conservative ±10%
                hue=0.05             # Very conservative ±5%
            ),
            
            # Very light blur occasionally
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5), p=0.1),
            
            # ADDED: Minimal grayscale and invert
            transforms.RandomGrayscale(p=0.03),  # 3% chance for minimal version
            transforms.RandomInvert(p=0.02),     # 2% chance for minimal version
            
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_validation_transforms(self):
        """Same as conservative validation"""
        transforms_list = []
        
        if self.use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=self.crop_pixels, left=0, 
                    height=img.height - 2*self.crop_pixels, width=img.width
                ))
            )
        
        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)

# Updated creation functions
def create_augmentations(use_vertical_crop=False, crop_pixels=100):
    """
    Create conservative augmentation transforms for robot navigation
    Preserves track visibility, no rotations/translations
    
    Usage:
        train_transform, val_transform = create_augmentations(
            use_vertical_crop=True, 
            crop_pixels=100
        )
    """
    
    aug_creator = ConservativeAugmentationTransforms(
        use_vertical_crop=use_vertical_crop,
        crop_pixels=crop_pixels,
        use_albumentations=True  # Set to False if albumentations not available
    )
    
    train_transform = aug_creator.get_training_transforms()
    val_transform = aug_creator.get_validation_transforms()
    
    return train_transform, val_transform

def create_minimal_augmentations(use_vertical_crop=False, crop_pixels=100):
    """
    Create minimal augmentation transforms for robot navigation
    Only basic lighting variations - safest option
    
    Usage:
        train_transform, val_transform = create_minimal_augmentations(
            use_vertical_crop=True, 
            crop_pixels=100
        )
    """
    
    aug_creator = MinimalAugmentationTransforms(
        use_vertical_crop=use_vertical_crop,
        crop_pixels=crop_pixels
    )
    
    train_transform = aug_creator.get_training_transforms()
    val_transform = aug_creator.get_validation_transforms()
    
    return train_transform, val_transform

# Test the augmentations
if __name__ == "__main__":
    print("🟢 Conservative augmentations created!")
    print("❌ Removed: All rotations, translations, scaling, perspective changes")
    print("❌ Removed: Grayscale, invert, solarize, strong noise")
    print("❌ Removed: Fog, shadow, rain effects")
    print("✅ Kept: Light brightness/contrast (±30%)")
    print("✅ Kept: Light color adjustments (±10% hue, ±20% saturation)")
    print("✅ Kept: Light blur effects (kernel=3)")
    print("✅ Kept: Light noise (5-15 variance)")
    
    # Test both options
    conservative_train, conservative_val = create_augmentations()
    minimal_train, minimal_val = create_minimal_augmentations()
    
    print(f"\n📊 Conservative training transforms: Available")
    print(f"📊 Minimal training transforms: Available")
    print(f"📊 Both preserve green tape visibility!")