# Fix for numpy/pandas compatibility issue
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from PIL import Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import ast
import json
import datetime

# Try to import tensorboard, make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("✅ TensorBoard available")
except ImportError as e:
    TENSORBOARD_AVAILABLE = False
    print(f"⚠️  TensorBoard not available: {e}")
    print("💡 To fix: pip3 install absl-py grpcio protobuf")
    SummaryWriter = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available, GradCAM visualizations will be simplified")
import torch.nn.functional as F

class RobotControlSequenceDataset(Dataset):
    """Custom dataset for robot control sequence prediction using preprocessed angular_z predictions with absolute image paths"""
    
    def __init__(self, csv_file, sequence_length=10, transform=None, is_training=True):
        """
        Args:
            csv_file (string): Path to the csv file with preprocessed angular_z predictions
            sequence_length (int): Number of future frames to predict (10, 30, or 60)
            transform (callable, optional): Optional transform to be applied on a sample
            is_training (bool): Whether this is training data (enables horizontal flip augmentation)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training
        
        # Clean column names (remove leading/trailing whitespace)
        self.data_frame.columns = self.data_frame.columns.str.strip()
        
        # Remove asterisks from column names if present
        self.data_frame.columns = [col.replace('*', '') for col in self.data_frame.columns]
        
        # Determine the column name based on sequence length
        if sequence_length == 10:
            self.target_column = 'manual_angular_z_next_10'
        elif sequence_length == 30:
            self.target_column = 'manual_angular_z_next_30'
        elif sequence_length == 60:
            self.target_column = 'manual_angular_z_next_60'
        else:
            raise ValueError(f"Sequence length {sequence_length} not supported. Use 10, 30, or 60.")
        
        # Check if the target column exists
        if self.target_column not in self.data_frame.columns:
            raise ValueError(f"Column '{self.target_column}' not found in CSV. Available columns: {list(self.data_frame.columns)}")
        
        # Filter out rows with missing target data
        self.data_frame = self.data_frame.dropna(subset=[self.target_column]).reset_index(drop=True)
        
        print(f"Dataset loaded with {len(self.data_frame)} samples")
        print(f"Sequence length: {sequence_length}")
        print(f"Using target column: {self.target_column}")
        print(f"Target: angular_z predictions for next {sequence_length} frames")
        print(f"Training mode: {is_training} (horizontal flip augmentation enabled)")
        
        # Show dataset source distribution if available
        if 'dataset_source' in self.data_frame.columns:
            print("Dataset source distribution:")
            try:
                print(self.data_frame['dataset_source'].value_counts())
            except Exception as e:
                print(f"Could not display dataset source distribution: {e}")
                # Fallback to manual counting
                source_counts = {}
                for source in self.data_frame['dataset_source']:
                    source_counts[source] = source_counts.get(source, 0) + 1
                for source, count in source_counts.items():
                    print(f"  {source}: {count}")
        else:
            print("No dataset_source column found - this may not be a combined dataset")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get absolute image path (no need to join with images_dir)
        img_path = self.data_frame.iloc[idx]['image_filename']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (640, 480), color='black')
        
        # Get preprocessed angular_z sequence
        target_str = self.data_frame.iloc[idx][self.target_column]
        
        try:
            # Parse the string representation of the list
            if isinstance(target_str, str):
                angular_z_sequence = ast.literal_eval(target_str)
            else:
                # If it's already a list or array
                angular_z_sequence = target_str
            
            # Ensure we have the expected length
            if len(angular_z_sequence) != self.sequence_length:
                print(f"Warning: Expected {self.sequence_length} values, got {len(angular_z_sequence)} for index {idx}")
                # Pad with zeros if too short, truncate if too long
                if len(angular_z_sequence) < self.sequence_length:
                    angular_z_sequence.extend([0.0] * (self.sequence_length - len(angular_z_sequence)))
                else:
                    angular_z_sequence = angular_z_sequence[:self.sequence_length]
            
            # Convert to tensor
            targets = torch.tensor(angular_z_sequence, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error parsing target sequence for index {idx}: {e}")
            print(f"Target string: {target_str}")
            # Return zeros if parsing fails
            targets = torch.zeros(self.sequence_length, dtype=torch.float32)
        
        # Apply horizontal flip augmentation during training with 0.5 probability
        if self.is_training and random.random() < 0.5:
            # Flip the image horizontally
            image = transforms.functional.hflip(image)
            # Invert the steering commands (negate angular_z values)
            targets = -targets
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, targets

class ResNet18AngularRegressor(nn.Module):
    """ResNet18 based regression model for predicting sequence of angular_z control values"""
    
    def __init__(self, sequence_length=10, pretrained=False):
        super(ResNet18AngularRegressor, self).__init__()
        
        self.sequence_length = sequence_length
        # Only predicting angular_z sequence
        self.num_outputs = sequence_length
        
        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.resnet18.fc.in_features
        
        # Replace the final fully connected layer with a more complex head
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_outputs)
        )
        
    def forward(self, x):
        output = self.resnet18(x)
        return output

def create_data_transforms(use_vertical_crop=False, crop_pixels=100):
    """Create data transforms for training and validation using PyTorch transforms"""
    
    def get_train_transforms():
        transforms_list = []
        
        # Add vertical crop if enabled (remove crop_pixels from top and bottom)
        if use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=crop_pixels, left=0, 
                    height=img.height - 2*crop_pixels, width=img.width
                ))
            )
        
        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 1.0)),
            transforms.ColorJitter(
                brightness=0.5, 
                contrast=0.5, 
                saturation=0, 
                hue=0.15
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_val_transforms():
        transforms_list = []
        
        # Add vertical crop if enabled (remove crop_pixels from top and bottom)
        if use_vertical_crop:
            transforms_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, top=crop_pixels, left=0, 
                    height=img.height - 2*crop_pixels, width=img.width
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
        
        # Note: Horizontal flip augmentation for validation is handled by the dataset's is_training parameter
        # when add_val_flip is enabled, not by transforms to ensure proper angular_z negation
        
        return transforms.Compose(transforms_list)
    
    return get_train_transforms(), get_val_transforms()

def apply_gradcam_visualization(model, val_loader, device, epoch, sequence_length=10, save_dir=None, num_samples=4, target_layer_name="layer4"):
    """
    Apply GradCAM visualization for angular_z sequence prediction model
    
    Args:
        target_layer_name: Which layer to use for GradCAM
                          - "layer4" (default): Highest semantic level (7x7 resolution)
                          - "layer3": Medium semantic level (14x14 resolution)  
                          - "layer2": Lower semantic level (28x28 resolution)
                          - "layer1": Lowest semantic level (56x56 resolution)
    """
    model.eval()
    
    # Hook to capture gradients and feature maps
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Find the target layer for GradCAM
    target_layer = None
    layer_info = ""
    
    if target_layer_name == "layer4":
        # Last conv layer in layer4 (highest semantic)
        for name, module in model.resnet18.named_modules():
            if name == "layer4.1.conv2":
                target_layer = module
                layer_info = "layer4.1.conv2 (512 filters, 7x7 output)"
                break
    elif target_layer_name == "layer3":
        # Last conv layer in layer3 (medium semantic)
        for name, module in model.resnet18.named_modules():
            if name == "layer3.1.conv2":
                target_layer = module
                layer_info = "layer3.1.conv2 (256 filters, 14x14 output)"
                break
    elif target_layer_name == "layer2":
        # Last conv layer in layer2 (lower semantic)
        for name, module in model.resnet18.named_modules():
            if name == "layer2.1.conv2":
                target_layer = module
                layer_info = "layer2.1.conv2 (128 filters, 28x28 output)"
                break
    elif target_layer_name == "layer1":
        # Last conv layer in layer1 (lowest semantic)
        for name, module in model.resnet18.named_modules():
            if name == "layer1.1.conv2":
                target_layer = module
                layer_info = "layer1.1.conv2 (64 filters, 56x56 output)"
                break
    else:
        # Fallback to last conv layer
        for name, module in model.resnet18.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                layer_info = f"Last conv layer found: {name}"
    
    if target_layer is None:
        print("Could not find target convolutional layer for GradCAM")
        return
    
    print(f"🎯 GradCAM using layer: {layer_info}")
    
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Get a batch of validation data (already preprocessed)
        data_iter = iter(val_loader)
        images, targets = next(data_iter)
        images, targets = images.to(device), targets.to(device)
        
        # Limit to num_samples
        images = images[:num_samples]
        targets = targets[:num_samples]
        
        # Create figure for visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Clear previous hooks data
            gradients.clear()
            activations.clear()
            
            # Forward pass for single image
            single_image = images[i:i+1]
            single_target = targets[i:i+1]
            
            # Forward pass
            output = model(single_image)
            
            # Focus on the mean of first few angular_z predictions
            loss = torch.mean(output[0, :min(5, sequence_length)])  # Average of first 5 predictions
            
            # Backward pass
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            if len(gradients) > 0 and len(activations) > 0:
                # Get gradients and activations
                grads = gradients[0]
                acts = activations[0]
                
                # Calculate importance weights
                weights = torch.mean(grads, dim=[2, 3], keepdim=True)
                
                # Generate CAM
                cam = torch.sum(weights * acts, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                # Normalize CAM
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Resize to input image size
                cam_resized = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
                cam_np = cam_resized.squeeze().cpu().detach().numpy()
                
                # Convert preprocessed image back to displayable format
                img_np = single_image.squeeze().cpu().detach().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                
                # Denormalize image (reverse the normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                # Plot preprocessed image (what model actually sees)
                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title(f'Model Input {i+1}\n(Preprocessed: Cropped+Resized)\nGradCAM: {layer_info}')
                axes[i, 0].axis('off')
                
                # Plot GradCAM heatmap
                im = axes[i, 1].imshow(cam_np, cmap='jet', alpha=0.8)
                axes[i, 1].set_title(f'GradCAM Heatmap {i+1}\n(Model Attention)')
                axes[i, 1].axis('off')
                plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
                
                # Plot overlay with sequence info
                axes[i, 2].imshow(img_np)
                axes[i, 2].imshow(cam_np, cmap='jet', alpha=0.4)
                
                # Show first few predictions
                pred_angular = output[0].cpu().detach().numpy()
                true_angular = single_target[0].cpu().detach().numpy()
                
                pred_str = ', '.join([f'{x:.3f}' for x in pred_angular[:3]])
                true_str = ', '.join([f'{x:.3f}' for x in true_angular[:3]])
                
                axes[i, 2].set_title(f'Attention Overlay {i+1}\nPred[0:3]: [{pred_str}]\nTrue[0:3]: [{true_str}]')
                axes[i, 2].axis('off')
                
            else:
                # If hooks didn't capture data, show preprocessed image only
                img_np = single_image.squeeze().cpu().detach().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                for j in range(3):
                    axes[i, j].imshow(img_np)
                    axes[i, j].set_title(f'Preprocessed Image {i+1} (GradCAM failed)')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        gradcam_base_dir = os.path.join(save_dir, 'gradcam_images')
        os.makedirs(gradcam_base_dir, exist_ok=True)

        gradcam_path = os.path.join(gradcam_base_dir, f'gradcam_angular_epoch_{epoch}.png')
        
        plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
        try:
            plt.show()
        except:
            print(f"Display not available, GradCAM saved as '{gradcam_path}'")
        plt.close()
        
    except Exception as e:
        print(f"Error in GradCAM visualization: {e}")
    
    finally:
        # Remove hooks
        handle_backward.remove()
        handle_forward.remove()
        model.train()

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    
    Returns:
        Dictionary with loaded training state
    """
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model state loaded from epoch {checkpoint['epoch']+1}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Scheduler state loaded")
    
    # Print checkpoint info
    print(f"📊 Checkpoint info:")
    print(f"   Epoch: {checkpoint['epoch']+1}")
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"   Learning rate: {checkpoint.get('learning_rate', 'N/A')}")
    print(f"   Sequence length: {checkpoint.get('sequence_length', 'N/A')}")
    
    return {
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
    }

def plot_training_curves(train_losses, val_losses, current_epoch, save_dir, best_epoch, best_val_loss):
    """
    Plot and save training and validation loss curves after each epoch
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        current_epoch: Current epoch number
        save_dir: Directory to save plots
        best_epoch: Epoch with best validation loss
        best_val_loss: Best validation loss value
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation losses
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    # Mark the best epoch
    if len(val_losses) > 0:
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'Best Model (Epoch {best_epoch})')
        plt.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5, 
                   marker='*', label=f'Best Val Loss: {best_val_loss:.6f}')
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'Training Progress - Epoch {current_epoch}\nAngular Z Sequence Prediction (Combined Datasets)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add current epoch info as text
    if len(train_losses) > 0 and len(val_losses) > 0:
        current_train_loss = train_losses[-1]
        current_val_loss = val_losses[-1]
        
        # Add text box with current stats
        textstr = f'Current Epoch: {current_epoch}\n'
        textstr += f'Train Loss: {current_train_loss:.6f}\n'
        textstr += f'Val Loss: {current_val_loss:.6f}\n'
        textstr += f'Best Val Loss: {best_val_loss:.6f}\n'
        textstr += f'Best Epoch: {best_epoch}'
        
        # Properties for the text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Set reasonable y-axis limits
    if len(train_losses) > 0 and len(val_losses) > 0:
        all_losses = train_losses + val_losses
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        margin = (max_loss - min_loss) * 0.1
        plt.ylim(max(0, min_loss - margin), max_loss + margin)
    
    # Save the plot
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save with epoch number for tracking progress
    plot_path = os.path.join(plots_dir, f'training_curves_epoch_{current_epoch:03d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    # Also save as the latest version (overwrite each time)
    latest_path = os.path.join(plots_dir, 'training_curves_latest.png')
    plt.savefig(latest_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.close()  # Close to free memory
    
    # Print save confirmation every 5 epochs to avoid spam
    if current_epoch % 5 == 0 or current_epoch == 1:
        print(f"📈 Training curves saved: {plot_path}")

def save_augmented_images_batch(images, targets, epoch, save_dir, dataset_type="train", max_images=8):
    """
    Save augmented images from the first batch to inspect augmentations
    
    Args:
        images: Batch of preprocessed images (tensor)
        targets: Corresponding targets (tensor)
        epoch: Current epoch number
        save_dir: Directory to save images
        dataset_type: "train" or "val"
        max_images: Maximum number of images to save per batch
    """
    if not save_dir:
        return
    
    # Create augmented images directory
    aug_dir = os.path.join(save_dir, 'augmented_images', f'epoch_{epoch:03d}')
    os.makedirs(aug_dir, exist_ok=True)
    
    # Denormalization parameters (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Limit number of images to save
    num_to_save = min(max_images, images.shape[0])
    
    for i in range(num_to_save):
        # Get single image and target
        img_tensor = images[i].cpu()
        target = targets[i].cpu()
        
        # Denormalize image
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy for saving
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Create filename with target info
        target_str = '_'.join([f'{val:.3f}' for val in target[:3]])  # First 3 angular_z values
        filename = f'{dataset_type}_img_{i:02d}_target_{target_str}.png'
        filepath = os.path.join(aug_dir, filename)
        
        # Save image
        img_pil.save(filepath)
    
    print(f"🖼️  Saved {num_to_save} {dataset_type} augmented images to: {aug_dir}")

def create_augmentation_summary(save_dir, num_epochs, max_epochs_to_show=5):
    """
    Create a summary image showing augmented images across different epochs
    """
    if not save_dir:
        return
        
    aug_base_dir = os.path.join(save_dir, 'augmented_images')
    if not os.path.exists(aug_base_dir):
        return
    
    # Get available epoch directories
    epoch_dirs = [d for d in os.listdir(aug_base_dir) if d.startswith('epoch_')]
    epoch_dirs.sort()
    
    if len(epoch_dirs) == 0:
        return
    
    # Select epochs to show (first, middle, last, etc.)
    epochs_to_show = []
    if len(epoch_dirs) <= max_epochs_to_show:
        epochs_to_show = epoch_dirs
    else:
        # Select first, last, and evenly spaced middle epochs
        indices = np.linspace(0, len(epoch_dirs)-1, max_epochs_to_show, dtype=int)
        epochs_to_show = [epoch_dirs[i] for i in indices]
    
    try:
        fig, axes = plt.subplots(len(epochs_to_show), 4, figsize=(16, 4*len(epochs_to_show)))
        if len(epochs_to_show) == 1:
            axes = axes.reshape(1, -1)
        
        for row, epoch_dir in enumerate(epochs_to_show):
            epoch_path = os.path.join(aug_base_dir, epoch_dir)
            
            # Get train and val images
            train_images = [f for f in os.listdir(epoch_path) if f.startswith('train_')]
            val_images = [f for f in os.listdir(epoch_path) if f.startswith('val_')]
            
            # Show 2 train and 2 val images
            images_to_show = train_images[:2] + val_images[:2]
            titles = ['Train 1', 'Train 2', 'Val 1', 'Val 2']
            
            for col, (img_file, title) in enumerate(zip(images_to_show, titles)):
                if col < len(images_to_show) and img_file:
                    img_path = os.path.join(epoch_path, img_file)
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        axes[row, col].imshow(img)
                        axes[row, col].set_title(f'{epoch_dir.replace("_", " ").title()}\n{title}')
                    else:
                        axes[row, col].text(0.5, 0.5, 'No Image', ha='center', va='center')
                        axes[row, col].set_title(f'{epoch_dir.replace("_", " ").title()}\n{title}')
                else:
                    axes[row, col].text(0.5, 0.5, 'No Image', ha='center', va='center')
                    axes[row, col].set_title(f'{epoch_dir.replace("_", " ").title()}\n{title}')
                
                axes[row, col].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(save_dir, 'plots', 'augmentation_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🖼️  Augmentation summary created: {summary_path}")
        
    except Exception as e:
        print(f"Warning: Could not create augmentation summary: {e}")

def train_model(model, train_loader, val_loader, sequence_length=10, num_epochs=50, learning_rate=0.001, device='cuda', save_dir=None, gradcam_freq=1, gradcam_layer='layer4', checkpoint_freq=5, start_epoch=0, initial_train_losses=None, initial_val_losses=None, initial_best_val_loss=float('inf'), save_augmented_images=False):
    """Train the angular_z sequence prediction model with checkpoint saving and resuming"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize tensorboard writer
    writer = None
    if save_dir and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
        print(f"Tensorboard logs will be saved to: {os.path.join(save_dir, 'tensorboard')}")
    elif save_dir and not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available - skipping tensorboard logging")
    
    # Initialize or resume training state
    train_losses = initial_train_losses if initial_train_losses else []
    val_losses = initial_val_losses if initial_val_losses else []
    best_val_loss = initial_best_val_loss
    best_model_state = None
    best_epoch = 0
    
    # Create checkpoints directory
    if save_dir:
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved every {checkpoint_freq} epochs to: {checkpoint_dir}")
    
    def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, is_best=False, checkpoint_type="regular"):
        """Save checkpoint with all training state"""
        if not save_dir:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'sequence_length': sequence_length,
        }
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            print(f"💾 Saving BEST checkpoint at epoch {epoch+1} (val_loss: {val_losses[-1]:.6f})")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
            print(f"💾 Saving checkpoint at epoch {epoch+1}")
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        # Flag to save augmented images from first batch only
        save_first_batch = save_augmented_images
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Save augmented images from first batch of each epoch
            if save_first_batch and batch_idx == 0:
                save_augmented_images_batch(images, targets, epoch + 1, save_dir, "train")
                save_first_batch = False  # Only save first batch
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
                
                # Log batch loss to tensorboard
                if writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/Batch', loss.item(), global_step)
        
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        # Flag to save validation augmented images from first batch only
        save_first_val_batch = save_augmented_images
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), targets.to(device)
                
                # Save augmented validation images from first batch of each epoch
                if save_first_val_batch and batch_idx == 0:
                    save_augmented_images_batch(images, targets, epoch + 1, save_dir, "val")
                    save_first_val_batch = False  # Only save first batch
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
        
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Check if this is the best model so far
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            # Save best checkpoint immediately
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, is_best=True)
        
        # Save regular checkpoint every checkpoint_freq epochs
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, is_best=False)
        
        # Apply GradCAM based on frequency setting
        if save_dir and (epoch + 1) % gradcam_freq == 0:
            print(f"Generating GradCAM visualizations for epoch {epoch + 1}...")
            apply_gradcam_visualization(model, val_loader, device, epoch + 1, sequence_length, save_dir, target_layer_name=gradcam_layer)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
        if is_best:
            print(f'  🌟 NEW BEST MODEL! 🌟')
        print('-' * 50)
        
        # Plot and save training curves after each epoch
        if save_dir:
            plot_training_curves(train_losses, val_losses, epoch + 1, save_dir, best_epoch + 1, best_val_loss)
    
    # Save final checkpoint
    save_checkpoint(num_epochs-1, model, optimizer, scheduler, train_losses, val_losses, is_best=False, checkpoint_type="final")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model from epoch {best_epoch+1} (val_loss: {best_val_loss:.6f})")
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, sequence_length=10, device='cuda', save_dir=None):
    """Evaluate the angular_z sequence prediction model"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate MSE for different time horizons
    horizons = [1, 5, 10, 15, 20, 25, 30, 60]
    
    print(f"Test Results for Angular Z Sequence Prediction:")
    print("Time Horizon | Angular Z MSE")
    print("-" * 30)
    
    for h in horizons:
        if h <= sequence_length:
            mse_angular = np.mean((all_predictions[:, :h] - all_targets[:, :h])**2)
            print(f"{h:11d} | {mse_angular:12.6f}")
    
    # Plot predictions vs ground truth for different time steps
    time_steps_to_plot = [0, 4, 9]  # 1st, 5th, 10th step
    if sequence_length >= 30:
        time_steps_to_plot.extend([14, 19, 29])  # 15th, 20th, 30th step
    if sequence_length >= 60:
        time_steps_to_plot.extend([44, 59])  # 45th, 60th step
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, step in enumerate(time_steps_to_plot[:6]):
        if step < sequence_length:
            axes[i].scatter(all_targets[:, step], all_predictions[:, step], alpha=0.6, color='orange')
            axes[i].plot([all_targets[:, step].min(), all_targets[:, step].max()], 
                        [all_targets[:, step].min(), all_targets[:, step].max()], 'r--', lw=2)
            axes[i].set_xlabel(f'True Angular Z (t+{step+1})')
            axes[i].set_ylabel(f'Pred Angular Z (t+{step+1})')
            mse = np.mean((all_predictions[:, step] - all_targets[:, step])**2)
            axes[i].set_title(f'Angular Z Step {step+1} (MSE: {mse:.6f})')
            axes[i].grid(True)
    
    # Hide unused subplots
    for i in range(len(time_steps_to_plot), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    predictions_plot_path = os.path.join(save_dir, 'plots', 'predictions_vs_ground_truth.png') if save_dir else 'angular_predictions_vs_ground_truth.png'
    plt.savefig(predictions_plot_path, dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except:
        print(f"Display not available, plot saved as '{predictions_plot_path}'")
    
    # Plot sequence evolution for a few examples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Show first 4 examples
    for i in range(min(4, len(all_predictions))):
        row = i // 2
        col = i % 2
        
        time_steps = np.arange(1, sequence_length + 1)
        
        # Plot angular_z sequence
        axes[row, col].plot(time_steps, all_targets[i], 'r-', label='True Angular Z', linewidth=2)
        axes[row, col].plot(time_steps, all_predictions[i], 'r--', label='Pred Angular Z', linewidth=2)
        
        axes[row, col].set_xlabel('Future Time Step')
        axes[row, col].set_ylabel('Angular Z')
        axes[row, col].set_title(f'Angular Z Sequence Prediction Example {i+1}')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    evolution_plot_path = os.path.join(save_dir, 'plots', 'sequence_evolution_examples.png') if save_dir else 'angular_sequence_evolution_examples.png'
    plt.savefig(evolution_plot_path, dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except:
        print(f"Display not available, plot saved as '{evolution_plot_path}'")
    
    return all_predictions, all_targets

def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 for Angular Z Sequence Prediction on Combined Datasets')
    parser.add_argument('--csv_path', type=str, 
                        default="/home/toon/car_datasets/combined_balanced_synchronized_dataset.csv",
                        help='Path to the combined CSV file with absolute image paths')
    parser.add_argument('--sequence_length', type=int, default=10, choices=[10, 30, 60],
                        help='Number of future frames to predict (10, 30, or 60)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (32 recommended for Xavier NX)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_vertical_crop', action='store_true',
                        help='Use vertical crop (remove pixels from top and bottom) (default: False)')
    parser.add_argument('--crop_pixels', type=int, default=100,
                        help='Number of pixels to remove from top and bottom when using vertical crop')
    parser.add_argument('--save_dir', type=str, default='/home/toon/train_results',
                        help='Base directory to save training results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--gradcam_freq', type=int, default=1,
                        help='Generate GradCAM visualizations every N epochs (default: 1)')
    parser.add_argument('--gradcam_layer', type=str, default='layer2', 
                        choices=['layer1', 'layer2', 'layer3', 'layer4'],
                        help='Which ResNet18 layer to use for GradCAM (default: layer2)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--train_size', type=float, default=0.7,
                        help='Fraction of dataset to use for training (default: 0.7)')
    parser.add_argument('--val_size', type=float, default=0.25,
                        help='Fraction of dataset to use for validation (default: 0.25)')
    parser.add_argument('--add_val_flip', action='store_true',
                        help='Add horizontal flip augmentation to validation set (default: False)')
    parser.add_argument('--save_augmented_images', action='store_true',
                        help='Save augmented images from first batch of each epoch for inspection (default: False)')
    
    args = parser.parse_args()
    
    # Create unique experiment directory
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"combined_angular_seq{args.sequence_length}_{timestamp}"
    
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    
    # Initialize training state
    start_epoch = 0
    initial_train_losses = []
    initial_val_losses = []
    initial_best_val_loss = float('inf')
     
    # Validate dataset size arguments
    if args.train_size + args.val_size >= 1.0:
        raise ValueError(f"train_size ({args.train_size}) + val_size ({args.val_size}) must be < 1.0 to leave room for test set")
    
    test_size = 1.0 - args.train_size - args.val_size
    if test_size < 0.05:  # Ensure at least 5% for test set
        raise ValueError(f"Calculated test_size ({test_size:.3f}) is too small. Reduce train_size or val_size.")
    
    print(f"Dataset split ratios: Train={args.train_size:.2f}, Val={args.val_size:.2f}, Test={test_size:.2f}")
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Using combined dataset: {args.csv_path}")
    print(f"Predicting next {args.sequence_length} angular_z frames")
    print(f"Use vertical crop: {args.use_vertical_crop}")
    if args.use_vertical_crop:
        print(f"Crop pixels (top/bottom): {args.crop_pixels}")
    print(f"Validation flip augmentation: {args.add_val_flip}")
    print(f"Save augmented images: {args.save_augmented_images}")
    print(f"GradCAM frequency: every {args.gradcam_freq} epoch(s)")
    print(f"GradCAM layer: {args.gradcam_layer}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} epoch(s)")
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms(args.use_vertical_crop, args.crop_pixels)
    
    # Create datasets (no images_dir needed since paths are absolute)
    train_dataset = RobotControlSequenceDataset(args.csv_path, args.sequence_length, train_transform, is_training=True)
    # For validation dataset, enable training mode only if val flip is enabled (to apply flip augmentation)
    val_dataset = RobotControlSequenceDataset(args.csv_path, args.sequence_length, val_transform, is_training=args.add_val_flip)
    test_dataset = RobotControlSequenceDataset(args.csv_path, args.sequence_length, val_transform, is_training=False)
    
    if args.add_val_flip:
        print("🔄 Validation horizontal flip augmentation enabled (p=0.5) with angular_z negation")
    
    # Split dataset indices using configurable sizes
    indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    train_size = int(args.train_size * len(indices))
    val_size = int(args.val_size * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    actual_train_ratio = len(train_indices) / len(indices)
    actual_val_ratio = len(val_indices) / len(indices)
    actual_test_ratio = len(test_indices) / len(indices)
    
    print(f"Dataset split: Train={len(train_indices)} ({actual_train_ratio:.3f}), Val={len(val_indices)} ({actual_val_ratio:.3f}), Test={len(test_indices)} ({actual_test_ratio:.3f})")
    print(f"Total samples: {len(indices)}")
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders (optimized for Xavier NX)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Test data loading
    print("\nTesting data loading...")
    for i, (image, targets) in enumerate(train_loader):
        print(f"Batch {i}: Image shape: {image.shape}, Targets shape: {targets.shape}")
        print(f"Expected targets shape: [batch_size, {args.sequence_length}]")
        print(f"Sample target (first 6 values): {targets[0, :6]}")
        if i >= 1:
            break
    
    # Create model
    model = ResNet18AngularRegressor(sequence_length=args.sequence_length, pretrained=True)
    model = model.to(DEVICE)
    
    print(f"Model architecture:\n {model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Output size: {args.sequence_length} (angular_z predictions)")
    
    # Save training configuration
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment_name': args.experiment_name,
        'csv_path': args.csv_path,
        'sequence_length': args.sequence_length,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'use_vertical_crop': args.use_vertical_crop,
        'crop_pixels': args.crop_pixels,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': test_size,
        'add_val_flip': args.add_val_flip,
        'save_augmented_images': args.save_augmented_images,
        'actual_train_ratio': actual_train_ratio,
        'actual_val_ratio': actual_val_ratio,
        'actual_test_ratio': actual_test_ratio,
        'total_samples': len(indices),
        'model_architecture': 'ResNet18AngularRegressor',
        'pretrained': True,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'num_workers': 2,
        'pin_memory': True,
        'gradcam_freq': args.gradcam_freq,
        'gradcam_layer': args.gradcam_layer,
        'checkpoint_freq': args.checkpoint_freq,
        'resume_from': args.resume_from,
        'start_epoch': start_epoch,
        'uses_combined_datasets': True,
        'absolute_image_paths': True
    }
    
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_path}")

    # Resume from checkpoint if specified
    if args.resume_from:
        if os.path.exists(args.resume_from):
            # Create optimizer and scheduler for loading checkpoint
            temp_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            temp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(temp_optimizer, mode='min', factor=0.5, patience=5)
            
            checkpoint_info = load_checkpoint(args.resume_from, model, temp_optimizer, temp_scheduler)
            start_epoch = checkpoint_info['epoch'] + 1
            initial_train_losses = checkpoint_info['train_losses']
            initial_val_losses = checkpoint_info['val_losses']
            initial_best_val_loss = checkpoint_info['best_val_loss']
            
            print(f"🔄 Resuming training from epoch {start_epoch + 1}")
            print(f"📊 Previous best validation loss: {initial_best_val_loss:.6f}")
        else:
            print(f"❌ Checkpoint file not found: {args.resume_from}")
            print("🆕 Starting fresh training...")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        device=DEVICE,
        save_dir=experiment_dir,
        gradcam_freq=args.gradcam_freq,
        gradcam_layer=args.gradcam_layer,
        checkpoint_freq=args.checkpoint_freq,
        start_epoch=start_epoch,
        initial_train_losses=initial_train_losses,
        initial_val_losses=initial_val_losses,
        initial_best_val_loss=initial_best_val_loss,
        save_augmented_images=args.save_augmented_images
    )
    
    # Plot final training curves summary
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark the best epoch
    best_epoch_idx = val_losses.index(min(val_losses))
    best_epoch_num = best_epoch_idx + 1
    plt.axvline(x=best_epoch_num, color='green', linestyle='--', alpha=0.7, linewidth=2, 
               label=f'Best Model (Epoch {best_epoch_num})')
    plt.scatter([best_epoch_num], [min(val_losses)], color='green', s=150, zorder=5, 
               marker='*', label=f'Best Val Loss: {min(val_losses):.6f}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Final Training Summary - Angular Z Sequence Prediction (Combined Datasets)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add final stats text box
    textstr = f'Total Epochs: {len(train_losses)}\n'
    textstr += f'Final Train Loss: {train_losses[-1]:.6f}\n'
    textstr += f'Final Val Loss: {val_losses[-1]:.6f}\n'
    textstr += f'Best Val Loss: {min(val_losses):.6f}\n'
    textstr += f'Best Epoch: {best_epoch_num}'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    final_curves_path = os.path.join(experiment_dir, 'plots', 'final_training_curves.png')
    plt.savefig(final_curves_path, dpi=300, bbox_inches='tight', facecolor='white')
    try:
        plt.show()
    except:
        print(f"Display not available, final plot saved as '{final_curves_path}'")
    plt.close()
    
    print(f"📈 Final training curves saved: {final_curves_path}")
    print(f"📊 Per-epoch training curves available in: {os.path.join(experiment_dir, 'plots', 'training_curves_epoch_*.png')}")
    
    # Create augmentation summary if images were saved
    if args.save_augmented_images:
        print("Creating augmentation summary...")
        create_augmentation_summary(experiment_dir, args.num_epochs)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    predictions, targets = evaluate_model(model, test_loader, sequence_length=args.sequence_length, device=DEVICE, save_dir=experiment_dir)
    
    # Save model
    model_filename = f'resnet18_angular_control_model_combined_{args.sequence_length}frames.pth'
    model_path = os.path.join(experiment_dir, 'models', model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': min(val_losses)
    }, model_path)
    
    print(f"Model saved as '{model_path}'")
    
    # Save training summary
    summary = {
        'experiment_name': args.experiment_name,
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'best_val_loss': float(min(val_losses)),
        'total_epochs': args.num_epochs,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'uses_combined_datasets': True,
        'csv_path': args.csv_path
    }
    
    summary_path = os.path.join(experiment_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("🎉 TRAINING ON COMBINED DATASETS COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"📊 Tensorboard logs: {os.path.join(experiment_dir, 'tensorboard')}")
    print(f"🤖 Best model: {model_path}")
    print(f"💾 Checkpoints: {os.path.join(experiment_dir, 'checkpoints')}")
    print(f"📈 Final training curves: {final_curves_path}")
    print(f"📊 Per-epoch plots: {os.path.join(experiment_dir, 'plots', 'training_curves_epoch_*.png')}")
    print(f"📈 Latest training plot: {os.path.join(experiment_dir, 'plots', 'training_curves_latest.png')}")
    if args.save_augmented_images:
        print(f"🖼️  Augmented images: {os.path.join(experiment_dir, 'augmented_images')}")
    print(f"📋 Config: {config_path}")
    print(f"📄 Summary: {summary_path}")
    print(f"\n💾 Checkpoint files:")
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        for cf in checkpoint_files:
            print(f"   - {cf}")
    print("\n🔥 To view tensorboard:")
    print(f"   tensorboard --logdir {os.path.join(experiment_dir, 'tensorboard')}")
    print("\n🔄 To resume training:")
    print(f"   python train_combined_datasets.py --resume_from {os.path.join(checkpoint_dir, 'best_checkpoint.pth')} --num_epochs 100")
    print("="*60)

if __name__ == "__main__":
    main()