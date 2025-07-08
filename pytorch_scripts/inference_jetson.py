#!/usr/bin/env python3
"""
Jetson Inference Script for Angular Control Sequence Prediction
Optimized for NVIDIA Jetson devices with GPU acceleration
"""

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
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import os
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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

class JetsonInference:
    """Optimized inference class for Jetson devices"""
    
    def __init__(self, model_path, device='cuda', use_vertical_crop=False, crop_pixels=100):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            use_vertical_crop: Whether to apply vertical cropping
            crop_pixels: Number of pixels to crop from top/bottom if using vertical crop
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_vertical_crop = use_vertical_crop
        self.crop_pixels = crop_pixels
        
        print(f"🚀 Initializing Jetson Inference Engine")
        print(f"📱 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model and configuration
        self.model, self.config = self._load_model(model_path)
        self.sequence_length = self.config.get('sequence_length', 10)
        
        # Create transforms
        self.transform = self._create_transforms()
        
        # Warm up the model
        self._warmup_model()
        
        print(f"✅ Inference engine ready!")
        print(f"📊 Sequence length: {self.sequence_length}")
        print(f"🔧 Vertical crop: {self.use_vertical_crop}")
        
    def _load_model(self, model_path):
        """Load the trained model and configuration"""
        print(f"📂 Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract config
        config = checkpoint.get('config', {})
        sequence_length = config.get('sequence_length', 10)
        
        # Create model
        model = ResNet18AngularRegressor(sequence_length=sequence_length, pretrained=False)
        
        # Load state dict (handle DataParallel module prefix)
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove module prefix from DataParallel
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📈 Training info:")
        print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"   Final train loss: {checkpoint.get('final_train_loss', 'N/A')}")
        print(f"   Final val loss: {checkpoint.get('final_val_loss', 'N/A')}")
        
        return model, config
    
    def _create_transforms(self):
        """Create image preprocessing transforms"""
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
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transforms_list)
    
    def _warmup_model(self):
        """Warm up the model with dummy input for consistent timing"""
        print("🔥 Warming up model...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            for _ in range(3):  # Multiple warmup runs
                _ = self.model(dummy_input)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Model warmed up")
    
    def predict_single_image(self, image_path, return_timing=True):
        """
        Predict angular_z sequence for a single image
        
        Args:
            image_path: Path to the input image
            return_timing: Whether to return timing information
            
        Returns:
            Dict containing predictions and optional timing info
        """
        start_time = time.time()
        
        # Load and preprocess image
        load_start = time.time()
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        load_time = time.time() - load_start
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(input_tensor)
            angular_z_sequence = predictions.squeeze().cpu().numpy()
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        result = {
            'image_path': image_path,
            'original_size': original_size,
            'angular_z_predictions': angular_z_sequence.tolist(),
            'sequence_length': self.sequence_length,
            'mean_angular_z': float(np.mean(angular_z_sequence)),
            'std_angular_z': float(np.std(angular_z_sequence)),
            'max_angular_z': float(np.max(angular_z_sequence)),
            'min_angular_z': float(np.min(angular_z_sequence))
        }
        
        if return_timing:
            result['timing'] = {
                'load_time_ms': load_time * 1000,
                'inference_time_ms': inference_time * 1000,
                'total_time_ms': total_time * 1000,
                'fps': 1.0 / total_time
            }
        
        return result
    
    def predict_from_csv(self, csv_path, image_column='image_filename', max_images=None, save_results=True):
        """
        Run inference on images specified in a CSV file
        
        Args:
            csv_path: Path to CSV file containing image paths
            image_column: Column name containing image paths
            max_images: Maximum number of images to process (None for all)
            save_results: Whether to save results to file
            
        Returns:
            List of prediction results
        """
        print(f"📊 Loading dataset from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = [col.replace('*', '') for col in df.columns]
        
        if image_column not in df.columns:
            raise ValueError(f"Column '{image_column}' not found in CSV. Available: {list(df.columns)}")
        
        # Get image paths
        image_paths = df[image_column].tolist()
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"🖼️  Processing {len(image_paths)} images...")
        
        results = []
        failed_images = []
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single_image(image_path, return_timing=True)
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0 or i == 0:
                    fps = result['timing']['fps']
                    print(f"✅ Processed {i+1}/{len(image_paths)} images | "
                          f"FPS: {fps:.1f} | "
                          f"Latest prediction mean: {result['mean_angular_z']:.3f}")
                
            except Exception as e:
                print(f"❌ Failed to process {image_path}: {e}")
                failed_images.append({'image_path': image_path, 'error': str(e)})
        
        # Calculate summary statistics
        if results:
            inference_times = [r['timing']['inference_time_ms'] for r in results]
            total_times = [r['timing']['total_time_ms'] for r in results]
            fps_values = [r['timing']['fps'] for r in results]
            
            summary = {
                'total_images': len(image_paths),
                'successful': len(results),
                'failed': len(failed_images),
                'avg_inference_time_ms': np.mean(inference_times),
                'avg_total_time_ms': np.mean(total_times),
                'avg_fps': np.mean(fps_values),
                'min_fps': np.min(fps_values),
                'max_fps': np.max(fps_values)
            }
            
            print(f"\n📈 Summary Statistics:")
            print(f"   Successful: {summary['successful']}/{summary['total_images']}")
            print(f"   Avg FPS: {summary['avg_fps']:.1f}")
            print(f"   Avg inference time: {summary['avg_inference_time_ms']:.1f}ms")
            print(f"   FPS range: {summary['min_fps']:.1f} - {summary['max_fps']:.1f}")
            
            # Save results if requested
            if save_results:
                output_dir = Path(csv_path).parent / 'inference_results'
                output_dir.mkdir(exist_ok=True)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Save detailed results
                results_file = output_dir / f'predictions_{timestamp}.json'
                with open(results_file, 'w') as f:
                    json.dump({
                        'summary': summary,
                        'results': results,
                        'failed_images': failed_images,
                        'config': self.config
                    }, f, indent=2)
                
                # Save CSV summary with error handling for numpy/pandas compatibility
                try:
                    csv_results = []
                    for r in results:
                        csv_row = {
                            'image_path': r['image_path'],
                            'mean_angular_z': float(r['mean_angular_z']),
                            'std_angular_z': float(r['std_angular_z']),
                            'max_angular_z': float(r['max_angular_z']),
                            'min_angular_z': float(r['min_angular_z']),
                            'inference_time_ms': float(r['timing']['inference_time_ms']),
                            'fps': float(r['timing']['fps'])
                        }
                        # Add individual predictions as floats
                        for j, pred in enumerate(r['angular_z_predictions']):
                            csv_row[f'angular_z_t{j+1}'] = float(pred)
                        csv_results.append(csv_row)
                    
                    csv_file = output_dir / f'predictions_summary_{timestamp}.csv'
                    
                    # Try pandas first, fallback to manual CSV writing
                    try:
                        df = pd.DataFrame(csv_results)
                        df.to_csv(csv_file, index=False)
                    except (AttributeError, TypeError) as pandas_error:
                        print(f"⚠️  Pandas CSV save failed ({pandas_error}), using manual CSV writing...")
                        # Manual CSV writing as fallback
                        import csv
                        if csv_results:
                            with open(csv_file, 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
                                writer.writeheader()
                                writer.writerows(csv_results)
                    
                    print(f"💾 Results saved:")
                    print(f"   Detailed: {results_file}")
                    print(f"   Summary CSV: {csv_file}")
                    
                except Exception as save_error:
                    print(f"⚠️  Warning: Could not save CSV summary: {save_error}")
                    print(f"💾 Detailed JSON results still saved: {results_file}")
                
        return results
    
    def predict_directory(self, image_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp'), 
                         max_images=None, save_results=True):
        """
        Run inference on all images in a directory
        
        Args:
            image_dir: Directory containing images
            extensions: Image file extensions to process
            max_images: Maximum number of images to process
            save_results: Whether to save results to file
            
        Returns:
            List of prediction results
        """
        print(f"📁 Scanning directory: {image_dir}")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Directory not found: {image_dir}")
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(f"*{ext}")))
            image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        image_paths = sorted([str(p) for p in image_paths])
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"🖼️  Found {len(image_paths)} images")
        
        if not image_paths:
            print("❌ No images found!")
            return []
        
        results = []
        failed_images = []
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single_image(image_path, return_timing=True)
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0 or i == 0:
                    fps = result['timing']['fps']
                    print(f"✅ Processed {i+1}/{len(image_paths)} | "
                          f"FPS: {fps:.1f} | "
                          f"Latest: {result['mean_angular_z']:.3f}")
                
            except Exception as e:
                print(f"❌ Failed: {Path(image_path).name}: {e}")
                failed_images.append({'image_path': image_path, 'error': str(e)})
        
        # Save results if requested
        if save_results and results:
            self._save_directory_results(results, failed_images, image_dir)
        
        return results
    
    def _save_directory_results(self, results, failed_images, image_dir):
        """Save results from directory processing"""
        output_dir = Path(image_dir) / 'inference_results'
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary
        inference_times = [r['timing']['inference_time_ms'] for r in results]
        fps_values = [r['timing']['fps'] for r in results]
        
        summary = {
            'total_images': len(results) + len(failed_images),
            'successful': len(results),
            'failed': len(failed_images),
            'avg_inference_time_ms': np.mean(inference_times),
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values)
        }
        
        # Save detailed results
        results_file = output_dir / f'predictions_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results,
                'failed_images': failed_images,
                'config': self.config
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def visualize_prediction(self, image_path, save_plot=True, show_plot=False):
        """
        Visualize prediction for a single image with sequence plot
        
        Args:
            image_path: Path to image
            save_plot: Whether to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Prediction result dictionary
        """
        # Get prediction
        result = self.predict_single_image(image_path, return_timing=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot image
        image = Image.open(image_path).convert('RGB')
        ax1.imshow(image)
        ax1.set_title(f'Input Image\n{Path(image_path).name}')
        ax1.axis('off')
        
        # Plot angular_z sequence prediction
        time_steps = np.arange(1, self.sequence_length + 1)
        angular_z_pred = result['angular_z_predictions']
        
        ax2.plot(time_steps, angular_z_pred, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Future Time Step')
        ax2.set_ylabel('Angular Z (rad/s)')
        ax2.set_title(f'Predicted Angular Z Sequence\n'
                     f'Mean: {result["mean_angular_z"]:.3f}, '
                     f'Std: {result["std_angular_z"]:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add stats text
        stats_text = f'FPS: {result["timing"]["fps"]:.1f}\n'
        stats_text += f'Inference: {result["timing"]["inference_time_ms"]:.1f}ms\n'
        stats_text += f'Min/Max: {result["min_angular_z"]:.3f}/{result["max_angular_z"]:.3f}'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            output_path = Path(image_path).parent / f'{Path(image_path).stem}_prediction.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {output_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Jetson Inference for Angular Control Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--mode', type=str, choices=['single', 'csv', 'directory'], default='single',
                        help='Inference mode: single image, CSV file, or directory')
    
    # Input arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to single image (for single mode)')
    parser.add_argument('--csv_path', type=str,
                        help='Path to CSV file containing image paths (for csv mode)')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory containing images (for directory mode)')
    parser.add_argument('--image_column', type=str, default='image_filename',
                        help='Column name containing image paths in CSV')
    
    # Processing arguments
    parser.add_argument('--max_images', type=int, default=1000,
                        help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--use_vertical_crop', action='store_true',
                        help='Apply vertical crop preprocessing')
    parser.add_argument('--crop_pixels', type=int, default=100,
                        help='Pixels to crop from top/bottom if using vertical crop')
    
    # Output arguments
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save results to file')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots (requires display)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single' and not args.image_path:
        parser.error("--image_path required for single mode")
    elif args.mode == 'csv' and not args.csv_path:
        parser.error("--csv_path required for csv mode")
    elif args.mode == 'directory' and not args.image_dir:
        parser.error("--image_dir required for directory mode")
    
    try:
        # Initialize inference engine
        engine = JetsonInference(
            model_path=args.model_path,
            device=args.device,
            use_vertical_crop=args.use_vertical_crop,
            crop_pixels=args.crop_pixels
        )
        
        print(f"\n🚀 Starting inference in {args.mode} mode...")
        
        if args.mode == 'single':
            # Single image inference
            if args.visualize:
                result = engine.visualize_prediction(
                    args.image_path, 
                    save_plot=True, 
                    show_plot=args.show_plots
                )
            else:
                result = engine.predict_single_image(args.image_path)
            
            print(f"\n📊 Prediction Results:")
            print(f"   Image: {Path(args.image_path).name}")
            print(f"   Mean Angular Z: {result['mean_angular_z']:.4f}")
            print(f"   Std Angular Z: {result['std_angular_z']:.4f}")
            print(f"   Range: [{result['min_angular_z']:.4f}, {result['max_angular_z']:.4f}]")
            if 'timing' in result:
                print(f"   FPS: {result['timing']['fps']:.1f}")
                print(f"   Inference time: {result['timing']['inference_time_ms']:.1f}ms")
            
            # Print first few predictions
            predictions = result['angular_z_predictions'][:min(5, len(result['angular_z_predictions']))]
            print(f"   First {len(predictions)} predictions: {[f'{p:.4f}' for p in predictions]}")
        
        elif args.mode == 'csv':
            # CSV file inference
            results = engine.predict_from_csv(
                args.csv_path,
                image_column=args.image_column,
                max_images=args.max_images,
                save_results=args.save_results
            )
            
            if results:
                print(f"\n✅ Successfully processed {len(results)} images from CSV")
            
        elif args.mode == 'directory':
            # Directory inference
            results = engine.predict_directory(
                args.image_dir,
                max_images=args.max_images,
                save_results=args.save_results
            )
            
            if results:
                print(f"\n✅ Successfully processed {len(results)} images from directory")
        
        print(f"\n🎉 Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()