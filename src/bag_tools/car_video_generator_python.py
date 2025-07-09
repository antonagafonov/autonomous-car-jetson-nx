#!/usr/bin/env python3
"""
Car Dataset Video Generator
Generates an MP4 video showing autonomous and manual angular_z values overlaid on images.
"""

import numpy as np

# Compatibility fix for numpy/pandas version issues
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int  
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

import pandas as pd
import cv2
import os
import argparse
from pathlib import Path
import math
from tqdm import tqdm

class CarDatasetVideoGenerator:
    def __init__(self, csv_path, images_dir, output_path="car_dataset_video.mp4"):
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.output_path = output_path
        self.data = None
        self.fps = 20
        self.font_scale = 0.5  # Reduced font size
        self.thickness = 1     # Reduced thickness
        
    def load_data(self):
        """Load and parse the CSV data."""
        print("Loading CSV data...")
        
        # Read CSV and clean column names (remove asterisks)
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.replace('*', '', regex=False)
        
        # Extract required columns
        required_cols = ['auto_angular_z', 'manual_angular_z', 'image_filename']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        self.data = df[required_cols].copy()
        
        # Filter for existing images
        existing_images = []
        for _, row in self.data.iterrows():
            image_path = self.images_dir / row['image_filename']
            if image_path.exists():
                existing_images.append(True)
            else:
                print(f"Warning: Image not found: {image_path}")
                existing_images.append(False)
        
        self.data = self.data[existing_images].reset_index(drop=True)
        print(f"Loaded {len(self.data)} frames with matching images")
        
    def draw_text_with_background(self, img, text, position, font_scale=0.5, 
                                 text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                                 thickness=1, padding=3):
        """Draw text with a background rectangle for better visibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Draw background rectangle
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # Draw text
        cv2.putText(img, text, position, font, font_scale, text_color, thickness)
        
        return text_height + padding * 2
    
    def draw_angular_indicator(self, img, auto_angular, manual_angular):
        """Draw circular angular velocity indicators in bottom-right corner."""
        # Position in bottom-right corner
        center_x = img.shape[1] - 70  # Smaller offset from right edge
        center_y = img.shape[0] - 70  # Offset from bottom edge
        radius = 30  # Smaller radius
        
        # Draw background circle
        cv2.circle(img, (center_x, center_y), radius, (50, 50, 50), -1)
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 1)
        
        # Draw center lines for reference (smaller)
        cv2.line(img, (center_x - radius//2, center_y), (center_x + radius//2, center_y), (100, 100, 100), 1)
        cv2.line(img, (center_x, center_y - radius//2), (center_x, center_y + radius//2), (100, 100, 100), 1)
        
        # Scale angular velocities for visualization (smaller arrows)
        scale_factor = 15  # Reduced scale factor
        max_arrow_length = radius - 8  # Ensure arrows stay within circle
        
        # Draw autonomous angular velocity arrow (green)
        if abs(auto_angular) > 0.001:  # Only draw if significant
            angle = auto_angular * scale_factor
            arrow_length = min(max_arrow_length, abs(auto_angular * 20))
            end_x = int(center_x + math.cos(angle) * arrow_length)
            end_y = int(center_y + math.sin(angle) * arrow_length)
            cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.4)
        
        # Draw manual angular velocity arrow (cyan)
        if abs(manual_angular) > 0.001:  # Only draw if significant
            angle = manual_angular * scale_factor
            arrow_length = min(max_arrow_length - 3, abs(manual_angular * 20))
            end_x = int(center_x + math.cos(angle) * arrow_length)
            end_y = int(center_y + math.sin(angle) * arrow_length)
            cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (255, 255, 0), 1, tipLength=0.4)
        
        # Draw center dot
        cv2.circle(img, (center_x, center_y), 2, (255, 255, 255), -1)
        
        # Add compact labels below the circle
        cv2.putText(img, "A", (center_x - radius - 15, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(img, "M", (center_x - radius - 15, center_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    def process_frame(self, image_path, auto_angular, manual_angular, frame_num, total_frames):
        """Process a single frame with overlays."""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Use smaller font for compact display
        small_font = self.font_scale * 0.6
        
        # Draw compact text overlays in top-left
        y_offset = 20
        
        # Frame information (very compact)
        frame_text = f"Frame: {frame_num}/{total_frames}"
        height = self.draw_text_with_background(img, frame_text, (10, y_offset), 
                                               small_font, (200, 200, 200), (0, 0, 50))
        y_offset += height + 2
        
        # Autonomous angular_z (green) - shorter format
        auto_text = f"Auto Z: {auto_angular:+6.3f}"
        height = self.draw_text_with_background(img, auto_text, (10, y_offset), 
                                               self.font_scale, (0, 255, 0), (0, 50, 0))
        y_offset += height + 2
        
        # Manual angular_z (cyan) - shorter format
        manual_text = f"Man Z: {manual_angular:+6.3f}"
        height = self.draw_text_with_background(img, manual_text, (10, y_offset), 
                                               self.font_scale, (255, 255, 0), (50, 50, 0))
        y_offset += height + 2
        
        # Difference (compact)
        diff = auto_angular - manual_angular
        diff_color = (0, 0, 255) if abs(diff) > 0.1 else (255, 255, 255)
        diff_text = f"Diff: {diff:+6.3f}"
        self.draw_text_with_background(img, diff_text, (10, y_offset), 
                                      small_font, diff_color, (50, 0, 0))
        
        # Draw angular velocity indicators in bottom-right
        self.draw_angular_indicator(img, auto_angular, manual_angular)
        
        # Add compact timestamp in bottom-left
        timestamp_text = f"Data: {os.path.basename(self.csv_path)}"
        cv2.putText(img, timestamp_text, (10, img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        return img
    
    def generate_video(self, fps=20, quality='high'):
        """Generate the video from processed frames."""
        if self.data is None:
            self.load_data()
        
        if len(self.data) == 0:
            raise ValueError("No valid data to process")
        
        self.fps = fps
        
        # Get video dimensions from first image
        first_image_path = self.images_dir / self.data.iloc[0]['image_filename']
        first_img = cv2.imread(str(first_image_path))
        height, width = first_img.shape[:2]
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if quality == 'high':
            bitrate = -1  # Use default high quality
        elif quality == 'medium':
            bitrate = 2000
        else:  # low
            bitrate = 1000
            
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        print(f"Generating video: {self.output_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Total frames: {len(self.data)}")
        
        # Process each frame
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Processing frames"):
            image_path = self.images_dir / row['image_filename']
            
            processed_frame = self.process_frame(
                image_path, 
                row['auto_angular_z'], 
                row['manual_angular_z'],
                idx + 1,
                len(self.data)
            )
            
            out.write(processed_frame)
        
        out.release()
        
        # Print save location prominently
        print("\n" + "="*60)
        print(f"🎬 VIDEO GENERATED SUCCESSFULLY!")
        print(f"📁 Save Location: {os.path.abspath(self.output_path)}")
        print(f"📏 File Size: {os.path.getsize(self.output_path) / (1024*1024):.2f} MB")
        print("="*60 + "\n")
        
        # Print statistics
        auto_mean = self.data['auto_angular_z'].mean()
        manual_mean = self.data['manual_angular_z'].mean()
        auto_std = self.data['auto_angular_z'].std()
        manual_std = self.data['manual_angular_z'].std()
        
        print(f"\nDataset Statistics:")
        print(f"Auto Angular Z   - Mean: {auto_mean:+7.4f}, Std: {auto_std:7.4f}")
        print(f"Manual Angular Z - Mean: {manual_mean:+7.4f}, Std: {manual_std:7.4f}")
        print(f"Correlation: {self.data['auto_angular_z'].corr(self.data['manual_angular_z']):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Generate video from car dataset')
    parser.add_argument('csv_path', help='Path to synchronized_dataset.csv')
    parser.add_argument('images_dir', help='Path to images directory')
    parser.add_argument('-o', '--output', default=None, 
                       help='Output video filename (default: saves to CSV directory as car_dataset_video.mp4)')
    parser.add_argument('--fps', type=int, default=20, 
                       help='Video frame rate (default: 20)')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='high',
                       help='Video quality (default: high)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    try:
        generator = CarDatasetVideoGenerator(args.csv_path, args.images_dir, args.output)
        generator.generate_video(fps=args.fps, quality=args.quality)
        print("Video generation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())