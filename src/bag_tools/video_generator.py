#!/usr/bin/env python3
"""
Car Dataset Video Generator
Creates a video from car dataset images with overlaid command information.
"""

import json
import cv2
import numpy as np
import os
from pathlib import Path

def load_dataset(json_path):
    """Load the synchronized dataset JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_video_with_commands(images_dir, json_path, output_video_path, 
                              start_seq=100, end_seq=500, fps=30):
    """
    Create video from images with command information overlay.
    
    Args:
        images_dir: Path to directory containing images
        json_path: Path to synchronized dataset JSON
        output_video_path: Output video file path
        start_seq: Starting image sequence number
        end_seq: Ending image sequence number
        fps: Frames per second for output video
    """
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(json_path)
    
    # Filter data for the specified sequence range
    filtered_data = [entry for entry in dataset 
                    if start_seq <= entry['image_seq'] <= end_seq]
    
    if not filtered_data:
        print(f"No data found for sequence range {start_seq}-{end_seq}")
        return
    
    print(f"Found {len(filtered_data)} frames in sequence range {start_seq}-{end_seq}")
    
    # Sort by image sequence to ensure proper order
    filtered_data.sort(key=lambda x: x['image_seq'])
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    # Process each frame
    for i, entry in enumerate(filtered_data):
        image_filename = entry['image_filename']
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found, skipping...")
            continue
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_filename}, skipping...")
            continue
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        # Initialize video writer with first valid image dimensions
        if video_writer is None:
            height, width = img.shape[:2]
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Video dimensions: {width}x{height}")
        
        # Extract command values
        cmd_linear_x = entry['cmd_linear_x']
        cmd_angular_z = entry['cmd_angular_z']
        image_seq = entry['image_seq']
        
        # Create overlay text
        overlay_img = img.copy()
        
        # Add semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_img, 0.3, 0, overlay_img)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # White text
        
        # Format command values
        linear_text = f"Linear X: {cmd_linear_x:.3f}"
        angular_text = f"Angular Z: {cmd_angular_z:.3f}"
        seq_text = f"Sequence: {image_seq}"
        
        # Add text to image
        cv2.putText(overlay_img, seq_text, (20, 35), font, font_scale, color, thickness)
        cv2.putText(overlay_img, linear_text, (20, 65), font, font_scale, color, thickness)
        cv2.putText(overlay_img, angular_text, (20, 95), font, font_scale, color, thickness)
        
        # Add speed/direction indicator
        speed = abs(cmd_linear_x)
        if speed > 0.1:  # Moving
            direction = "FORWARD" if cmd_linear_x > 0 else "BACKWARD"
            speed_color = (0, 255, 0) if cmd_linear_x > 0 else (0, 255, 255)  # Green for forward, Yellow for backward
        else:
            direction = "STOPPED"
            speed_color = (0, 0, 255)  # Red for stopped
        
        cv2.putText(overlay_img, f"Status: {direction}", (20, 125), font, font_scale, speed_color, thickness)
        
        # Add turning indicator
        if abs(cmd_angular_z) > 0.1:
            turn_direction = "LEFT" if cmd_angular_z > 0 else "RIGHT"
            turn_color = (255, 0, 255)  # Magenta for turning
            cv2.putText(overlay_img, f"Turn: {turn_direction}", (200, 125), font, font_scale, turn_color, thickness)
        
        # Write frame to video
        video_writer.write(overlay_img)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(filtered_data)} frames...")
    
    # Clean up
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video created successfully: {output_video_path}")
    print(f"Total frames processed: {len(filtered_data)}")

def main():
    # Configuration
    images_dir = "/home/toon/car_datasets/behavior_20250702_081106_extracted/images"
    json_path = "/home/toon/car_datasets/behavior_20250702_081106_extracted/data/synchronized_dataset.json"
    output_video_path = "/home/toon/car_datasets/behavior_20250702_081106_extracted/car_behavior_video_100_500.mp4"
    
    # Sequence range
    start_seq = 100
    end_seq = 500
    
    # Video settings
    fps = 30  # Frames per second
    
    # Validate paths
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Starting video generation...")
    print(f"Images directory: {images_dir}")
    print(f"JSON file: {json_path}")
    print(f"Output video: {output_video_path}")
    print(f"Sequence range: {start_seq} to {end_seq}")
    print(f"FPS: {fps}")
    print("-" * 50)
    
    # Generate video
    create_video_with_commands(
        images_dir=images_dir,
        json_path=json_path,
        output_video_path=output_video_path,
        start_seq=start_seq,
        end_seq=end_seq,
        fps=fps
    )

if __name__ == "__main__":
    main()