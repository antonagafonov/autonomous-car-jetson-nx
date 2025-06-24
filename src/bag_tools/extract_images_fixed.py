#!/usr/bin/env python3
import sqlite3
import numpy as np
import cv2
import os
import json
import csv
from pathlib import Path

def extract_images():
    print("🎬 Fixed Image Extractor")
    
    # Setup output directory (use absolute path)
    output_dir = Path("/home/toon/car_datasets/behavior_20250624_180608_extracted")
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    
    # Create directories with proper permissions
    output_dir.mkdir(exist_ok=True, mode=0o755)
    images_dir.mkdir(exist_ok=True, mode=0o755)
    data_dir.mkdir(exist_ok=True, mode=0o755)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Images directory: {images_dir}")
    
    # Connect to database (use absolute path)
    db_path = "/home/toon/car_datasets/behavior_20250624_180608/behavior_20250624_180608_0.db3"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get image topic
    cursor.execute("SELECT id FROM topics WHERE name = '/camera/image_raw'")
    result = cursor.fetchone()
    if not result:
        print("❌ No camera topic found")
        return
        
    image_topic_id = result[0]
    
    # Get all image messages
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (image_topic_id,))
    image_messages = cursor.fetchall()
    
    print(f"📸 Processing {len(image_messages)} images...")
    
    # Extract using proven format
    IMAGE_DATA_OFFSET = 56
    IMAGE_SIZE = 921600
    WIDTH = 640
    HEIGHT = 480
    
    images_metadata = []
    successful = 0
    failed = 0
    
    for i, (timestamp, data) in enumerate(image_messages):
        try:
            if len(data) >= IMAGE_DATA_OFFSET + IMAGE_SIZE:
                # Extract image data
                img_data = data[IMAGE_DATA_OFFSET:IMAGE_DATA_OFFSET + IMAGE_SIZE]
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = img_array.reshape((HEIGHT, WIDTH, 3))
                
                # Create filename with absolute path
                filename = f"image_{i:06d}.png"
                filepath = images_dir / filename
                
                # Write image with error checking
                write_success = cv2.imwrite(str(filepath), img)
                
                if write_success and filepath.exists():
                    # Verify file was actually created and has size
                    file_size = filepath.stat().st_size
                    if file_size > 1000:  # At least 1KB
                        images_metadata.append({
                            'filename': filename,
                            'timestamp': timestamp,
                            'width': WIDTH,
                            'height': HEIGHT,
                            'encoding': 'bgr8',
                            'frame_id': 'camera_link',
                            'seq': i,
                            'file_size': file_size
                        })
                        successful += 1
                        
                        if i % 50 == 0:
                            print(f"📸 Progress: {successful}/{i+1} images saved...")
                    else:
                        print(f"⚠️  Image {i} saved but file too small ({file_size} bytes)")
                        failed += 1
                else:
                    print(f"❌ Failed to write image {i} to {filepath}")
                    failed += 1
            else:
                print(f"❌ Image {i} has insufficient data")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error processing image {i}: {e}")
            failed += 1
    
    conn.close()
    
    print(f"\n✅ Extraction complete!")
    print(f"   📸 Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Success rate: {(successful/(successful+failed)*100):.1f}%")
    
    # Save metadata
    if images_metadata:
        # CSV format
        csv_path = data_dir / "images.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=images_metadata[0].keys())
            writer.writeheader()
            writer.writerows(images_metadata)
        
        # JSON format
        json_path = data_dir / "images.json"
        with open(json_path, 'w') as f:
            json.dump(images_metadata, f, indent=2)
        
        print(f"💾 Metadata saved to {data_dir}/")
        print(f"📄 CSV: {csv_path}")
        print(f"📄 JSON: {json_path}")
    
    # Test a few images
    print(f"\n�� Testing first few extracted images...")
    for i in range(min(3, successful)):
        test_path = images_dir / f"image_{i:06d}.png"
        if test_path.exists():
            test_img = cv2.imread(str(test_path))
            if test_img is not None:
                print(f"✅ image_{i:06d}.png: {test_img.shape}, range {test_img.min()}-{test_img.max()}")
            else:
                print(f"❌ image_{i:06d}.png: Could not read back")
    
    return successful

if __name__ == "__main__":
    result = extract_images()
    print(f"\n🎯 Final result: {result} images extracted")
