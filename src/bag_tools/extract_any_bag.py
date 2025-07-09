#!/usr/bin/env python3
import sqlite3
import numpy as np
import cv2
import os
import json
import csv
import sys
from pathlib import Path

def extract_images(bag_path):
    print("🎬 Universal Bag Image Extractor (FIXED)")
    
    bag_path = Path(bag_path)
    print(f"📁 Input bag: {bag_path}")
    
    # Find the .db3 file
    db_files = list(bag_path.glob('*.db3*'))
    if not db_files:
        print("❌ No .db3 files found")
        return 0
    
    db_file = db_files[0]
    
    # Handle compressed files
    if db_file.suffix == '.zstd':
        print(f"🗜️  Decompressing {db_file.name}...")
        import zstandard as zstd
        
        decompressed_path = db_file.with_suffix('')
        with open(db_file, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with open(decompressed_path, 'wb') as decompressed_file:
                dctx.copy_stream(compressed_file, decompressed_file)
        
        db_file = decompressed_path
        print(f"✅ Decompressed to {db_file.name}")
    
    # Setup output directory
    output_dir = bag_path.parent / f"{bag_path.name}_extracted"
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    
    # Create directories with proper permissions
    output_dir.mkdir(exist_ok=True, mode=0o755)
    images_dir.mkdir(exist_ok=True, mode=0o755)
    data_dir.mkdir(exist_ok=True, mode=0o755)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Images directory: {images_dir}")
    
    # Connect to database
    print(f"🎒 Opening database: {db_file}")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Get image topic
    cursor.execute("SELECT id FROM topics WHERE name = '/camera/image_raw'")
    result = cursor.fetchone()
    if not result:
        print("❌ No camera topic found")
        conn.close()
        return 0
        
    image_topic_id = result[0]
    
    # Get all image messages
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (image_topic_id,))
    image_messages = cursor.fetchall()
    
    print(f"📸 Processing {len(image_messages)} images...")
    
    # FIXED PARAMETERS - Based on diagnostic results
    IMAGE_DATA_OFFSET = 56      # Header size (was correct)
    IMAGE_SIZE = 230400         # 320x240x3 (was 921600 for 640x480x3)
    WIDTH = 320                 # Actual width (was 640)
    HEIGHT = 240                # Actual height (was 480)
    
    print(f"🔧 Using parameters:")
    print(f"   📐 Resolution: {WIDTH}x{HEIGHT}")
    print(f"   📊 Image size: {IMAGE_SIZE} bytes")
    print(f"   ⚡ Offset: {IMAGE_DATA_OFFSET} bytes")
    
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
                
                # Create filename
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
                        
                        if (i + 1) % 50 == 0:
                            print(f"📸 Progress: {successful}/{i+1} images saved...")
                    else:
                        failed += 1
                        print(f"❌ Image {i}: File too small ({file_size} bytes)")
                else:
                    failed += 1
                    print(f"❌ Image {i}: Failed to write")
            else:
                failed += 1
                expected_total = IMAGE_DATA_OFFSET + IMAGE_SIZE
                print(f"❌ Image {i}: Data too small ({len(data)} < {expected_total} bytes)")
                
        except Exception as e:
            print(f"❌ Error processing image {i}: {e}")
            failed += 1
    
    conn.close()
    
    print(f"\n✅ Extraction complete!")
    print(f"   📸 Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    if successful + failed > 0:
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
    
    # Test a few images
    print(f"\n🔍 Testing first few extracted images...")
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
    if len(sys.argv) != 2:
        print("Usage: python3 extract_fixed.py <bag_directory>")
        print("Example: python3 extract_fixed.py behavior_20250709_201225")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    result = extract_images(bag_path)
    print(f"\n🎯 Final result: {result} images extracted")
    
    if result > 0:
        print(f"\n🎉 SUCCESS! Dataset ready for behavior cloning!")
        output_dir = Path(bag_path).parent / f"{Path(bag_path).name}_extracted"
        print(f"📁 Find your extracted data in: {output_dir}")
        print(f"   📸 Images: {output_dir}/images/")
        print(f"   📊 Metadata: {output_dir}/data/")
    else:
        print(f"\n❌ No images extracted. Check the error messages above.")