#!/usr/bin/env python3
import sqlite3
import numpy as np
import struct
import os
import sys
from pathlib import Path

def analyze_bag_images(bag_path):
    print("🔍 ROS Bag Image Format Analyzer")
    
    bag_path = Path(bag_path)
    print(f"📁 Analyzing bag: {bag_path}")
    
    # Find the .db3 file
    db_files = list(bag_path.glob('*.db3*'))
    if not db_files:
        print("❌ No .db3 files found")
        return
    
    db_file = db_files[0]
    print(f"🎒 Database file: {db_file}")
    
    # Connect to database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Find all topics
    print("\n📡 Available topics:")
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    for topic_id, name, topic_type in topics:
        cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
        count = cursor.fetchone()[0]
        print(f"   {topic_id}: {name} ({topic_type}) - {count} messages")
    
    # Find image topics
    image_topics = [t for t in topics if 'image' in t[1].lower() or 'sensor_msgs' in t[2]]
    
    if not image_topics:
        print("❌ No image topics found")
        conn.close()
        return
    
    print(f"\n📸 Found {len(image_topics)} potential image topic(s)")
    
    for topic_id, topic_name, topic_type in image_topics:
        print(f"\n🔍 Analyzing topic: {topic_name}")
        
        # Get first few messages
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 5", (topic_id,))
        messages = cursor.fetchall()
        
        if not messages:
            print("   ❌ No messages found")
            continue
            
        for i, (timestamp, data) in enumerate(messages):
            print(f"\n   📊 Message {i+1}:")
            print(f"      Timestamp: {timestamp}")
            print(f"      Data size: {len(data)} bytes")
            
            if len(data) < 100:
                print("      ⚠️  Data too small for image")
                continue
            
            # Try to parse ROS message header
            try:
                # ROS messages typically start with header info
                print(f"      First 64 bytes (hex): {data[:64].hex()}")
                print(f"      First 32 bytes (ascii): {repr(data[:32])}")
                
                # Look for common image formats
                analyze_image_data(data, i+1)
                
            except Exception as e:
                print(f"      ❌ Error analyzing: {e}")
    
    conn.close()

def analyze_image_data(data, msg_num):
    """Analyze potential image data to determine format"""
    print(f"      🔍 Searching for image patterns...")
    
    # Common image header sizes in ROS
    possible_offsets = [0, 12, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
    
    # Common image sizes
    common_resolutions = [
        (640, 480),   # VGA
        (1280, 720),  # HD
        (1920, 1080), # Full HD
        (320, 240),   # QVGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
    ]
    
    # Common pixel formats
    formats = [
        ('rgb8', 3),    # 3 bytes per pixel
        ('bgr8', 3),    # 3 bytes per pixel  
        ('mono8', 1),   # 1 byte per pixel
        ('rgba8', 4),   # 4 bytes per pixel
        ('bgra8', 4),   # 4 bytes per pixel
    ]
    
    found_candidates = []
    
    for offset in possible_offsets:
        if offset >= len(data):
            continue
            
        remaining_data = len(data) - offset
        
        for width, height in common_resolutions:
            for fmt_name, bytes_per_pixel in formats:
                expected_size = width * height * bytes_per_pixel
                
                if remaining_data >= expected_size:
                    # Check if this makes sense
                    end_offset = offset + expected_size
                    
                    # Basic sanity check - image data shouldn't be all zeros or all 255s
                    img_data = data[offset:end_offset]
                    if len(img_data) >= expected_size:
                        zero_count = img_data.count(0)
                        max_count = img_data.count(255)
                        total = len(img_data)
                        
                        # If less than 80% is zeros or 255s, it might be real image data
                        if zero_count < 0.8 * total and max_count < 0.8 * total:
                            found_candidates.append({
                                'offset': offset,
                                'width': width,
                                'height': height,
                                'format': fmt_name,
                                'bytes_per_pixel': bytes_per_pixel,
                                'size': expected_size,
                                'remaining_after': remaining_data - expected_size
                            })
    
    if found_candidates:
        print(f"      ✅ Found {len(found_candidates)} potential image format(s):")
        for candidate in found_candidates:
            print(f"         📐 {candidate['width']}x{candidate['height']} {candidate['format']}")
            print(f"            Offset: {candidate['offset']}, Size: {candidate['size']}, Remaining: {candidate['remaining_after']}")
    else:
        print(f"      ❌ No standard image formats detected")
        
        # Try to find patterns anyway
        print(f"      🔍 Looking for other patterns...")
        
        # Check for JPEG/PNG headers
        for i in range(min(200, len(data) - 10)):
            chunk = data[i:i+10]
            if chunk.startswith(b'\xff\xd8\xff'):  # JPEG
                print(f"         🖼️  JPEG header found at offset {i}")
            elif chunk.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                print(f"         🖼️  PNG header found at offset {i}")
        
        # Try to parse as ROS Image message
        try:
            parse_ros_image_message(data)
        except Exception as e:
            print(f"         ❌ ROS message parsing failed: {e}")

def parse_ros_image_message(data):
    """Try to parse data as a ROS sensor_msgs/Image message"""
    print(f"      🤖 Attempting ROS Image message parsing...")
    
    # ROS messages have a specific structure
    # Let's try to find the image dimensions and encoding
    
    # Look for common encoding strings
    encodings = [b'rgb8', b'bgr8', b'mono8', b'rgba8', b'bgra8']
    
    for encoding in encodings:
        idx = data.find(encoding)
        if idx != -1:
            print(f"         🎯 Found encoding '{encoding.decode()}' at offset {idx}")
            
            # Try to extract width/height (usually before encoding in ROS messages)
            # Width and height are typically uint32 values
            try:
                # Look backwards from encoding to find width/height
                search_start = max(0, idx - 100)
                search_data = data[search_start:idx]
                
                # Try different positions for width/height
                for i in range(0, len(search_data) - 8, 4):
                    try:
                        width = struct.unpack('<I', search_data[i:i+4])[0]
                        height = struct.unpack('<I', search_data[i+4:i+8])[0]
                        
                        # Sanity check
                        if 100 <= width <= 4000 and 100 <= height <= 3000:
                            bytes_per_pixel = len(encoding) - 1 if encoding.endswith(b'8') else 3
                            if encoding == b'mono8':
                                bytes_per_pixel = 1
                            elif encoding in [b'rgba8', b'bgra8']:
                                bytes_per_pixel = 4
                            
                            expected_size = width * height * bytes_per_pixel
                            print(f"         💡 Potential: {width}x{height}, encoding: {encoding.decode()}")
                            print(f"            Expected image size: {expected_size} bytes")
                            print(f"            Total message size: {len(data)} bytes")
                            
                    except:
                        continue
                        
            except Exception as e:
                print(f"         ❌ Error parsing dimensions: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 bag_diagnostics.py <bag_directory>")
        print("Example: python3 bag_diagnostics.py behavior_20250709_201225")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    analyze_bag_images(bag_path)
