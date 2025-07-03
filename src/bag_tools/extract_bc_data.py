#!/usr/bin/env python3
import sqlite3
import numpy as np
import cv2
import os
import json
import csv
import sys
import struct
from pathlib import Path

def decode_twist_message(data):
    """Decode a geometry_msgs/msg/Twist message from CDR format"""
    try:
        # CDR format: header (4 bytes) + linear (3 float64) + angular (3 float64)
        if len(data) < 52:  # 4 + 24 + 24 = 52 bytes minimum
            return None
        
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Extract linear velocity (3 x float64 = 24 bytes)
        linear_x = struct.unpack('<d', data[offset:offset+8])[0]
        linear_y = struct.unpack('<d', data[offset+8:offset+16])[0]
        linear_z = struct.unpack('<d', data[offset+16:offset+24])[0]
        offset += 24
        
        # Extract angular velocity (3 x float64 = 24 bytes)
        angular_x = struct.unpack('<d', data[offset:offset+8])[0]
        angular_y = struct.unpack('<d', data[offset+8:offset+16])[0]
        angular_z = struct.unpack('<d', data[offset+16:offset+24])[0]
        
        return {
            'linear': {'x': linear_x, 'y': linear_y, 'z': linear_z},
            'angular': {'x': angular_x, 'y': angular_y, 'z': angular_z}
        }
    except Exception as e:
        print(f"Error decoding Twist message: {e}")
        return None

def decode_joy_message(data):
    """Decode a sensor_msgs/msg/Joy message from CDR format"""
    try:
        if len(data) < 12:  # Minimum size check
            return None
        
        offset = 4  # Skip CDR header
        
        # Read header (std_msgs/Header)
        # Skip header for now (timestamp + frame_id)
        # This is a simplified version - full header parsing would be more complex
        
        # Find the axes array
        # Joy message structure: header + axes[] + buttons[]
        # We'll look for the typical joy message pattern
        
        # For now, let's try a simple approach for common joystick messages
        # Most joy messages have a predictable structure
        
        # Skip to where axes data typically starts (after header)
        axes_offset = 24  # Approximate offset after header
        
        if len(data) < axes_offset + 32:  # Need at least space for some axes
            return None
        
        # Try to extract some axes (assuming 4-8 axes typical)
        axes = []
        buttons = []
        
        # This is a simplified extraction - real CDR parsing would be more robust
        try:
            # Read array length for axes
            if axes_offset + 4 <= len(data):
                axes_length = struct.unpack('<I', data[axes_offset:axes_offset+4])[0]
                axes_offset += 4
                
                # Read axes data (float32 values)
                for i in range(min(axes_length, 8)):  # Limit to reasonable number
                    if axes_offset + 4 <= len(data):
                        axis_value = struct.unpack('<f', data[axes_offset:axes_offset+4])[0]
                        axes.append(axis_value)
                        axes_offset += 4
                
                # Try to read buttons array
                if axes_offset + 4 <= len(data):
                    buttons_length = struct.unpack('<I', data[axes_offset:axes_offset+4])[0]
                    axes_offset += 4
                    
                    for i in range(min(buttons_length, 12)):  # Limit to reasonable number
                        if axes_offset + 1 <= len(data):
                            button_value = struct.unpack('<B', data[axes_offset:axes_offset+1])[0]
                            buttons.append(int(button_value))
                            axes_offset += 1
        except:
            # If structured parsing fails, return basic info
            pass
        
        return {
            'axes': axes,
            'buttons': buttons
        }
    except Exception as e:
        print(f"Error decoding Joy message: {e}")
        return None

def extract_complete_data(bag_path):
    print("🎬 Complete ROS Bag Data Extractor")
    
    bag_path = Path(bag_path)
    print(f"📁 Input bag: {bag_path}")
    
    # Find the .db3 file
    db_files = list(bag_path.glob('*.db3'))
    if not db_files:
        print("❌ No .db3 files found")
        return False
    
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
    
    # Connect to database
    print(f"🎒 Opening database: {db_file}")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Get topic IDs
    topics = {
        '/camera/image_raw': None,
        '/cmd_vel_manual': None,
        '/cmd_vel': None,
        '/joy': None
    }
    
    for topic_name in topics.keys():
        cursor.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
        result = cursor.fetchone()
        if result:
            topics[topic_name] = result[0]
            print(f"✅ Found topic: {topic_name} (ID: {result[0]})")
        else:
            print(f"⚠️  Topic not found: {topic_name}")
    
    # Extract Images
    images_metadata = []
    if topics['/camera/image_raw']:
        print(f"\n📸 Extracting images...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/camera/image_raw'],))
        image_messages = cursor.fetchall()
        
        print(f"📸 Processing {len(image_messages)} images...")
        
        # Extract using proven format
        IMAGE_DATA_OFFSET = 56
        IMAGE_SIZE = 921600
        WIDTH = 640
        HEIGHT = 480
        
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
                        file_size = filepath.stat().st_size
                        if file_size > 1000:
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
                                print(f"📸 Progress: {successful}/{i+1} images...")
                        else:
                            failed += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Error processing image {i}: {e}")
                failed += 1
        
        print(f"✅ Images: {successful} successful, {failed} failed")
    
    # Extract CMD_VEL_MANUAL data
    cmd_vel_manual_data = []
    if topics['/cmd_vel_manual']:
        print(f"\n🎮 Extracting manual velocity commands...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/cmd_vel_manual'],))
        cmd_messages = cursor.fetchall()
        
        print(f"🎮 Processing {len(cmd_messages)} manual commands...")
        
        for i, (timestamp, data) in enumerate(cmd_messages):
            twist_data = decode_twist_message(data)
            if twist_data:
                cmd_vel_manual_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'linear_x': twist_data['linear']['x'],
                    'linear_y': twist_data['linear']['y'],
                    'linear_z': twist_data['linear']['z'],
                    'angular_x': twist_data['angular']['x'],
                    'angular_y': twist_data['angular']['y'],
                    'angular_z': twist_data['angular']['z']
                })
        
        print(f"✅ Manual commands: {len(cmd_vel_manual_data)} decoded")
    
    # Extract CMD_VEL data
    cmd_vel_data = []
    if topics['/cmd_vel']:
        print(f"\n🚗 Extracting velocity commands...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/cmd_vel'],))
        cmd_messages = cursor.fetchall()
        
        print(f"🚗 Processing {len(cmd_messages)} commands...")
        
        for i, (timestamp, data) in enumerate(cmd_messages):
            twist_data = decode_twist_message(data)
            if twist_data:
                cmd_vel_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'linear_x': twist_data['linear']['x'],
                    'linear_y': twist_data['linear']['y'],
                    'linear_z': twist_data['linear']['z'],
                    'angular_x': twist_data['angular']['x'],
                    'angular_y': twist_data['angular']['y'],
                    'angular_z': twist_data['angular']['z']
                })
        
        print(f"✅ Commands: {len(cmd_vel_data)} decoded")
    
    # Extract JOY data
    joy_data = []
    if topics['/joy']:
        print(f"\n🕹️  Extracting joystick data...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/joy'],))
        joy_messages = cursor.fetchall()
        
        print(f"🕹️  Processing {len(joy_messages)} joystick messages...")
        
        for i, (timestamp, data) in enumerate(joy_messages):
            joy_msg = decode_joy_message(data)
            if joy_msg:
                joy_entry = {
                    'timestamp': timestamp,
                    'seq': i,
                }
                
                # Add axes data
                for j, axis_value in enumerate(joy_msg['axes']):
                    joy_entry[f'axis_{j}'] = axis_value
                
                # Add button data
                for j, button_value in enumerate(joy_msg['buttons']):
                    joy_entry[f'button_{j}'] = button_value
                
                joy_data.append(joy_entry)
        
        print(f"✅ Joystick: {len(joy_data)} decoded")
    
    conn.close()
    
    # Save all data
    print(f"\n💾 Saving extracted data...")
    
    # Save images metadata
    if images_metadata:
        csv_path = data_dir / "images.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=images_metadata[0].keys())
            writer.writeheader()
            writer.writerows(images_metadata)
        
        json_path = data_dir / "images.json"
        with open(json_path, 'w') as f:
            json.dump(images_metadata, f, indent=2)
        print(f"✅ Images metadata saved ({len(images_metadata)} entries)")
    
    # Save manual velocity commands
    if cmd_vel_manual_data:
        csv_path = data_dir / "cmd_vel_manual.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cmd_vel_manual_data[0].keys())
            writer.writeheader()
            writer.writerows(cmd_vel_manual_data)
        
        json_path = data_dir / "cmd_vel_manual.json"
        with open(json_path, 'w') as f:
            json.dump(cmd_vel_manual_data, f, indent=2)
        print(f"✅ Manual commands saved ({len(cmd_vel_manual_data)} entries)")
    
    # Save velocity commands
    if cmd_vel_data:
        csv_path = data_dir / "cmd_vel.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cmd_vel_data[0].keys())
            writer.writeheader()
            writer.writerows(cmd_vel_data)
        
        json_path = data_dir / "cmd_vel.json"
        with open(json_path, 'w') as f:
            json.dump(cmd_vel_data, f, indent=2)
        print(f"✅ Commands saved ({len(cmd_vel_data)} entries)")
    
    # Save joystick data
    if joy_data:
        csv_path = data_dir / "joy.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=joy_data[0].keys())
            writer.writeheader()
            writer.writerows(joy_data)
        
        json_path = data_dir / "joy.json"
        with open(json_path, 'w') as f:
            json.dump(joy_data, f, indent=2)
        print(f"✅ Joystick data saved ({len(joy_data)} entries)")
    
    # Create a synchronized dataset
    print(f"\n🔄 Creating synchronized dataset...")
    create_synchronized_dataset(data_dir, images_metadata, cmd_vel_manual_data, cmd_vel_data, joy_data)
    
    # Summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   📸 Images: {len(images_metadata)}")
    print(f"   🎮 Manual commands: {len(cmd_vel_manual_data)}")
    print(f"   🚗 Velocity commands: {len(cmd_vel_data)}")
    print(f"   🕹️  Joystick inputs: {len(joy_data)}")
    
    print(f"\n🎉 Complete dataset extracted to: {output_dir}")
    print(f"   📸 Images: {images_dir}/")
    print(f"   📊 Data files: {data_dir}/")
    
    return True

def create_synchronized_dataset(data_dir, images_metadata, cmd_vel_manual_data, cmd_vel_data, joy_data):
    """Create a synchronized dataset matching images with closest commands"""
    if not images_metadata:
        return
    
    print("🔄 Synchronizing data streams...")
    
    synchronized_data = []
    
    for img_meta in images_metadata:
        img_timestamp = img_meta['timestamp']
        
        sync_entry = {
            'image_filename': img_meta['filename'],
            'image_timestamp': img_timestamp,
            'image_seq': img_meta['seq']
        }
        
        # Find closest manual command
        if cmd_vel_manual_data:
            closest_manual = min(cmd_vel_manual_data, 
                                key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'manual_linear_x': closest_manual['linear_x'],
                'manual_linear_y': closest_manual['linear_y'],
                'manual_angular_z': closest_manual['angular_z'],
                'manual_timestamp': closest_manual['timestamp'],
                'manual_time_diff': abs(closest_manual['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest velocity command
        if cmd_vel_data:
            closest_cmd = min(cmd_vel_data, 
                             key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'cmd_linear_x': closest_cmd['linear_x'],
                'cmd_linear_y': closest_cmd['linear_y'],
                'cmd_angular_z': closest_cmd['angular_z'],
                'cmd_timestamp': closest_cmd['timestamp'],
                'cmd_time_diff': abs(closest_cmd['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest joystick input
        if joy_data:
            closest_joy = min(joy_data, 
                             key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'joy_timestamp': closest_joy['timestamp'],
                'joy_time_diff': abs(closest_joy['timestamp'] - img_timestamp) / 1e9
            })
            
            # Add available axes and buttons
            for key, value in closest_joy.items():
                if key.startswith('axis_') or key.startswith('button_'):
                    sync_entry[f'joy_{key}'] = value
        
        synchronized_data.append(sync_entry)
    
    # Save synchronized dataset
    if synchronized_data:
        csv_path = data_dir / "synchronized_dataset.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=synchronized_data[0].keys())
            writer.writeheader()
            writer.writerows(synchronized_data)
        
        json_path = data_dir / "synchronized_dataset.json"
        with open(json_path, 'w') as f:
            json.dump(synchronized_data, f, indent=2)
        
        print(f"✅ Synchronized dataset created ({len(synchronized_data)} entries)")
        print(f"   📄 Files: synchronized_dataset.csv, synchronized_dataset.json")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_complete_bag.py <bag_directory>")
        print("Example: python3 extract_complete_bag.py behavior_20250627_174455")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    success = extract_complete_data(bag_path)
    
    if success:
        print(f"\n🎯 SUCCESS! Complete behavior cloning dataset ready!")
        output_dir = Path(bag_path).parent / f"{Path(bag_path).name}_extracted"
        print(f"\n📁 Your dataset structure:")
        print(f"   {output_dir}/")
        print(f"   ├── images/              # Camera images")
        print(f"   └── data/                # All extracted data")
        print(f"       ├── images.csv       # Image metadata")
        print(f"       ├── cmd_vel_manual.csv  # Manual control commands")
        print(f"       ├── cmd_vel.csv      # Velocity commands")
        print(f"       ├── joy.csv          # Joystick inputs")
        print(f"       └── synchronized_dataset.csv  # Time-aligned data")
        print(f"\n💡 Use synchronized_dataset.csv for behavior cloning training!")
    else:
        print(f"\n❌ Extraction failed. Check the bag file and try again.")