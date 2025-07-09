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

def decode_float64_message(data):
    """Decode a std_msgs/msg/Float64 message from CDR format"""
    try:
        if len(data) < 12:  # 4 bytes header + 8 bytes float64
            return None
        
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Extract float64 value
        value = struct.unpack('<d', data[offset:offset+8])[0]
        
        return {'data': value}
    except Exception as e:
        print(f"Error decoding Float64 message: {e}")
        return None

def decode_int32_message(data):
    """Decode a std_msgs/msg/Int32 message from CDR format"""
    try:
        if len(data) < 8:  # 4 bytes header + 4 bytes int32
            return None
        
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Extract int32 value
        value = struct.unpack('<i', data[offset:offset+4])[0]
        
        return {'data': value}
    except Exception as e:
        print(f"Error decoding Int32 message: {e}")
        return None

def decode_string_message(data):
    """Decode a std_msgs/msg/String message from CDR format"""
    try:
        if len(data) < 8:  # Minimum size
            return None
        
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Read string length (4 bytes)
        string_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        if len(data) < offset + string_length:
            return None
        
        # Extract string data
        string_data = data[offset:offset+string_length].decode('utf-8', errors='ignore')
        
        return {'data': string_data.rstrip('\x00')}  # Remove null terminators
    except Exception as e:
        print(f"Error decoding String message: {e}")
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

def save_data_to_csv_and_json(data, fieldnames, data_dir, filename_base, data_type):
    """Save data to both CSV and JSON files with proper fieldname handling"""
    if not data:
        print(f"⚠️  No data to save for {data_type}")
        return
    
    csv_path = data_dir / f"{filename_base}.csv"
    json_path = data_dir / f"{filename_base}.json"
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Create a row with all fieldnames, filling missing values with None
            complete_row = {field: row.get(field, None) for field in fieldnames}
            writer.writerow(complete_row)
    
    # Write JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ {data_type} saved ({len(data)} entries)")

def diagnose_image_format(data):
    """Diagnose the actual image format in the data"""
    print(f"🔍 Diagnosing image format...")
    print(f"   Total data length: {len(data)} bytes")
    
    # Try different offsets and sizes
    potential_configs = [
        {"offset": 56, "size": 921600, "width": 640, "height": 480, "channels": 3},  # Original
        {"offset": 56, "size": 230400, "width": 320, "height": 240, "channels": 3},  # Current
        {"offset": 64, "size": 921600, "width": 640, "height": 480, "channels": 3},  # Different offset
        {"offset": 64, "size": 230400, "width": 320, "height": 240, "channels": 3},  # Different offset
        {"offset": 72, "size": 921600, "width": 640, "height": 480, "channels": 3},  # Different offset
        {"offset": 72, "size": 230400, "width": 320, "height": 240, "channels": 3},  # Different offset
    ]
    
    for config in potential_configs:
        offset = config["offset"]
        size = config["size"]
        width = config["width"]
        height = config["height"]
        channels = config["channels"]
        
        if len(data) >= offset + size:
            print(f"   ✅ Config possible: offset={offset}, size={size}, {width}x{height}x{channels}")
            
            # Try to extract and check if it looks like valid image data
            img_data = data[offset:offset + size]
            
            # Check for reasonable pixel values (0-255)
            sample_values = img_data[:100]  # Check first 100 bytes
            valid_pixels = sum(1 for b in sample_values if 0 <= b <= 255)
            print(f"      Valid pixel ratio: {valid_pixels/100:.2f}")
            
            # Check for variation (not all same value)
            unique_values = len(set(sample_values))
            print(f"      Unique values in sample: {unique_values}")
            
        else:
            print(f"   ❌ Config impossible: offset={offset}, size={size}, {width}x{height}x{channels} (not enough data)")
    
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
    
    # Get topic IDs - including new autonomous topics
    topics = {
        '/camera/image_raw': None,
        '/cmd_vel_manual': None,
        '/cmd_vel': None,
        '/joy': None,
        '/cmd_vel_autonomous': None,        # Main inference commands
        '/car/angular_prediction': None,    # Raw angular velocity predictions
        '/car/inference_confidence': None,  # Confidence scores
        '/car/queue_size': None,           # Prediction queue status
        '/car/inference_status': None,     # Inference node status
    }
    
    for topic_name in topics.keys():
        cursor.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
        result = cursor.fetchone()
        if result:
            topics[topic_name] = result[0]
            print(f"✅ Found topic: {topic_name} (ID: {result[0]})")
        else:
            print(f"⚠️  Topic not found: {topic_name}")
    
    # Extract Images with detailed debugging
    images_metadata = []
    if topics['/camera/image_raw']:
        print(f"\n📸 Extracting images...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/camera/image_raw'],))
        image_messages = cursor.fetchall()
        
        print(f"📸 Found {len(image_messages)} image messages...")
        
        if len(image_messages) > 0:
            # Diagnose the first image to understand format
            print(f"🔍 Analyzing first image message...")
            first_timestamp, first_data = image_messages[0]
            diagnose_image_format(first_data)
        
        # Try multiple configurations
        configs_to_try = [
            {"offset": 56, "size": 230400, "width": 320, "height": 240},  # Current config
            {"offset": 56, "size": 921600, "width": 640, "height": 480},  # Original config
            {"offset": 64, "size": 230400, "width": 320, "height": 240},  # Different offset
            {"offset": 72, "size": 230400, "width": 320, "height": 240},  # Different offset
        ]
        
        best_config = None
        best_success_rate = 0
        
        for config in configs_to_try:
            print(f"\n🧪 Testing config: offset={config['offset']}, {config['width']}x{config['height']}")
            
            successful = 0
            failed = 0
            
            # Test first 10 images with this config
            test_messages = image_messages[:min(10, len(image_messages))]
            
            for i, (timestamp, data) in enumerate(test_messages):
                try:
                    if len(data) >= config['offset'] + config['size']:
                        # Extract image data
                        img_data = data[config['offset']:config['offset'] + config['size']]
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img = img_array.reshape((config['height'], config['width'], 3))
                        
                        # Create test filename
                        test_filename = f"test_image_{i:06d}.png"
                        test_filepath = images_dir / test_filename
                        
                        # Write image with error checking
                        write_success = cv2.imwrite(str(test_filepath), img)
                        
                        if write_success and test_filepath.exists():
                            file_size = test_filepath.stat().st_size
                            if file_size > 1000:
                                successful += 1
                                # Clean up test file
                                test_filepath.unlink()
                            else:
                                failed += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    failed += 1
            
            success_rate = successful / len(test_messages) if test_messages else 0
            print(f"   Success rate: {success_rate:.2f} ({successful}/{len(test_messages)})")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_config = config
        
        if best_config and best_success_rate > 0:
            print(f"\n✅ Using best config: offset={best_config['offset']}, {best_config['width']}x{best_config['height']}")
            
            # Extract all images with best config
            successful = 0
            failed = 0
            
            for i, (timestamp, data) in enumerate(image_messages):
                try:
                    if len(data) >= best_config['offset'] + best_config['size']:
                        # Extract image data
                        img_data = data[best_config['offset']:best_config['offset'] + best_config['size']]
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img = img_array.reshape((best_config['height'], best_config['width'], 3))
                        
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
                                    'width': best_config['width'],
                                    'height': best_config['height'],
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
        else:
            print(f"❌ No working image configuration found!")
    else:
        print(f"⚠️  No camera topic found - skipping image extraction")
    
    # Debug: Print images_metadata status
    print(f"\n🔍 Images metadata status: {len(images_metadata)} entries")
    if len(images_metadata) == 0:
        print("❌ WARNING: No images extracted! This will cause empty synchronized dataset.")
    
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
    
    # Extract CMD_VEL_AUTONOMOUS data
    cmd_vel_autonomous_data = []
    if topics['/cmd_vel_autonomous']:
        print(f"\n🤖 Extracting autonomous velocity commands...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/cmd_vel_autonomous'],))
        autonomous_messages = cursor.fetchall()
        
        print(f"🤖 Processing {len(autonomous_messages)} autonomous commands...")
        
        for i, (timestamp, data) in enumerate(autonomous_messages):
            twist_data = decode_twist_message(data)
            if twist_data:
                cmd_vel_autonomous_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'linear_x': twist_data['linear']['x'],
                    'linear_y': twist_data['linear']['y'],
                    'linear_z': twist_data['linear']['z'],
                    'angular_x': twist_data['angular']['x'],
                    'angular_y': twist_data['angular']['y'],
                    'angular_z': twist_data['angular']['z']
                })
        
        print(f"✅ Autonomous commands: {len(cmd_vel_autonomous_data)} decoded")
    
    # Extract ANGULAR_PREDICTION data
    angular_prediction_data = []
    if topics['/car/angular_prediction']:
        print(f"\n🔮 Extracting angular predictions...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/car/angular_prediction'],))
        prediction_messages = cursor.fetchall()
        
        print(f"🔮 Processing {len(prediction_messages)} predictions...")
        
        for i, (timestamp, data) in enumerate(prediction_messages):
            prediction_data = decode_float64_message(data)
            if prediction_data:
                angular_prediction_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'angular_prediction': prediction_data['data']
                })
        
        print(f"✅ Angular predictions: {len(angular_prediction_data)} decoded")
    
    # Extract INFERENCE_CONFIDENCE data
    inference_confidence_data = []
    if topics['/car/inference_confidence']:
        print(f"\n📊 Extracting inference confidence...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/car/inference_confidence'],))
        confidence_messages = cursor.fetchall()
        
        print(f"📊 Processing {len(confidence_messages)} confidence scores...")
        
        for i, (timestamp, data) in enumerate(confidence_messages):
            confidence_data = decode_float64_message(data)
            if confidence_data:
                inference_confidence_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'confidence': confidence_data['data']
                })
        
        print(f"✅ Confidence scores: {len(inference_confidence_data)} decoded")
    
    # Extract QUEUE_SIZE data
    queue_size_data = []
    if topics['/car/queue_size']:
        print(f"\n📦 Extracting queue size data...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/car/queue_size'],))
        queue_messages = cursor.fetchall()
        
        print(f"📦 Processing {len(queue_messages)} queue size messages...")
        
        for i, (timestamp, data) in enumerate(queue_messages):
            queue_data = decode_int32_message(data)
            if queue_data:
                queue_size_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'queue_size': queue_data['data']
                })
        
        print(f"✅ Queue sizes: {len(queue_size_data)} decoded")
    
    # Extract INFERENCE_STATUS data
    inference_status_data = []
    if topics['/car/inference_status']:
        print(f"\n📡 Extracting inference status...")
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                      (topics['/car/inference_status'],))
        status_messages = cursor.fetchall()
        
        print(f"📡 Processing {len(status_messages)} status messages...")
        
        for i, (timestamp, data) in enumerate(status_messages):
            status_data = decode_string_message(data)
            if status_data:
                inference_status_data.append({
                    'timestamp': timestamp,
                    'seq': i,
                    'status': status_data['data']
                })
        
        print(f"✅ Status messages: {len(inference_status_data)} decoded")
    
    # Extract JOY data with improved fieldname handling
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
        save_data_to_csv_and_json(images_metadata, images_metadata[0].keys(), 
                                data_dir, "images", "Images metadata")
    
    # Save manual velocity commands
    if cmd_vel_manual_data:
        save_data_to_csv_and_json(cmd_vel_manual_data, cmd_vel_manual_data[0].keys(), 
                                data_dir, "cmd_vel_manual", "Manual commands")
    
    # Save velocity commands
    if cmd_vel_data:
        save_data_to_csv_and_json(cmd_vel_data, cmd_vel_data[0].keys(), 
                                data_dir, "cmd_vel", "Commands")
    
    # Save autonomous velocity commands
    if cmd_vel_autonomous_data:
        save_data_to_csv_and_json(cmd_vel_autonomous_data, cmd_vel_autonomous_data[0].keys(), 
                                data_dir, "cmd_vel_autonomous", "Autonomous commands")
    
    # Save angular predictions
    if angular_prediction_data:
        save_data_to_csv_and_json(angular_prediction_data, angular_prediction_data[0].keys(), 
                                data_dir, "angular_prediction", "Angular predictions")
    
    # Save inference confidence
    if inference_confidence_data:
        save_data_to_csv_and_json(inference_confidence_data, inference_confidence_data[0].keys(), 
                                data_dir, "inference_confidence", "Inference confidence")
    
    # Save queue size data
    if queue_size_data:
        save_data_to_csv_and_json(queue_size_data, queue_size_data[0].keys(), 
                                data_dir, "queue_size", "Queue size data")
    
    # Save inference status
    if inference_status_data:
        save_data_to_csv_and_json(inference_status_data, inference_status_data[0].keys(), 
                                data_dir, "inference_status", "Inference status")
    
    # Save joystick data with all possible fieldnames
    if joy_data:
        # Collect ALL possible fieldnames from ALL joy entries
        all_joy_fieldnames = set()
        for entry in joy_data:
            all_joy_fieldnames.update(entry.keys())
        
        # Sort fieldnames for consistent ordering
        sorted_fieldnames = sorted(all_joy_fieldnames)
        
        save_data_to_csv_and_json(joy_data, sorted_fieldnames, 
                                data_dir, "joy", "Joystick data")
    
    # Create a synchronized dataset
    print(f"\n🔄 Creating synchronized dataset...")
    create_synchronized_dataset(data_dir, images_metadata, cmd_vel_manual_data, cmd_vel_data, 
                              cmd_vel_autonomous_data, angular_prediction_data, 
                              inference_confidence_data, queue_size_data, inference_status_data, joy_data)
    
    # Summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   📸 Images: {len(images_metadata)}")
    print(f"   🎮 Manual commands: {len(cmd_vel_manual_data)}")
    print(f"   🚗 Velocity commands: {len(cmd_vel_data)}")
    print(f"   🤖 Autonomous commands: {len(cmd_vel_autonomous_data)}")
    print(f"   🔮 Angular predictions: {len(angular_prediction_data)}")
    print(f"   📊 Confidence scores: {len(inference_confidence_data)}")
    print(f"   📦 Queue sizes: {len(queue_size_data)}")
    print(f"   📡 Status messages: {len(inference_status_data)}")
    print(f"   🕹️  Joystick inputs: {len(joy_data)}")
    
    print(f"\n🎉 Complete dataset extracted to: {output_dir}")
    print(f"   📸 Images: {images_dir}/")
    print(f"   📊 Data files: {data_dir}/")
    
    return True

def create_synchronized_dataset(data_dir, images_metadata, cmd_vel_manual_data, cmd_vel_data, 
                              cmd_vel_autonomous_data, angular_prediction_data, 
                              inference_confidence_data, queue_size_data, inference_status_data, joy_data):
    """Create a synchronized dataset matching images with closest commands and inference data"""
    
    print("🔄 Synchronizing data streams...")
    print(f"   📸 Images available: {len(images_metadata)}")
    print(f"   🎮 Manual commands available: {len(cmd_vel_manual_data)}")
    print(f"   🚗 Velocity commands available: {len(cmd_vel_data)}")
    print(f"   🤖 Autonomous commands available: {len(cmd_vel_autonomous_data)}")
    print(f"   🔮 Angular predictions available: {len(angular_prediction_data)}")
    print(f"   📊 Confidence scores available: {len(inference_confidence_data)}")
    print(f"   📦 Queue sizes available: {len(queue_size_data)}")
    print(f"   📡 Status messages available: {len(inference_status_data)}")
    print(f"   🕹️  Joystick inputs available: {len(joy_data)}")
    
    if not images_metadata:
        print("❌ WARNING: No images available - cannot create synchronized dataset!")
        print("   The synchronized dataset requires images as the primary time reference.")
        print("   Please check why image extraction failed.")
        return
    
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
        
        # Find closest autonomous command
        if cmd_vel_autonomous_data:
            closest_auto = min(cmd_vel_autonomous_data, 
                              key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'auto_linear_x': closest_auto['linear_x'],
                'auto_linear_y': closest_auto['linear_y'],
                'auto_angular_z': closest_auto['angular_z'],
                'auto_timestamp': closest_auto['timestamp'],
                'auto_time_diff': abs(closest_auto['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest angular prediction
        if angular_prediction_data:
            closest_prediction = min(angular_prediction_data, 
                                   key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'angular_prediction': closest_prediction['angular_prediction'],
                'prediction_timestamp': closest_prediction['timestamp'],
                'prediction_time_diff': abs(closest_prediction['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest confidence score
        if inference_confidence_data:
            closest_confidence = min(inference_confidence_data, 
                                   key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'inference_confidence': closest_confidence['confidence'],
                'confidence_timestamp': closest_confidence['timestamp'],
                'confidence_time_diff': abs(closest_confidence['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest queue size
        if queue_size_data:
            closest_queue = min(queue_size_data, 
                               key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'queue_size': closest_queue['queue_size'],
                'queue_timestamp': closest_queue['timestamp'],
                'queue_time_diff': abs(closest_queue['timestamp'] - img_timestamp) / 1e9
            })
        
        # Find closest inference status
        if inference_status_data:
            closest_status = min(inference_status_data, 
                                key=lambda x: abs(x['timestamp'] - img_timestamp))
            sync_entry.update({
                'inference_status': closest_status['status'],
                'status_timestamp': closest_status['timestamp'],
                'status_time_diff': abs(closest_status['timestamp'] - img_timestamp) / 1e9
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
    
    print(f"   ✅ Created {len(synchronized_data)} synchronized entries")
    
    # Save synchronized dataset with all possible fieldnames
    if synchronized_data:
        # Collect all possible fieldnames
        all_sync_fieldnames = set()
        for entry in synchronized_data:
            all_sync_fieldnames.update(entry.keys())
        
        sorted_sync_fieldnames = sorted(all_sync_fieldnames)
        print(f"   📋 Synchronized dataset will have {len(sorted_sync_fieldnames)} columns:")
        for fieldname in sorted_sync_fieldnames:
            print(f"      - {fieldname}")
        
        save_data_to_csv_and_json(synchronized_data, sorted_sync_fieldnames, 
                                data_dir, "synchronized_dataset", "Synchronized dataset")
        
        print(f"   📄 Files: synchronized_dataset.csv, synchronized_dataset.json")
    else:
        print("❌ No synchronized data created!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_bc_data.py <bag_directory>")
        print("Example: python3 extract_bc_data.py behavior_20250627_174455")
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
        print(f"       ├── cmd_vel_manual.csv     # Manual control commands")
        print(f"       ├── cmd_vel.csv            # Velocity commands")
        print(f"       ├── cmd_vel_autonomous.csv # Autonomous commands")
        print(f"       ├── angular_prediction.csv # Raw predictions")
        print(f"       ├── inference_confidence.csv # Confidence scores")
        print(f"       ├── queue_size.csv         # Queue status")
        print(f"       ├── inference_status.csv   # Node status")
        print(f"       ├── joy.csv                # Joystick inputs")
        print(f"       └── synchronized_dataset.csv # Time-aligned data")
        print(f"\n💡 Use synchronized_dataset.csv for behavior cloning training!")
        print(f"🔍 Analyze inference performance with confidence and prediction data!")
    else:
        print(f"\n❌ Extraction failed. Check the bag file and try again.")
