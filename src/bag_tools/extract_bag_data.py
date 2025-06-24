#!/usr/bin/env python3
"""
Integrated ROS2 Bag Data Extractor

Combines SQLite direct access for commands with ROS2 tools for images.
This approach is reliable and works without complex CDR parsing.
"""

import os
import sys
import json
import csv
import sqlite3
import struct
import subprocess
import signal
import time
import threading
from pathlib import Path
from datetime import datetime
import argparse


class IntegratedBagExtractor:
    """Extract data from ROS2 bag files using hybrid approach."""
    
    def __init__(self, bag_path, output_path):
        self.bag_path = Path(bag_path)
        self.output_path = Path(output_path)
        
        # Find the .db3 file
        self.db_file = None
        self.original_bag_path = self.bag_path
        
        if self.bag_path.is_file() and self.bag_path.suffix == '.db3':
            self.db_file = self.bag_path
        elif self.bag_path.is_file() and self.bag_path.suffix == '.zstd':
            # Handle compressed files
            self.db_file = self.bag_path
        else:
            # Look for .db3 files in directory
            db_files = list(self.bag_path.glob('*.db3*'))
            if db_files:
                self.db_file = db_files[0]
        
        if not self.db_file:
            raise ValueError(f"No .db3 file found in {bag_path}")
        
        # Handle compressed files
        if self.db_file.suffix == '.zstd':
            self.decompress_file()
        
        # Create output directories
        self.setup_output_directories()
        
        # Data storage
        self.images_data = []
        self.cmd_vel_data = []
        self.cmd_vel_manual_data = []
        self.joy_data = []
        
        # Tracking
        self.image_count = 0
        self.bag_duration = 0
        self.has_images = False
        
    def decompress_file(self):
        """Decompress .zstd file if needed."""
        try:
            import zstandard as zstd
            
            decompressed_path = self.db_file.with_suffix('')  # Remove .zstd
            
            print(f"🗜️  Decompressing {self.db_file.name}...")
            
            with open(self.db_file, 'rb') as compressed_file:
                dctx = zstd.ZstdDecompressor()
                with open(decompressed_path, 'wb') as decompressed_file:
                    dctx.copy_stream(compressed_file, decompressed_file)
            
            self.db_file = decompressed_path
            print(f"✅ Decompressed to {self.db_file.name}")
            
        except ImportError:
            print("❌ zstandard library not found. Install with: pip install zstandard")
            print("    Or decompress manually with: zstd -d filename.db3.zstd")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            sys.exit(1)
    
    def setup_output_directories(self):
        """Create organized output directory structure."""
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "images").mkdir(exist_ok=True)
        (self.output_path / "data").mkdir(exist_ok=True)
        (self.output_path / "metadata").mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_path}")
        print(f"📁 Images will be saved to: {self.output_path}/images/")
        print(f"📁 Data files will be saved to: {self.output_path}/data/")
    
    def extract_data(self):
        """Extract all data from the bag file."""
        print(f"🎒 Starting extraction from: {self.original_bag_path}")
        
        # First, extract command data from SQLite
        success = self.extract_command_data()
        if not success:
            print("❌ Failed to extract command data")
            return False
        
        # Then, extract images using ROS2 tools
        if self.has_images:
            print("\n📸 Extracting images using ROS2 tools...")
            success = self.extract_images_ros2()
            if not success:
                print("⚠️  Image extraction failed, but continuing with command data")
        else:
            print("📸 No image topic found, skipping image extraction")
        
        # Save all data to files
        self.save_data_files()
        self.create_summary()
        
        return True
    
    def extract_command_data(self):
        """Extract command and joystick data using SQLite."""
        print(f"🎒 Opening database: {self.db_file}")
        
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            # Get topics
            cursor.execute("SELECT id, name, type FROM topics")
            topics = {topic_id: {'name': name, 'type': topic_type} 
                     for topic_id, name, topic_type in cursor.fetchall()}
            
            print(f"📋 Found topics: {[t['name'] for t in topics.values()]}")
            
            # Check if we have image topic
            self.has_images = any(t['name'] == '/camera/image_raw' for t in topics.values())
            
            # Get all messages for command topics only
            command_topics = [tid for tid, info in topics.items() 
                            if info['name'] in ['/cmd_vel', '/cmd_vel_manual', '/joy']]
            
            if command_topics:
                placeholders = ','.join(['?'] * len(command_topics))
                cursor.execute(f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({placeholders}) ORDER BY timestamp", 
                             command_topics)
                messages = cursor.fetchall()
                
                print(f"📊 Processing {len(messages)} command messages...")
                
                # Process each message
                for i, (topic_id, timestamp, data) in enumerate(messages):
                    if i % 100 == 0 and i > 0:
                        print(f"📊 Processed {i}/{len(messages)} messages...")
                    
                    topic_info = topics[topic_id]
                    topic_name = topic_info['name']
                    
                    try:
                        # Process based on topic
                        if topic_name == '/cmd_vel':
                            self.process_twist_data(data, timestamp, 'cmd_vel')
                        elif topic_name == '/cmd_vel_manual':
                            self.process_twist_data(data, timestamp, 'cmd_vel_manual')
                        elif topic_name == '/joy':
                            self.process_joy_data_simple(data, timestamp)
                    
                    except Exception as e:
                        # Silent fail for individual messages
                        continue
            
            # Calculate bag duration for image extraction
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
            time_range = cursor.fetchone()
            if time_range[0] and time_range[1]:
                self.bag_duration = (time_range[1] - time_range[0]) / 1e9  # Convert to seconds
                
            conn.close()
            print(f"✅ Successfully extracted command data")
            print(f"   🚗 Motor commands: {len(self.cmd_vel_data)}")
            print(f"   🎮 Manual commands: {len(self.cmd_vel_manual_data)}")
            print(f"   🕹️  Joystick records: {len(self.joy_data)}")
            
        except Exception as e:
            print(f"❌ Error reading database: {e}")
            return False
        
        return True
    
    def process_twist_data(self, data, timestamp, data_type):
        """Process Twist message data with simple CDR parsing."""
        try:
            # Simple CDR parsing for geometry_msgs/Twist
            # Skip CDR header (4 bytes) and read 6 doubles (48 bytes)
            if len(data) < 52:  # 4 + 48
                return
                
            # Read 6 doubles (linear.x, linear.y, linear.z, angular.x, angular.y, angular.z)
            values = struct.unpack('<6d', data[4:52])
            
            twist_data = {
                'timestamp': timestamp,
                'linear_x': values[0],
                'linear_y': values[1], 
                'linear_z': values[2],
                'angular_x': values[3],
                'angular_y': values[4],
                'angular_z': values[5]
            }
            
            if data_type == 'cmd_vel':
                self.cmd_vel_data.append(twist_data)
            elif data_type == 'cmd_vel_manual':
                self.cmd_vel_manual_data.append(twist_data)
                
        except Exception:
            pass  # Silent fail
    
    def process_joy_data_simple(self, data, timestamp):
        """Process Joy message data with simple approach."""
        try:
            # Very simple approach - just extract some basic joystick info
            # This is a fallback that might not get all data but won't crash
            joy_data = {
                'timestamp': timestamp,
                'axes': [0.0, 0.0, 0.0, 0.0],  # Default values
                'buttons': [0, 0, 0, 0, 0, 0, 0, 0],  # Default values
                'header_frame_id': ''
            }
            
            # Try to extract some floats from the data (axes)
            if len(data) > 20:
                try:
                    # Look for float values in the data
                    for i in range(4, len(data) - 8, 4):
                        val = struct.unpack('<f', data[i:i+4])[0]
                        if -2.0 <= val <= 2.0:  # Reasonable joystick range
                            if len(joy_data['axes']) < 4:
                                joy_data['axes'][len(joy_data['axes'])] = val
                except:
                    pass
            
            self.joy_data.append(joy_data)
            
        except Exception:
            pass  # Silent fail
    
    def extract_images_ros2(self):
        """Extract images using proven manual method first, then ROS2 as fallback."""
        try:
            print(f"📸 Starting image extraction...")
            print(f"   📏 Expected duration: {self.bag_duration:.1f} seconds")
            
            # Try the proven manual method first
            print("🎯 Using proven extraction method (offset 56, BGR format)...")
            success = self.extract_images_manual_enhanced()
            if success:
                print(f"✅ Proven method extracted {self.image_count} images successfully!")
                return True
            
            # If proven method fails, try ROS2 approach as fallback
            print("🔄 Proven method failed, trying ROS2 tools as fallback...")
            return self.extract_images_ros2_fallback()
            
        except Exception as e:
            print(f"❌ Image extraction failed: {e}")
            return False
    
    def extract_images_ros2_fallback(self):
        """Fallback ROS2 image extraction with proper Foxy syntax."""
        try:
            print(f"🔄 Fallback: ROS2 tools extraction...")
            
            # Create temporary images directory
            temp_images_dir = self.original_bag_path.parent / "temp_extracted_images"
            temp_images_dir.mkdir(exist_ok=True)
            
            print(f"📁 Temporary images directory: {temp_images_dir}")
            
            # Try ROS2 approach with correct Foxy syntax
            print("🔄 Using ROS2 tools with correct syntax...")
            
            # First start image extraction tool (before bag play)
            extract_cmd = [
                'ros2', 'run', 'image_view', 'extract_images',
                '--ros-args', '--remap', 'image:=/camera/image_raw',
                '-p', f'filename_format:={temp_images_dir}/image_%06d.png'
            ]
            
            print(f"📸 Starting image extraction: {' '.join(extract_cmd)}")
            extract_process = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Give extract_images time to start up
            time.sleep(3)
            
            # Then start bag playback
            bag_cmd = [
                'ros2', 'bag', 'play', str(self.original_bag_path),
                '--topics', '/camera/image_raw',
                '--rate', '1.0'
            ]
            
            print(f"🎬 Starting bag playback: {' '.join(bag_cmd)}")
            bag_process = subprocess.Popen(
                bag_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.original_bag_path.parent)
            )
            
            # Monitor progress
            wait_time = max(self.bag_duration + 10, 20)  # Give extra time
            print(f"⏳ Monitoring extraction for {wait_time:.1f} seconds...")
            
            for i in range(int(wait_time)):
                time.sleep(1)
                
                # Check if bag finished
                if bag_process.poll() is not None:
                    print(f"🏁 Bag playback finished at {i}s")
                    time.sleep(2)  # Give extract_images time to process
                    break
                    
                # Check progress
                current_files = len(list(temp_images_dir.glob('*.png')))
                if i % 5 == 0 and current_files > 0:
                    print(f"📸 Progress: {current_files} images extracted...")
                elif i % 10 == 0:
                    print(f"⏳ Still waiting... ({i}s)")
            
            # Stop processes gracefully
            print("🛑 Stopping processes...")
            try:
                if bag_process.poll() is None:
                    bag_process.terminate()
                    time.sleep(2)
                    if bag_process.poll() is None:
                        bag_process.kill()
                        
                if extract_process.poll() is None:
                    extract_process.terminate()
                    time.sleep(2)
                    if extract_process.poll() is None:
                        extract_process.kill()
            except:
                pass
            
            # Check results
            extracted_files = list(temp_images_dir.glob('*.png'))
            print(f"📸 Found {len(extracted_files)} extracted images")
            
            if extracted_files:
                # Process extracted images
                return self.process_extracted_images(extracted_files, temp_images_dir)
            else:
                print("❌ ROS2 fallback method also failed")
                return False
                
        except Exception as e:
            print(f"❌ ROS2 fallback failed: {e}")
            return False
    
    def try_alternative_extraction(self):
        """Try alternative image extraction methods."""
        print("🔄 Trying alternative extraction methods...")
        
        # Method 1: Simple ros2 bag play to file
        try:
            print("📝 Method 1: Play bag and save to video...")
            temp_dir = self.original_bag_path.parent / "temp_video"
            temp_dir.mkdir(exist_ok=True)
            
            # Try to record video using gstreamer
            record_cmd = [
                'ros2', 'bag', 'play', str(self.original_bag_path),
                '--topics', '/camera/image_raw'
            ]
            
            # Start bag play in background and try to capture
            # This is a simpler approach - just get the first few frames
            
            print("🎬 Playing bag for video capture...")
            process = subprocess.Popen(record_cmd, cwd=str(self.original_bag_path.parent))
            
            # Let it run briefly
            time.sleep(min(self.bag_duration + 2, 10))
            
            # Stop it
            process.terminate()
            try:
                process.wait(timeout=3)
            except:
                process.kill()
            
        except Exception as e:
            print(f"📝 Video method failed: {e}")
        
        # Method 2: Enhanced manual extraction
        print("🔧 Method 2: Enhanced manual extraction...")
        return self.extract_images_manual_enhanced()
    
    def extract_images_manual_enhanced(self):
        """Enhanced manual image extraction with the proven format - FIXED VERSION."""
        try:
            print("🔧 Starting proven manual extraction method...")
            
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            # Get image topic ID
            cursor.execute("SELECT id FROM topics WHERE name = '/camera/image_raw'")
            result = cursor.fetchone()
            if not result:
                print("❌ No camera topic found")
                return False
                
            image_topic_id = result[0]
            
            # Get all image messages
            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", 
                         (image_topic_id,))
            image_messages = cursor.fetchall()
            
            print(f"📸 Processing {len(image_messages)} image messages with proven format...")
            
            if len(image_messages) == 0:
                return False
            
            # Use the proven format: offset 56, size 921600, 640x480x3 BGR
            IMAGE_DATA_OFFSET = 56
            IMAGE_SIZE = 921600  # 640 * 480 * 3
            WIDTH = 640
            HEIGHT = 480
            
            extracted_count = 0
            failed_count = 0
            
            # Ensure output directory exists with proper permissions
            self.output_path.mkdir(exist_ok=True, mode=0o755)
            (self.output_path / "images").mkdir(exist_ok=True, mode=0o755)
            
            for i, (timestamp, data) in enumerate(image_messages):
                if i % 50 == 0:
                    print(f"📸 Processing {i+1}/{len(image_messages)} images...")
                
                try:
                    # Check if we have enough data
                    if len(data) >= IMAGE_DATA_OFFSET + IMAGE_SIZE:
                        # Extract image data using proven format
                        img_data = data[IMAGE_DATA_OFFSET:IMAGE_DATA_OFFSET + IMAGE_SIZE]
                        
                        # Reshape to BGR image
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img = img_array.reshape((HEIGHT, WIDTH, 3))
                        
                        # Save image with absolute path
                        filename = f"image_{extracted_count:06d}.png"
                        filepath = self.output_path / "images" / filename
                        
                        # Write image with verification
                        write_success = cv2.imwrite(str(filepath), img)
                        
                        if write_success and filepath.exists():
                            # Verify file was actually created and has reasonable size
                            file_size = filepath.stat().st_size
                            if file_size > 1000:  # At least 1KB
                                # Add to metadata
                                self.images_data.append({
                                    'filename': filename,
                                    'timestamp': timestamp,
                                    'width': WIDTH,
                                    'height': HEIGHT,
                                    'encoding': 'bgr8',
                                    'frame_id': 'camera_link',
                                    'seq': extracted_count,
                                    'file_size': file_size
                                })
                                extracted_count += 1
                            else:
                                print(f"⚠️  Image {i} file too small ({file_size} bytes)")
                                failed_count += 1
                                # Remove the bad file
                                try:
                                    filepath.unlink()
                                except:
                                    pass
                        else:
                            print(f"❌ Failed to write image {i}")
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"❌ Error processing image {i}: {e}")
                    failed_count += 1
                    continue
            
            conn.close()
            
            if extracted_count > 0:
                print(f"✅ Proven method successful: {extracted_count} images extracted")
                if failed_count > 0:
                    print(f"⚠️  {failed_count} images failed to extract")
                
                # Set image count for summary
                self.image_count = extracted_count
                
                # Test a few images to verify quality
                print(f"🔍 Verifying extracted images...")
                for i in range(min(3, extracted_count)):
                    test_path = self.output_path / "images" / f"image_{i:06d}.png"
                    if test_path.exists():
                        test_img = cv2.imread(str(test_path))
                        if test_img is not None:
                            print(f"✅ image_{i:06d}.png: {test_img.shape}, range {test_img.min()}-{test_img.max()}")
                        else:
                            print(f"❌ image_{i:06d}.png: Could not read back")
                
                return True
            else:
                print("❌ Proven method failed - no images extracted")
                return False
                
        except Exception as e:
            print(f"❌ Proven extraction error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_image_format(self, data):
        """Analyze CDR data to determine image format."""
        try:
            if len(data) < 100:
                return None
            
            # Skip CDR header (4 bytes) and parse sensor_msgs/Image header
            pos = 4
            
            # Skip header timestamp and frame_id (rough estimate)
            pos += 12  # timestamp
            
            # Try to read string length for frame_id
            if pos + 4 < len(data):
                frame_id_len = struct.unpack('<I', data[pos:pos+4])[0]
                pos += 4 + ((frame_id_len + 3) // 4) * 4  # Align to 4 bytes
            
            # Read image dimensions
            if pos + 8 < len(data):
                height = struct.unpack('<I', data[pos:pos+4])[0]
                width = struct.unpack('<I', data[pos+4:pos+8])[0]
                pos += 8
                
                # Skip encoding string
                if pos + 4 < len(data):
                    encoding_len = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4 + ((encoding_len + 3) // 4) * 4
                
                # Skip is_bigendian and step
                pos += 8
                
                # Get data array size
                if pos + 4 < len(data):
                    data_len = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4
                    
                    # Determine channels based on data size
                    expected_size_rgb = height * width * 3
                    expected_size_bgr = height * width * 3
                    expected_size_mono = height * width
                    
                    if data_len == expected_size_rgb or data_len == expected_size_bgr:
                        return {
                            'width': width,
                            'height': height,
                            'channels': 3,
                            'encoding': 'bgr8',
                            'data_offset': pos,
                            'data_size': data_len
                        }
                    elif data_len == expected_size_mono:
                        return {
                            'width': width,
                            'height': height,
                            'channels': 1,
                            'encoding': 'mono8',
                            'data_offset': pos,
                            'data_size': data_len
                        }
            
            return None
            
        except Exception as e:
            print(f"🔍 Format analysis error: {e}")
            return None
    
    def extract_single_image(self, data, config):
        """Extract a single image using the detected format."""
        try:
            offset = config['data_offset']
            size = config['data_size']
            
            if len(data) < offset + size:
                return None
            
            # Extract image data
            img_data = data[offset:offset + size]
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            
            # Reshape to image
            if config['channels'] == 3:
                img = img_array.reshape((config['height'], config['width'], 3))
                # Convert RGB to BGR if needed (OpenCV uses BGR)
                if config['encoding'] == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = img_array.reshape((config['height'], config['width']))
            
            return img
            
        except Exception as e:
            return None
    
    def process_extracted_images(self, extracted_files, temp_dir):
        """Process successfully extracted images from ROS2 tools."""
        try:
            print(f"📸 Processing {len(extracted_files)} extracted images...")
            
            for i, img_file in enumerate(sorted(extracted_files)):
                final_name = f"image_{i:06d}.png"
                final_path = self.output_path / "images" / final_name
                
                # Move file
                img_file.rename(final_path)
                
                # Get image info
                img = cv2.imread(str(final_path))
                height, width = img.shape[:2] if img is not None else (480, 640)
                
                # Add to metadata
                self.images_data.append({
                    'filename': final_name,
                    'timestamp': 0,
                    'width': width,
                    'height': height,
                    'encoding': 'bgr8',
                    'frame_id': 'camera',
                    'seq': i
                })
            
            self.image_count = len(extracted_files)
            print(f"✅ Successfully processed {self.image_count} images")
            
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing extracted images: {e}")
            return False
    
    def save_data_files(self):
        """Save all extracted data to CSV and JSON files."""
        print("💾 Saving data files...")
        
        # Save images metadata
        if self.images_data:
            with open(self.output_path / "data" / "images.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.images_data[0].keys())
                writer.writeheader()
                writer.writerows(self.images_data)
            
            with open(self.output_path / "data" / "images.json", 'w') as f:
                json.dump(self.images_data, f, indent=2)
            
            print(f"📸 Saved {len(self.images_data)} image records")
        
        # Save cmd_vel data
        if self.cmd_vel_data:
            with open(self.output_path / "data" / "cmd_vel.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.cmd_vel_data[0].keys())
                writer.writeheader()
                writer.writerows(self.cmd_vel_data)
            
            with open(self.output_path / "data" / "cmd_vel.json", 'w') as f:
                json.dump(self.cmd_vel_data, f, indent=2)
            
            print(f"🚗 Saved {len(self.cmd_vel_data)} cmd_vel records")
        
        # Save cmd_vel_manual data
        if self.cmd_vel_manual_data:
            with open(self.output_path / "data" / "cmd_vel_manual.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.cmd_vel_manual_data[0].keys())
                writer.writeheader()
                writer.writerows(self.cmd_vel_manual_data)
            
            with open(self.output_path / "data" / "cmd_vel_manual.json", 'w') as f:
                json.dump(self.cmd_vel_manual_data, f, indent=2)
            
            print(f"🎮 Saved {len(self.cmd_vel_manual_data)} cmd_vel_manual records")
        
        # Save joy data
        if self.joy_data:
            with open(self.output_path / "data" / "joy.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'axes', 'buttons', 'header_frame_id'])
                writer.writeheader()
                for row in self.joy_data:
                    csv_row = row.copy()
                    csv_row['axes'] = str(row['axes'])
                    csv_row['buttons'] = str(row['buttons'])
                    writer.writerow(csv_row)
            
            with open(self.output_path / "data" / "joy.json", 'w') as f:
                json.dump(self.joy_data, f, indent=2)
            
            print(f"🕹️  Saved {len(self.joy_data)} joystick records")
    
    def create_summary(self):
        """Create a summary of extracted data."""
        summary = {
            'extraction_info': {
                'bag_path': str(self.bag_path),
                'output_path': str(self.output_path),
                'extraction_time': datetime.now().isoformat(),
                'extractor_version': '3.0_integrated',
                'bag_duration_seconds': self.bag_duration
            },
            'data_summary': {
                'total_images': len(self.images_data),
                'total_cmd_vel': len(self.cmd_vel_data),
                'total_cmd_vel_manual': len(self.cmd_vel_manual_data),
                'total_joy': len(self.joy_data)
            },
            'extraction_methods': {
                'commands': 'SQLite direct access',
                'images': 'ROS2 bag play + image_view',
                'joystick': 'SQLite simple parsing'
            }
        }
        
        # Add timing analysis
        if self.cmd_vel_manual_data:
            timestamps = [cmd['timestamp'] for cmd in self.cmd_vel_manual_data]
            summary['timing_analysis'] = {
                'start_time': min(timestamps),
                'end_time': max(timestamps),
                'duration_ns': max(timestamps) - min(timestamps),
                'duration_seconds': (max(timestamps) - min(timestamps)) / 1e9,
                'command_frequency': len(timestamps) / ((max(timestamps) - min(timestamps)) / 1e9)
            }
        
        # Save summary
        with open(self.output_path / "metadata" / "extraction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n📊 EXTRACTION SUMMARY:")
        print(f"Images extracted: {summary['data_summary']['total_images']}")
        print(f"Motor commands: {summary['data_summary']['total_cmd_vel']}")
        print(f"Manual commands: {summary['data_summary']['total_cmd_vel_manual']}")
        print(f"Joystick records: {summary['data_summary']['total_joy']}")
        
        if 'timing_analysis' in summary:
            print(f"Duration: {summary['timing_analysis']['duration_seconds']:.2f} seconds")
            print(f"Command frequency: {summary['timing_analysis']['command_frequency']:.2f} Hz")
        
        print(f"\n📁 All data saved to: {self.output_path}")
        
        # Print file structure
        print(f"\n📋 File structure:")
        print(f"├── images/ ({len(self.images_data)} PNG files)")
        print(f"├── data/ (CSV and JSON files)")
        print(f"└── metadata/ (extraction summary)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract data from ROS2 bag files (Integrated approach)')
    parser.add_argument('bag_path', help='Path to the bag directory or .db3 file')
    parser.add_argument('-o', '--output', help='Output directory (default: bag_path + _extracted)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output:
        output_path = args.output
    else:
        bag_name = Path(args.bag_path).stem
        if bag_name.endswith('_0'):
            bag_name = bag_name[:-2]  # Remove _0 suffix
        output_path = Path(args.bag_path).parent / f"{bag_name}_extracted"
    
    print("🎒 Integrated ROS2 Bag Data Extractor")
    print(f"Input: {args.bag_path}")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    try:
        # Extract data
        extractor = IntegratedBagExtractor(args.bag_path, output_path)
        success = extractor.extract_data()
        
        if success:
            print("\n✅ Extraction completed successfully!")
            print("\n🎯 Ready for behavior cloning!")
            print("   📸 Images: Visual input for neural networks")
            print("   🎮 Manual commands: Training labels (what you wanted)")
            print("   🚗 Motor commands: Actual outputs (what robot did)")
        else:
            print("\n❌ Extraction failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())