#!/usr/bin/env python3
"""
Alternative Smart ROS2 Bag Recorder using subprocess calls
Compatible with ROS2 Foxy on Jetson Xavier NX

This version uses subprocess calls to 'ros2 bag record' instead of rosbag2_py
to avoid import issues on some systems.
"""

import os
import time
import json
import yaml
import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

# Message types
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, Image
from std_srvs.srv import SetBool, Trigger


class BagRecorder(Node):
    """Smart ROS2 bag recorder using subprocess for better compatibility."""
    
    def __init__(self):
        super().__init__('bag_collect')
        
        # Callback group for services
        self.service_group = ReentrantCallbackGroup()
        
        # Load configuration
        self.load_config()
        
        # Initialize state
        self.is_recording = False
        self.current_session = None
        self.session_start_time = None
        self.last_activity_time = time.time()
        self.recording_process = None
        self.session_path = None
        self.process_monitor_thread = None
        
        # Topic monitoring
        self.topic_last_received = {}
        self.topic_message_counts = {}
        
        # Quality metrics
        self.recording_stats = {
            'total_messages': 0,
            'total_images': 0,
            'total_commands': 0,
            'session_duration': 0.0,
            'average_fps': 0.0,
            'storage_size_mb': 0.0
        }
        
        self.setup_subscribers()
        self.setup_services()
        self.setup_timers()
        
        self.get_logger().info("🎬 Bag Recorder initialized (subprocess mode)")
        self.get_logger().info(f"📁 Storage path: {self.config['storage']['base_path']}")
        
    def load_config(self):
        """Load configuration with fallback to defaults."""
        self.config = {
            'storage': {
                'base_path': os.path.expanduser('~/car_datasets'),
                'max_bagfile_size': 0,  # 0 means no limit for Foxy
                'compression_mode': 'none',  # Foxy only supports 'none' and 'file'
                'storage_id': 'sqlite3'
            },
            'quality': {
                'min_recording_duration': 10.0,
                'inactive_timeout': 5.0,
                'min_linear_velocity': 0.1,
                'min_angular_velocity': 0.05
            },
            'session': {
                'use_timestamp_naming': True,
                'session_prefix': 'behavior_',
                'auto_start_on_joystick': True,
                'auto_stop_on_inactive': True
            }
        }
        
        # Topics to record
        self.topics_to_record = [
            '/camera/image_raw',
            '/cmd_vel_manual', 
            '/joy',
            '/cmd_vel'
        ]
        
        # Create base directory if it doesn't exist
        os.makedirs(self.config['storage']['base_path'], exist_ok=True)
        
        self.get_logger().info("✅ Configuration loaded (using defaults)")
    
    def setup_subscribers(self):
        """Setup subscribers for monitoring topics."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Joy subscriber for activity monitoring
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.joy_callback, qos
        )
        
        # Command subscriber for activity monitoring
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_manual', self.cmd_callback, qos
        )
        
        # Image subscriber for quality monitoring
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos
        )
        
        self.get_logger().info("👂 Topic subscribers setup complete")
    
    def setup_services(self):
        """Setup service interfaces for recording control."""
        # Start/stop recording service
        self.start_recording_srv = self.create_service(
            SetBool, 'start_recording', 
            self.start_recording_callback,
            callback_group=self.service_group
        )
        
        # Get recording status service
        self.status_srv = self.create_service(
            Trigger, 'recording_status',
            self.status_callback,
            callback_group=self.service_group
        )
        
        # Stop recording service
        self.stop_recording_srv = self.create_service(
            Trigger, 'stop_recording',
            self.stop_recording_callback,
            callback_group=self.service_group
        )
        
        self.get_logger().info("🔧 Services setup complete")
    
    def setup_timers(self):
        """Setup periodic timers for monitoring and cleanup."""
        # Activity monitoring timer
        self.activity_timer = self.create_timer(1.0, self.check_activity)
        
        # Stats monitoring timer
        self.stats_timer = self.create_timer(5.0, self.update_stats)
    
    def joy_callback(self, msg: Joy):
        """Handle joystick messages for activity detection."""
        self.last_activity_time = time.time()
        
        # Check for significant joystick input
        if any(abs(axis) > 0.1 for axis in msg.axes) or any(msg.buttons):
            if (self.config['session']['auto_start_on_joystick'] and 
                not self.is_recording):
                self.get_logger().info("🎮 Joystick activity detected - auto-starting recording")
                self.start_recording()
    
    def cmd_callback(self, msg: Twist):
        """Handle command messages for activity detection."""
        # Check for significant movement commands
        if (abs(msg.linear.x) > self.config['quality']['min_linear_velocity'] or
            abs(msg.angular.z) > self.config['quality']['min_angular_velocity']):
            self.last_activity_time = time.time()
            
            if self.is_recording:
                self.recording_stats['total_commands'] += 1
    
    def image_callback(self, msg: Image):
        """Handle image messages for quality monitoring."""
        if self.is_recording:
            self.recording_stats['total_images'] += 1
            
        # Update topic monitoring
        self.topic_last_received['/camera/image_raw'] = time.time()
        self.topic_message_counts['/camera/image_raw'] = \
            self.topic_message_counts.get('/camera/image_raw', 0) + 1
    
    def check_activity(self):
        """Check for activity and auto-stop if inactive."""
        if not self.is_recording:
            return
            
        current_time = time.time()
        inactive_duration = current_time - self.last_activity_time
        
        if (self.config['session']['auto_stop_on_inactive'] and
            inactive_duration > self.config['quality']['inactive_timeout']):
            
            self.get_logger().info(
                f"⏱️  No activity for {inactive_duration:.1f}s - auto-stopping recording"
            )
            self.stop_recording()
    
    def update_stats(self):
        """Update recording statistics."""
        if not self.is_recording or not self.session_start_time:
            return
            
        current_time = time.time()
        self.recording_stats['session_duration'] = current_time - self.session_start_time
        
        # Calculate FPS
        if self.recording_stats['session_duration'] > 0:
            self.recording_stats['average_fps'] = (
                self.recording_stats['total_images'] / 
                self.recording_stats['session_duration']
            )
        
        # Log stats periodically
        if int(current_time) % 30 == 0:  # Every 30 seconds
            self.log_stats()
    
    def log_stats(self):
        """Log current recording statistics."""
        stats = self.recording_stats
        self.get_logger().info(
            f"📊 Recording stats: "
            f"{stats['session_duration']:.1f}s, "
            f"{stats['total_images']} images, "
            f"{stats['total_commands']} commands, "
            f"{stats['average_fps']:.1f} fps"
        )
    
    def start_recording_callback(self, request, response):
        """Handle start recording service call."""
        try:
            if request.data:
                success = self.start_recording()
                response.success = success
                response.message = "Recording started" if success else "Failed to start recording"
            else:
                success = self.stop_recording()
                response.success = success  
                response.message = "Recording stopped" if success else "Failed to stop recording"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
    
    def status_callback(self, request, response):
        """Handle status service call."""
        response.success = True
        
        if self.is_recording:
            stats = self.recording_stats
            response.message = (
                f"Recording: {self.current_session}\n"
                f"Duration: {stats['session_duration']:.1f}s\n"
                f"Images: {stats['total_images']}\n"
                f"Commands: {stats['total_commands']}\n"
                f"FPS: {stats['average_fps']:.1f}"
            )
        else:
            response.message = "Not recording"
            
        return response
    
    def stop_recording_callback(self, request, response):
        """Handle stop recording service call."""
        try:
            success = self.stop_recording()
            response.success = success
            response.message = "Recording stopped" if success else "Not recording"
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
    
    def start_process_monitor(self):
        """Start a thread to monitor the recording process output."""
        if self.process_monitor_thread and self.process_monitor_thread.is_alive():
            return
            
        self.process_monitor_thread = threading.Thread(
            target=self._monitor_process_output,
            daemon=True
        )
        self.process_monitor_thread.start()
    
    def _monitor_process_output(self):
        """Monitor the subprocess output in a separate thread."""
        try:
            if not self.recording_process:
                return
                
            # Read stdout and stderr
            while self.recording_process.poll() is None:
                # Check if process is still running
                if self.recording_process.stdout:
                    output = self.recording_process.stdout.readline()
                    if output:
                        self.get_logger().info(f"📝 Bag record: {output.strip()}")
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
        except Exception as e:
            self.get_logger().warn(f"⚠️  Process monitor error: {e}")
    
    def start_recording(self) -> bool:
        """Start bag recording session using subprocess."""
        if self.is_recording:
            self.get_logger().warn("⚠️  Already recording")
            return False
            
        try:
            # Create session directory
            session_name = self.generate_session_name()
            self.session_path = os.path.join(
                self.config['storage']['base_path'], 
                session_name
            )
            
            # Ensure directory exists
            os.makedirs(self.session_path, exist_ok=True)
            
            # Build ros2 bag record command for Foxy
            cmd = ['ros2', 'bag', 'record', '-o', self.session_path]
            
            # Add storage (use -s for Foxy)
            cmd.extend(['-s', self.config['storage']['storage_id']])
            
            # Only add max bag size if not 0 (Foxy uses -b)
            if self.config['storage']['max_bagfile_size'] > 0:
                cmd.extend(['-b', str(self.config['storage']['max_bagfile_size'])])
            
            # Add compression only if supported (Foxy is limited)
            if self.config['storage']['compression_mode'] == 'file':
                cmd.extend(['--compression-mode', 'file'])
            
            # Add topics
            cmd.extend(self.topics_to_record)
            
            self.get_logger().info(f"🚀 Starting recording with command: {' '.join(cmd)}")
            
            # Start subprocess with better error handling
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.session_path,
                env=os.environ.copy(),
                text=True,
                bufsize=1
            )
            
            # Wait a moment to check if process started successfully
            time.sleep(2.0)
            
            if self.recording_process.poll() is not None:
                # Process has already terminated
                stdout, stderr = self.recording_process.communicate()
                self.get_logger().error(f"❌ Recording process failed to start:")
                self.get_logger().error(f"STDOUT: {stdout}")
                self.get_logger().error(f"STDERR: {stderr}")
                return False
            
            # Start a thread to monitor the process output
            self.start_process_monitor()
            
            # Update state
            self.is_recording = True
            self.current_session = session_name
            self.session_start_time = time.time()
            self.last_activity_time = time.time()
            
            # Reset stats
            self.recording_stats = {
                'total_messages': 0,
                'total_images': 0,
                'total_commands': 0,
                'session_duration': 0.0,
                'average_fps': 0.0,
                'storage_size_mb': 0.0
            }
            
            # Save session metadata
            self.save_session_metadata()
            
            self.get_logger().info(f"🎬 Recording started: {session_name}")
            self.get_logger().info(f"📁 Path: {self.session_path}")
            self.get_logger().info(f"📋 Topics: {len(self.topics_to_record)}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to start recording: {e}")
            self.is_recording = False
            self.current_session = None
            return False
    
    def stop_recording(self) -> bool:
        """Stop current recording session."""
        if not self.is_recording:
            self.get_logger().warn("⚠️  Not currently recording")
            return False
            
        try:
            # Stop subprocess
            if self.recording_process and self.recording_process.poll() is None:
                self.get_logger().info("🛑 Stopping recording process...")
                self.recording_process.terminate()
                
                # Give it some time to terminate gracefully
                try:
                    self.recording_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.get_logger().warn("⚠️  Recording process didn't stop gracefully, killing...")
                    self.recording_process.kill()
                    self.recording_process.wait()
            
            # Calculate final stats
            if self.session_start_time:
                final_duration = time.time() - self.session_start_time
                self.recording_stats['session_duration'] = final_duration
                
                # Check minimum duration
                if final_duration < self.config['quality']['min_recording_duration']:
                    self.get_logger().warn(
                        f"⚠️  Short recording: {final_duration:.1f}s "
                        f"(min: {self.config['quality']['min_recording_duration']}s)"
                    )
            
            # Update session metadata with final stats
            if self.session_path:
                self.update_session_metadata()
            
            # Log final stats
            self.log_final_stats()
            
            # Reset state
            self.is_recording = False
            session_name = self.current_session
            self.current_session = None
            self.session_start_time = None
            self.recording_process = None
            
            self.get_logger().info(f"🏁 Recording stopped: {session_name}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to stop recording: {e}")
            return False
    
    def generate_session_name(self) -> str:
        """Generate unique session name."""
        if self.config['session']['use_timestamp_naming']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.config['session']['session_prefix']}{timestamp}"
        else:
            # Find next available session number
            base_path = self.config['storage']['base_path']
            prefix = self.config['session']['session_prefix']
            
            session_num = 1
            while True:
                session_name = f"{prefix}{session_num:04d}"
                session_path = os.path.join(base_path, session_name)
                if not os.path.exists(session_path):
                    return session_name
                session_num += 1
    
    def save_session_metadata(self):
        """Save session metadata to JSON file."""
        try:
            metadata = {
                'session_info': {
                    'session_name': self.current_session,
                    'start_time': datetime.now().isoformat(),
                    'start_timestamp': self.session_start_time,
                    'topics': self.topics_to_record
                },
                'system_info': {
                    'ros_distro': os.environ.get('ROS_DISTRO', 'unknown'),
                    'hostname': os.uname().nodename,
                    'platform': os.uname().sysname,
                    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
                },
                'config': self.config,
                'statistics': self.recording_stats.copy()
            }
            
            metadata_path = os.path.join(self.session_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.get_logger().info(f"💾 Metadata saved: {metadata_path}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save metadata: {e}")
    
    def update_session_metadata(self):
        """Update session metadata with final statistics."""
        try:
            metadata_path = os.path.join(self.session_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Update with final stats
            metadata['session_info']['end_time'] = datetime.now().isoformat()
            metadata['session_info']['duration_seconds'] = self.recording_stats['session_duration']
            metadata['statistics'] = self.recording_stats.copy()
            
            # Calculate storage size
            storage_size = self.calculate_directory_size(self.session_path)
            metadata['statistics']['storage_size_mb'] = storage_size / (1024 * 1024)
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to update metadata: {e}")
    
    def calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except Exception as e:
            self.get_logger().warn(f"⚠️  Failed to calculate directory size: {e}")
        return total_size
    
    def log_final_stats(self):
        """Log final recording statistics."""
        stats = self.recording_stats
        self.get_logger().info("📊 Final Recording Statistics:")
        self.get_logger().info(f"  Duration: {stats['session_duration']:.1f} seconds")
        self.get_logger().info(f"  Images: {stats['total_images']}")
        self.get_logger().info(f"  Commands: {stats['total_commands']}")
        self.get_logger().info(f"  Average FPS: {stats['average_fps']:.1f}")
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        if self.is_recording:
            self.get_logger().info("🛑 Stopping recording due to shutdown...")
            self.stop_recording()
        super().destroy_node()


def main(args=None):
    """Main entry point for the bag recorder node."""
    rclpy.init(args=args)
    
    try:
        node = BagRecorder()
        
        # Handle shutdown gracefully
        def signal_handler(signum, frame):
            node.get_logger().info("🛑 Shutdown signal received")
            node.destroy_node()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error in bag recorder: {e}")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()