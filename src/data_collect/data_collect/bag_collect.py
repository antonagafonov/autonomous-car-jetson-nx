#!/usr/bin/env python3
"""
Alternative Smart ROS2 Bag Recorder using subprocess calls
Compatible with ROS2 Foxy on Jetson Xavier NX

This version uses subprocess calls to 'ros2 bag record' instead of rosbag2_py
to avoid import issues on some systems.

Modified to listen to recording_trigger from joystick controller.
Enhanced with inference/perception topic monitoring.
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
from std_msgs.msg import Bool, String, Float32, Int32
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
        self.failed_attempts = 0
        self.last_failure_time = 0
        self.recording_trigger_active = False  # Track trigger state
        self.shutdown_initiated = False  # Prevent multiple shutdowns
        self.metadata_save_timer = None  # Track metadata save timer
        
        # Topic monitoring
        self.topic_last_received = {}
        self.topic_message_counts = {}
        
        # Inference monitoring
        self.inference_active = False
        self.last_inference_confidence = 0.0
        self.last_inference_status = "unknown"
        self.inference_queue_size = 0
        
        # Quality metrics
        self.recording_stats = {
            'total_messages': 0,
            'total_images': 0,
            'total_commands': 0,
            'total_autonomous_commands': 0,  # NEW: Track autonomous commands
            'total_inference_predictions': 0,  # NEW: Track inference predictions
            'session_duration': 0.0,
            'average_fps': 0.0,
            'average_inference_confidence': 0.0,  # NEW: Track average confidence
            'inference_active_time': 0.0,  # NEW: Track how long inference was active
            'storage_size_mb': 0.0
        }
        
        self.setup_subscribers()
        self.setup_services()
        self.setup_timers()
        
        self.get_logger().info("🎬 Bag Recorder initialized (subprocess mode)")
        self.get_logger().info(f"📁 Storage path: {self.config['storage']['base_path']}")
        self.get_logger().info("🎮 Recording trigger mode: Waiting for joystick X button")
        self.get_logger().info("🤖 Enhanced with inference topic monitoring")
        
    def start_process_monitor(self):
        """Start a thread to monitor the recording process output."""
        def monitor_process():
            if not self.recording_process:
                return
                
            try:
                # Monitor stderr for errors
                while self.recording_process and self.recording_process.poll() is None:
                    if self.recording_process.stderr:
                        line = self.recording_process.stderr.readline()
                        if line:
                            self.get_logger().warn(f"📦 Bag record stderr: {line.strip()}")
                    time.sleep(0.1)
                    
                # Check final exit status
                if self.recording_process:
                    exit_code = self.recording_process.returncode
                    if exit_code != 0:
                        stdout, stderr = self.recording_process.communicate()
                        self.get_logger().error(f"❌ Recording process exited with code {exit_code}")
                        if stderr:
                            self.get_logger().error(f"STDERR: {stderr}")
                    
            except Exception as e:
                # Only log if we're still recording (avoid shutdown errors)
                if self.is_recording and not self.shutdown_initiated:
                    self.get_logger().error(f"❌ Error monitoring process: {e}")
        
        self.process_monitor_thread = threading.Thread(target=monitor_process, daemon=True)
        self.process_monitor_thread.start()
    
    def load_config(self):
        """Load configuration with fallback to defaults."""
        self.config = {
            'storage': {
                'base_path': os.path.expanduser('~/car_datasets'),
                'max_bagfile_size': '1GB',
                'storage_id': 'sqlite3'  # This is the correct default
            },
            'quality': {
                'min_recording_duration': 10.0,
                'inactive_timeout': 30.0,  # Increased timeout when using manual trigger
                'min_linear_velocity': 0.1,
                'min_angular_velocity': 0.05
            },
            'session': {
                'use_timestamp_naming': True,
                'session_prefix': 'behavior_',
                'auto_start_on_joystick': False,  # Disabled - now using trigger
                'auto_stop_on_inactive': True
            }
        }
        
        # Topics to record - Enhanced with inference topics
        self.topics_to_record = [
            # Sensor data
            '/camera/image_raw',
            
            # Manual control
            '/cmd_vel_manual', 
            '/joy',
            '/cmd_vel',
            '/recording_trigger',
            
            # Inference predictions (autonomous driving)
            '/cmd_vel_autonomous',        # Main inference commands
            '/car/angular_prediction',    # Raw angular velocity predictions
            '/car/inference_confidence',  # Confidence scores
            '/car/queue_size',           # Prediction queue status
            '/car/inference_status',     # Inference node status
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
        
        # Manual command subscriber for activity monitoring
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_manual', self.cmd_callback, qos
        )
        
        # Autonomous command subscriber for monitoring
        self.autonomous_cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_autonomous', self.autonomous_cmd_callback, qos
        )
        
        # Image subscriber for quality monitoring
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos
        )
        
        # Recording trigger subscriber
        self.recording_trigger_sub = self.create_subscription(
            Bool, '/recording_trigger', self.recording_trigger_callback, 10
        )
        
        # NEW: Inference monitoring subscribers
        self.angular_prediction_sub = self.create_subscription(
            Float32, '/car/angular_prediction', self.angular_prediction_callback, qos
        )
        
        self.inference_confidence_sub = self.create_subscription(
            Float32, '/car/inference_confidence', self.inference_confidence_callback, qos
        )
        
        self.queue_size_sub = self.create_subscription(
            Float32, '/car/queue_size', self.queue_size_callback, qos
        )
        
        self.inference_status_sub = self.create_subscription(
            String, '/car/inference_status', self.inference_status_callback, qos
        )
        
        self.get_logger().info("👂 Topic subscribers setup complete (including inference topics)")
    
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
    
    def recording_trigger_callback(self, msg: Bool):
        """Handle recording trigger from joystick controller."""
        self.recording_trigger_active = msg.data
        
        if msg.data and not self.is_recording:
            # Start recording
            self.get_logger().info("🎮 Recording trigger activated - starting recording")
            success = self.start_recording()
            if not success:
                self.get_logger().error("❌ Failed to start recording on trigger")
        elif not msg.data and self.is_recording:
            # Stop recording
            self.get_logger().info("🎮 Recording trigger deactivated - stopping recording")
            success = self.stop_recording()
            if not success:
                self.get_logger().error("❌ Failed to stop recording on trigger")
    
    def joy_callback(self, msg: Joy):
        """Handle joystick messages for activity detection."""
        self.last_activity_time = time.time()
        
        # NOTE: Auto-start functionality disabled when using trigger mode
        # The recording is now controlled exclusively by the recording_trigger topic
    
    def cmd_callback(self, msg: Twist):
        """Handle manual command messages for activity detection."""
        # Check for significant movement commands
        if (abs(msg.linear.x) > self.config['quality']['min_linear_velocity'] or
            abs(msg.angular.z) > self.config['quality']['min_angular_velocity']):
            self.last_activity_time = time.time()
            
            if self.is_recording:
                self.recording_stats['total_commands'] += 1
    
    def autonomous_cmd_callback(self, msg: Twist):
        """NEW: Handle autonomous command messages for monitoring."""
        if self.is_recording:
            self.recording_stats['total_autonomous_commands'] += 1
            
        # Update topic monitoring
        self.topic_last_received['/cmd_vel_autonomous'] = time.time()
        self.topic_message_counts['/cmd_vel_autonomous'] = \
            self.topic_message_counts.get('/cmd_vel_autonomous', 0) + 1
            
        # Check for significant autonomous movement
        if (abs(msg.linear.x) > self.config['quality']['min_linear_velocity'] or
            abs(msg.angular.z) > self.config['quality']['min_angular_velocity']):
            self.last_activity_time = time.time()
            self.inference_active = True
    
    def angular_prediction_callback(self, msg: Float32):
        """NEW: Handle angular prediction messages for monitoring."""
        if self.is_recording:
            self.recording_stats['total_inference_predictions'] += 1
            
        # Update topic monitoring
        self.topic_last_received['/car/angular_prediction'] = time.time()
        self.topic_message_counts['/car/angular_prediction'] = \
            self.topic_message_counts.get('/car/angular_prediction', 0) + 1
    
    def inference_confidence_callback(self, msg: Float32):
        """NEW: Handle inference confidence messages for monitoring."""
        self.last_inference_confidence = msg.data
        
        # Update average confidence (simple moving average)
        if self.is_recording:
            current_avg = self.recording_stats['average_inference_confidence']
            total_predictions = self.recording_stats['total_inference_predictions']
            
            if total_predictions > 0:
                # Update running average
                self.recording_stats['average_inference_confidence'] = \
                    (current_avg * (total_predictions - 1) + msg.data) / total_predictions
            else:
                self.recording_stats['average_inference_confidence'] = msg.data
        
        # Update topic monitoring
        self.topic_last_received['/car/inference_confidence'] = time.time()
        self.topic_message_counts['/car/inference_confidence'] = \
            self.topic_message_counts.get('/car/inference_confidence', 0) + 1
    
    def queue_size_callback(self, msg: Int32):
        """NEW: Handle queue size messages for monitoring."""
        self.inference_queue_size = msg.data
        
        # Update topic monitoring
        self.topic_last_received['/car/queue_size'] = time.time()
        self.topic_message_counts['/car/queue_size'] = \
            self.topic_message_counts.get('/car/queue_size', 0) + 1
    
    def inference_status_callback(self, msg: String):
        """NEW: Handle inference status messages for monitoring."""
        self.last_inference_status = msg.data
        
        # Track inference active time
        if self.is_recording and msg.data.lower() in ['active', 'running', 'predicting']:
            self.inference_active = True
        
        # Update topic monitoring
        self.topic_last_received['/car/inference_status'] = time.time()
        self.topic_message_counts['/car/inference_status'] = \
            self.topic_message_counts.get('/car/inference_status', 0) + 1
    
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
        
        # Track inference active time
        if self.inference_active and self.is_recording:
            self.recording_stats['inference_active_time'] += 1.0  # Add 1 second
            self.inference_active = False  # Reset for next check
        
        # Only auto-stop on inactivity if the recording trigger is still active
        # This prevents stopping due to inactivity when the user wants to keep recording
        if (self.config['session']['auto_stop_on_inactive'] and
            self.recording_trigger_active and  # Check trigger is still active
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
        trigger_status = "🟢 ACTIVE" if self.recording_trigger_active else "🔴 INACTIVE"
        inference_status = f"🤖 {self.last_inference_status}" if self.last_inference_status else "🤖 Unknown"
        
        self.get_logger().info(
            f"📊 Recording stats: "
            f"{stats['session_duration']:.1f}s, "
            f"{stats['total_images']} images, "
            f"{stats['total_commands']} manual cmds, "
            f"{stats['total_autonomous_commands']} auto cmds, "
            f"{stats['total_inference_predictions']} predictions, "
            f"{stats['average_fps']:.1f} fps, "
            f"conf: {self.last_inference_confidence:.2f}, "
            f"queue: {self.inference_queue_size}, "
            f"Trigger: {trigger_status}, "
            f"Inference: {inference_status}"
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
        
        trigger_status = "ACTIVE" if self.recording_trigger_active else "INACTIVE"
        
        if self.is_recording:
            stats = self.recording_stats
            response.message = (
                f"Recording: {self.current_session}\n"
                f"Duration: {stats['session_duration']:.1f}s\n"
                f"Images: {stats['total_images']}\n"
                f"Manual Commands: {stats['total_commands']}\n"
                f"Autonomous Commands: {stats['total_autonomous_commands']}\n"
                f"Inference Predictions: {stats['total_inference_predictions']}\n"
                f"FPS: {stats['average_fps']:.1f}\n"
                f"Avg Confidence: {stats['average_inference_confidence']:.2f}\n"
                f"Inference Active Time: {stats['inference_active_time']:.1f}s\n"
                f"Current Confidence: {self.last_inference_confidence:.2f}\n"
                f"Queue Size: {self.inference_queue_size}\n"
                f"Inference Status: {self.last_inference_status}\n"
                f"Trigger: {trigger_status}"
            )
        else:
            response.message = (
                f"Not recording (Trigger: {trigger_status})\n"
                f"Last Inference Status: {self.last_inference_status}\n"
                f"Last Confidence: {self.last_inference_confidence:.2f}"
            )
            
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
    
    def start_recording(self) -> bool:
        """Start bag recording session using subprocess."""
        if self.is_recording:
            self.get_logger().warn("⚠️  Already recording")
            return False
            
        try:
            # Create session directory path but DON'T create it yet
            # ros2 bag record will create the directory
            session_name = self.generate_session_name()
            self.session_path = os.path.join(
                self.config['storage']['base_path'], 
                session_name
            )
            
            # Ensure base directory exists (but not the session directory)
            os.makedirs(self.config['storage']['base_path'], exist_ok=True)
            
            # Build ros2 bag record command with proper Foxy arguments
            cmd = [
                'ros2', 'bag', 'record',
                '-o', self.session_path  # ros2 bag will create this directory
            ]
            
            # Try to add compression if available
            try:
                # Test compression support
                test_cmd = ['ros2', 'bag', 'record', '--compression-mode', 'file', '--help']
                result = subprocess.run(test_cmd, capture_output=True, timeout=3)
                if result.returncode == 0:
                    cmd.extend(['--compression-mode', 'file', '--compression-format', 'zstd'])
                    self.get_logger().info("✅ Using compression")
                else:
                    self.get_logger().warn("⚠️  Compression not available, using uncompressed")
            except:
                self.get_logger().warn("⚠️  Could not test compression, using uncompressed")
            
            # Add topics to record
            cmd.extend(self.topics_to_record)
            
            self.get_logger().info(f"🚀 Starting recording with command: {' '.join(cmd)}")
            
            # Start subprocess with better error handling
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config['storage']['base_path'],  # Run from base directory
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
                
                # Track failure
                self.failed_attempts += 1
                self.last_failure_time = time.time()
                
                return False
            
            # Start a thread to monitor the process output
            self.start_process_monitor()
            
            # Reset failure counter on successful start
            self.failed_attempts = 0
            
            # Update state
            self.is_recording = True
            self.current_session = session_name
            self.session_start_time = time.time()
            self.last_activity_time = time.time()
            
            # Reset stats including new inference stats
            self.recording_stats = {
                'total_messages': 0,
                'total_images': 0,
                'total_commands': 0,
                'total_autonomous_commands': 0,
                'total_inference_predictions': 0,
                'session_duration': 0.0,
                'average_fps': 0.0,
                'average_inference_confidence': 0.0,
                'inference_active_time': 0.0,
                'storage_size_mb': 0.0
            }
            
            # Save session metadata after a short delay to let ros2 bag create the directory
            self.metadata_save_timer = self.create_timer(3.0, self.save_session_metadata_delayed)
            
            self.get_logger().info(f"🎬 Recording started: {session_name}")
            self.get_logger().info(f"📁 Path: {self.session_path}")
            self.get_logger().info(f"📋 Topics: {len(self.topics_to_record)} (including inference topics)")
            
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
                # Make sure we have proper ROS2 metadata
                self.create_ros2_metadata_if_missing()
                # Update our custom metadata
                self.update_session_metadata()
            
            # Log final stats
            self.log_final_stats()
            
            # Reset state
            self.is_recording = False
            session_name = self.current_session
            self.current_session = None
            self.session_start_time = None
            
            # Clean up process reference
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
    
    def save_session_metadata_delayed(self):
        """Save session metadata after ros2 bag has created the directory."""
        try:
            # Prevent multiple calls
            if self.metadata_save_timer is not None:
                self.metadata_save_timer.cancel()
                self.metadata_save_timer = None
                
            # Check if directory was created by ros2 bag
            if os.path.exists(self.session_path):
                # First create the ROS2 metadata.yaml if it doesn't exist
                self.create_ros2_metadata_if_missing()
                
                # Then save our custom metadata
                self.save_session_metadata()
                self.get_logger().info(f"💾 Metadata saved: {self.session_path}/metadata.json")
            else:
                # Try again in a few seconds, but only once more
                self.metadata_save_timer = self.create_timer(2.0, self.save_session_metadata_delayed)
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save delayed metadata: {e}")
    
    def create_ros2_metadata_if_missing(self):
        """Create metadata.yaml if ros2 bag didn't create it."""
        metadata_yaml_path = os.path.join(self.session_path, 'metadata.yaml')
        
        if not os.path.exists(metadata_yaml_path):
            try:
                # Get the actual bag file name
                bag_files = [f for f in os.listdir(self.session_path) if f.endswith('.db3')]
                
                if bag_files:
                    bag_file = bag_files[0]  # Use the first bag file
                    
                    # Query the database for actual information
                    db_path = os.path.join(self.session_path, bag_file)
                    
                    # Get basic info from the database
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get topics
                    cursor.execute("SELECT name, type FROM topics")
                    topics_data = cursor.fetchall()
                    
                    # Get message counts
                    cursor.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id")
                    message_counts = dict(cursor.fetchall())
                    
                    # Get time range
                    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
                    time_range = cursor.fetchone()
                    
                    conn.close()
                    
                    # Create metadata.yaml content
                    topics_with_counts = []
                    total_messages = 0
                    
                    for i, (topic_name, topic_type) in enumerate(topics_data):
                        count = message_counts.get(i + 1, 0)  # topic_id starts from 1
                        total_messages += count
                        
                        # Correct YAML structure for topics_with_message_count
                        topic_entry = {
                            'topic_metadata': {
                                'name': topic_name,
                                'type': topic_type,
                                'serialization_format': 'cdr'
                            },
                            'message_count': count
                        }
                        topics_with_counts.append(topic_entry)
                    
                    # Calculate duration
                    start_time = time_range[0] if time_range[0] else 0
                    end_time = time_range[1] if time_range[1] else 0
                    duration = end_time - start_time
                    
                    # Create the metadata structure
                    metadata = {
                        'rosbag2_bagfile_information': {
                            'version': 4,
                            'storage_identifier': 'sqlite3',
                            'relative_file_paths': [bag_file],
                            'duration': {
                                'nanoseconds': int(duration)
                            },
                            'starting_time': {
                                'nanoseconds_since_epoch': int(start_time)
                            },
                            'message_count': total_messages,
                            'topics_with_message_count': topics_with_counts,
                            'compression_format': '',
                            'compression_mode': ''
                        }
                    }
                    
                    # Write metadata.yaml with proper formatting
                    import yaml
                    with open(metadata_yaml_path, 'w') as f:
                        yaml.dump(metadata, f, default_flow_style=False, indent=2, sort_keys=False)
                    
                    self.get_logger().info(f"✅ Created metadata.yaml with {total_messages} messages")
                    
                else:
                    self.get_logger().warn("⚠️  No .db3 files found in recording directory")
                    
            except Exception as e:
                self.get_logger().error(f"❌ Failed to create metadata.yaml: {e}")
                # Create a minimal fallback metadata.yaml
                self.create_minimal_metadata_yaml(metadata_yaml_path)
    
    def create_minimal_metadata_yaml(self, metadata_yaml_path):
        """Create a minimal metadata.yaml as fallback."""
        try:
            bag_files = [f for f in os.listdir(self.session_path) if f.endswith('.db3')]
            bag_file = bag_files[0] if bag_files else 'rosbag2_0.db3'
            
            minimal_metadata = {
                'rosbag2_bagfile_information': {
                    'version': 4,
                    'storage_identifier': 'sqlite3',
                    'relative_file_paths': [bag_file],
                    'duration': {'nanoseconds': 0},
                    'starting_time': {'nanoseconds_since_epoch': 0},
                    'message_count': 0,
                    'topics_with_message_count': [],
                    'compression_format': '',
                    'compression_mode': ''
                }
            }
            
            import yaml
            with open(metadata_yaml_path, 'w') as f:
                yaml.dump(minimal_metadata, f, default_flow_style=False)
                
            self.get_logger().info("✅ Created minimal metadata.yaml")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to create minimal metadata.yaml: {e}")
    
    def save_session_metadata(self):
        """Save session metadata to JSON file."""
        try:
            metadata = {
                'session_info': {
                    'session_name': self.current_session,
                    'start_time': datetime.now().isoformat(),
                    'start_timestamp': self.session_start_time,
                    'topics': self.topics_to_record,
                    'recording_trigger_mode': True,
                    'inference_topics_included': True  # NEW: Flag for inference topics
                },
                'system_info': {
                    'ros_distro': os.environ.get('ROS_DISTRO', 'unknown'),
                    'hostname': os.uname().nodename,
                    'platform': os.uname().sysname,
                    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
                },
                'config': self.config,
                'statistics': self.recording_stats.copy(),
                'inference_info': {  # NEW: Inference-specific metadata
                    'last_inference_status': self.last_inference_status,
                    'final_confidence': self.last_inference_confidence,
                    'final_queue_size': self.inference_queue_size,
                    'inference_topics': [
                        '/cmd_vel_autonomous',
                        '/car/angular_prediction',
                        '/car/inference_confidence',
                        '/car/queue_size',
                        '/car/inference_status'
                    ]
                }
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
            
            # Update inference info
            if 'inference_info' not in metadata:
                metadata['inference_info'] = {}
            
            metadata['inference_info'].update({
                'final_inference_status': self.last_inference_status,
                'final_confidence': self.last_inference_confidence,
                'final_queue_size': self.inference_queue_size,
                'total_inference_time': self.recording_stats['inference_active_time']
            })
            
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
        self.get_logger().info(f"  Manual Commands: {stats['total_commands']}")
        self.get_logger().info(f"  Autonomous Commands: {stats['total_autonomous_commands']}")
        self.get_logger().info(f"  Inference Predictions: {stats['total_inference_predictions']}")
        self.get_logger().info(f"  Average FPS: {stats['average_fps']:.1f}")
        self.get_logger().info(f"  Average Inference Confidence: {stats['average_inference_confidence']:.2f}")
        self.get_logger().info(f"  Inference Active Time: {stats['inference_active_time']:.1f}s")
        self.get_logger().info(f"  Final Inference Status: {self.last_inference_status}")
        self.get_logger().info(f"  Final Queue Size: {self.inference_queue_size}")
        self.get_logger().info(f"  Recording trigger mode: {'ON' if self.recording_trigger_active else 'OFF'}")
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        if self.shutdown_initiated:
            return
        self.shutdown_initiated = True
        
        self.get_logger().info("🏠 Bag collector shutting down gracefully...")
        
        # Cancel any pending metadata save timer
        if self.metadata_save_timer is not None:
            self.metadata_save_timer.cancel()
            self.metadata_save_timer = None
        
        if self.is_recording:
            self.get_logger().info("🛑 Stopping recording due to shutdown...")
            self.stop_recording()
            # Give a moment for the recording to stop properly
            import time
            time.sleep(1.0)
        
        self.get_logger().info("✅ Bag collector shutdown complete")
        super().destroy_node()


def main(args=None):
    """Main entry point for the bag recorder node."""
    rclpy.init(args=args)
    
    try:
        node = BagRecorder()
        
        # Handle shutdown gracefully
        def signal_handler(signum, frame):
            if node.shutdown_initiated:
                return
            node.get_logger().info("🏠 Shutdown signal received - cleaning up...")
            if node.is_recording:
                node.get_logger().info("🛑 Stopping active recording...")
                node.stop_recording()
                # Give time for recording to stop
                import time
                time.sleep(1.0)
            node.get_logger().info("✅ Cleanup complete")
            node.destroy_node()
            rclpy.shutdown()
            exit(0)
        
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