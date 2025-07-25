#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, String
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import threading
import time
from collections import deque
import os
import json


class ResNet18AngularRegressor(nn.Module):
    """ResNet18 based regression model for predicting sequence of angular_z control values"""
    
    def __init__(self, sequence_length=10, pretrained=False):
        super(ResNet18AngularRegressor, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_outputs = sequence_length
        
        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.resnet18.fc.in_features
        
        # Replace the final fully connected layer with a more complex head
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_outputs)
        )
        
    def forward(self, x):
        output = self.resnet18(x)
        return output

class CommandSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_linear = 0.0
        self.prev_angular = 0.0
    
    def smooth(self, linear, angular):
        # Exponential moving average
        smooth_linear = self.alpha * linear + (1 - self.alpha) * self.prev_linear
        smooth_angular = self.alpha * angular + (1 - self.alpha) * self.prev_angular
        
        self.prev_linear = smooth_linear
        self.prev_angular = smooth_angular
        
        return smooth_linear, smooth_angular

class BurstSteeringController:
    """Controls steering commands in bursts with history tracking"""
    
    def __init__(self, command_history_size=30, burst_steer_count=2, burst_zero_count=3, 
                 max_steer_bursts=15, forced_straight_count=15, steering_threshold=0.1):
        
        # Configuration parameters
        self.command_history_size = command_history_size
        self.burst_steer_count = burst_steer_count
        self.burst_zero_count = burst_zero_count
        self.max_steer_bursts = max_steer_bursts
        self.forced_straight_count = forced_straight_count
        self.steering_threshold = steering_threshold
        
        # Command history tracking
        self.command_history = deque(maxlen=command_history_size)
        
        # Burst state tracking
        self.current_burst_step = 0  # 0 to (burst_steer_count + burst_zero_count - 1)
        self.in_steer_phase = True   # True during steer commands, False during zero commands
        self.pending_steer_command = 0.0
        
        # Burst counting for forcing straight periods
        self.recent_steer_bursts = deque(maxlen=max_steer_bursts)  # Track recent bursts
        self.forced_straight_remaining = 0
        
        # Statistics
        self.total_commands = 0
        self.total_steer_commands = 0
        self.total_zero_commands = 0
        self.total_forced_straight = 0
        self.burst_count = 0
        
    def process(self, linear, angular):
        """Process command through burst control system"""
        
        self.total_commands += 1
        
        # Check if we're in forced straight mode
        if self.forced_straight_remaining > 0:
            self.forced_straight_remaining -= 1
            self.total_forced_straight += 1
            
            # Add to history
            final_angular = 0.0
            self.command_history.append({
                'linear': linear,
                'angular': final_angular,
                'original_angular': angular,
                'type': 'forced_straight',
                'timestamp': time.time()
            })
            
            return linear, final_angular
        
        # Determine if model wants to steer
        wants_to_steer = abs(angular) > self.steering_threshold
        
        # Update pending command if model wants to steer
        if wants_to_steer:
            self.pending_steer_command = angular
        
        # Burst logic
        if self.in_steer_phase:
            # We're in steering phase of burst
            if self.current_burst_step < self.burst_steer_count:
                # Send steering command (either pending or zero if no pending)
                if abs(self.pending_steer_command) > self.steering_threshold:
                    final_angular = self.pending_steer_command
                    command_type = 'burst_steer'
                    self.total_steer_commands += 1
                else:
                    final_angular = 0.0
                    command_type = 'burst_steer_zero'
                    self.total_zero_commands += 1
                
                self.current_burst_step += 1
                
                # Check if steering phase is complete
                if self.current_burst_step >= self.burst_steer_count:
                    self.in_steer_phase = False
                    self.current_burst_step = 0
                    
                    # Record this burst (True if it contained any steering)
                    burst_had_steering = abs(self.pending_steer_command) > self.steering_threshold
                    self.recent_steer_bursts.append(burst_had_steering)
                    self.burst_count += 1
                    
                    # Clear pending command
                    self.pending_steer_command = 0.0
                    
                    # Check if we need to force straight period
                    if self._check_force_straight_needed():
                        self.forced_straight_remaining = self.forced_straight_count
            else:
                # This shouldn't happen, but handle gracefully
                final_angular = 0.0
                command_type = 'burst_error'
                self.total_zero_commands += 1
        
        else:
            # We're in zero phase of burst
            final_angular = 0.0
            command_type = 'burst_zero'
            self.total_zero_commands += 1
            
            self.current_burst_step += 1
            
            # Check if zero phase is complete
            if self.current_burst_step >= self.burst_zero_count:
                self.in_steer_phase = True
                self.current_burst_step = 0
        
        # Add to history
        self.command_history.append({
            'linear': linear,
            'angular': final_angular,
            'original_angular': angular,
            'type': command_type,
            'timestamp': time.time()
        })
        
        return linear, final_angular
    
    def _check_force_straight_needed(self):
        """Check if we need to force a straight period"""
        if len(self.recent_steer_bursts) < self.max_steer_bursts:
            return False
        
        # Count steering bursts in recent history
        steer_burst_count = sum(self.recent_steer_bursts)
        
        # Force straight if too many recent bursts had steering
        return steer_burst_count >= self.max_steer_bursts
    
    def get_statistics(self):
        """Get current statistics"""
        if self.total_commands == 0:
            return {}
        
        steer_pct = (self.total_steer_commands / self.total_commands) * 100
        zero_pct = (self.total_zero_commands / self.total_commands) * 100
        forced_pct = (self.total_forced_straight / self.total_commands) * 100
        
        recent_steer_bursts = sum(self.recent_steer_bursts) if self.recent_steer_bursts else 0
        
        return {
            'total_commands': self.total_commands,
            'steer_commands': self.total_steer_commands,
            'zero_commands': self.total_zero_commands,
            'forced_straight': self.total_forced_straight,
            'steer_percentage': steer_pct,
            'zero_percentage': zero_pct,
            'forced_percentage': forced_pct,
            'burst_count': self.burst_count,
            'recent_steer_bursts': recent_steer_bursts,
            'forced_straight_remaining': self.forced_straight_remaining,
            'current_phase': 'steer' if self.in_steer_phase else 'zero',
            'burst_step': self.current_burst_step
        }
    
    def get_recent_commands(self, count=10):
        """Get recent commands from history"""
        recent = list(self.command_history)[-count:]
        return recent

class InferenceNode(Node):
    """ROS2 node for autonomous car control with burst steering control"""
    
    def __init__(self):
        super().__init__('car_inference_node')
        
        # Declare existing parameters
        self.declare_parameter('model_path', '/home/toon/train_results/angular_seq10_20250707_003809/checkpoints/best_checkpoint.pth')
        self.declare_parameter('sequence_length', 10)
        self.declare_parameter('publish_rate', 30.0)  # Hz - rate for publishing from queue
        self.declare_parameter('smoothing_alpha', 0.7)  # Smoothing factor for command smoothing
        self.declare_parameter('inference_rate', 5.0)  # Hz - rate for running inference
        self.declare_parameter('use_vertical_crop', False)
        self.declare_parameter('crop_pixels', 100)
        self.declare_parameter('max_angular_velocity', 3.0)  # rad/s
        self.declare_parameter('max_linear_velocity', 1.5)   # m/s
        self.declare_parameter('confidence_threshold', 0.1)
        self.declare_parameter('safety_timeout', 2.0)  # seconds
        self.declare_parameter('enable_autonomous', True)
        self.declare_parameter('device', 'cuda')
        
        # NEW: Declare burst control parameters
        self.declare_parameter('command_history_size', 30)     # Size of command history
        self.declare_parameter('burst_steer_count', 2)         # Steering commands per burst  
        self.declare_parameter('burst_zero_count', 3)          # Zero commands per burst
        self.declare_parameter('max_steer_bursts', 15)         # Max steering bursts before forcing straight
        self.declare_parameter('forced_straight_count', 15)    # Commands to force straight
        self.declare_parameter('steering_threshold', 0.1)      # Minimum angular to consider steering
        
        # NEW: Prediction lookahead parameter
        self.declare_parameter('prediction_skip_count', 0)     # Skip first N predictions (use future predictions)
        
        # Get all parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.sequence_length = self.get_parameter('sequence_length').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.inference_rate = self.get_parameter('inference_rate').get_parameter_value().double_value
        self.use_vertical_crop = self.get_parameter('use_vertical_crop').get_parameter_value().bool_value
        self.crop_pixels = self.get_parameter('crop_pixels').get_parameter_value().integer_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.safety_timeout = self.get_parameter('safety_timeout').get_parameter_value().double_value
        self.enable_autonomous = self.get_parameter('enable_autonomous').get_parameter_value().bool_value
        self.device_name = self.get_parameter('device').get_parameter_value().string_value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        
        # NEW: Get burst control parameters
        command_history_size = self.get_parameter('command_history_size').get_parameter_value().integer_value
        burst_steer_count = self.get_parameter('burst_steer_count').get_parameter_value().integer_value
        burst_zero_count = self.get_parameter('burst_zero_count').get_parameter_value().integer_value
        max_steer_bursts = self.get_parameter('max_steer_bursts').get_parameter_value().integer_value
        forced_straight_count = self.get_parameter('forced_straight_count').get_parameter_value().integer_value
        steering_threshold = self.get_parameter('steering_threshold').get_parameter_value().double_value
        
        # NEW: Get prediction lookahead parameter
        self.prediction_skip_count = self.get_parameter('prediction_skip_count').get_parameter_value().integer_value
        
        # Initialize components
        self.command_smoother = CommandSmoother(alpha=self.smoothing_alpha)
        
        # NEW: Initialize burst steering controller
        self.burst_controller = BurstSteeringController(
            command_history_size=command_history_size,
            burst_steer_count=burst_steer_count,
            burst_zero_count=burst_zero_count,
            max_steer_bursts=max_steer_bursts,
            forced_straight_count=forced_straight_count,
            steering_threshold=steering_threshold
        )

        # Initialize device
        if self.device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.get_logger().info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.get_logger().info("Using CPU device")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Initialize data transforms
        self.init_transforms()
        
        # Initialize state variables
        self.latest_image = None
        self.latest_image_time = None
        self.inference_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        
        # Prediction queue - stores angular velocity predictions to be published
        # Adjust queue size based on skip count
        effective_sequence_length = max(1, self.sequence_length - self.prediction_skip_count)
        self.prediction_queue = deque(maxlen=effective_sequence_length)
        self.queue_confidence = 0.0
        self.last_inference_time = time.time()
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.autonomous_enable_sub = self.create_subscription(
            Bool,
            '/car/autonomous/enable',
            self.autonomous_enable_callback,
            10
        )
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_autonomous', 10)
        self.angular_prediction_pub = self.create_publisher(Float32, '/car/angular_prediction', 10)
        self.confidence_pub = self.create_publisher(Float32, '/car/inference_confidence', 10)
        self.status_pub = self.create_publisher(String, '/car/inference_status', 10)
        self.queue_size_pub = self.create_publisher(Float32, '/car/queue_size', 10)
        
        # NEW: Burst control publishers
        self.burst_stats_pub = self.create_publisher(String, '/car/burst_stats', 10)
        
        # Create separate timers for inference and publishing
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate,
            self.inference_callback
        )
        
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_callback
        )
        
        # NEW: Timer for burst statistics
        self.burst_stats_timer = self.create_timer(5.0, self.publish_burst_stats)
        
        # Status monitoring
        self.inference_count = 0
        self.publish_count = 0
        self.error_count = 0
        
        self.get_logger().info("Car Inference Node initialized with burst steering control")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Sequence length: {self.sequence_length}")
        self.get_logger().info(f"Inference rate: {self.inference_rate} Hz")
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz")
        self.get_logger().info(f"🔮 Prediction lookahead: Skip first {self.prediction_skip_count} predictions")
        self.get_logger().info(f"📊 Effective queue size: {effective_sequence_length} (was {self.sequence_length})")
        self.get_logger().info(f"Burst control: {burst_steer_count} steer + {burst_zero_count} zero per burst")
        self.get_logger().info(f"Forced straight: {forced_straight_count} commands after {max_steer_bursts} steer bursts")
        self.get_logger().info(f"Command history: {command_history_size} commands")
        self.get_logger().info(f"Autonomous mode: {'ENABLED' if self.enable_autonomous else 'DISABLED'}")
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"Model file not found: {self.model_path}")
                return
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            self.model = ResNet18AngularRegressor(
                sequence_length=self.sequence_length,
                pretrained=False
            )
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.get_logger().info("Loaded model from checkpoint")
            else:
                self.model.load_state_dict(checkpoint)
                self.get_logger().info("Loaded model state dict directly")
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            
            # Print model info if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.get_logger().info(f"Model config: {json.dumps(config, indent=2)}")
            
            if 'best_val_loss' in checkpoint:
                self.get_logger().info(f"Model best validation loss: {checkpoint['best_val_loss']:.6f}")
            
            self.get_logger().info("Model loaded successfully!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def init_transforms(self):
        """Initialize image preprocessing transforms"""
        transforms_list = []
        
        # Add vertical crop if enabled
        if self.use_vertical_crop:
            def vertical_crop(img):
                h, w = img.shape[:2]
                return img[self.crop_pixels:h-self.crop_pixels, :]
            transforms_list.append(transforms.Lambda(vertical_crop))
        
        # Standard preprocessing
        transforms_list.extend([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.transform = transforms.Compose(transforms_list)
        
        self.get_logger().info(f"Image transforms initialized (crop: {self.use_vertical_crop})")
    
    def image_callback(self, msg):
        """Callback for incoming camera images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store latest image with timestamp
            with self.inference_lock:
                self.latest_image = cv_image
                self.latest_image_time = self.get_clock().now()
                
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    def autonomous_enable_callback(self, msg):
        """Callback for autonomous enable/disable"""
        self.enable_autonomous = msg.data
        status = "ENABLED" if self.enable_autonomous else "DISABLED"
        self.get_logger().info(f"Autonomous mode: {status}")
        
        # Clear queue when disabling autonomous mode
        if not self.enable_autonomous:
            with self.queue_lock:
                self.prediction_queue.clear()
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Autonomous mode: {status}"
        self.status_pub.publish(status_msg)
    
    def preprocess_image(self, cv_image):
        """Preprocess image for model inference"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            tensor_image = self.transform(rgb_image)
            
            # Add batch dimension
            tensor_image = tensor_image.unsqueeze(0)
            
            return tensor_image.to(self.device)
            
        except Exception as e:
            self.get_logger().error(f"Error preprocessing image: {e}")
            return None
    
    def predict_angular_velocity(self, image_tensor):
        """Predict angular velocity sequence from image"""
        try:
            with torch.no_grad():
                # Forward pass
                predictions = self.model(image_tensor)
                
                # Get predictions as numpy array
                pred_numpy = predictions.cpu().numpy()[0]  # Remove batch dimension
                
                return pred_numpy
                
        except Exception as e:
            self.get_logger().error(f"Error in model prediction: {e}")
            return None
    
    def calculate_confidence(self, predictions):
        """Calculate confidence score for predictions"""
        try:
            # Simple confidence based on prediction variance
            variance = np.var(predictions[:min(5, len(predictions))])  # Variance of first 5 predictions
            confidence = 1.0 / (1.0 + variance)  # Higher variance = lower confidence
            
            return np.clip(confidence, 0.0, 1.0)
            
        except:
            return 0.0
    
    def update_prediction_queue(self, new_predictions, confidence):
        """Update the prediction queue with future predictions (skipping first N predictions)"""
        with self.queue_lock:
            # Clear existing queue
            self.prediction_queue.clear()
            
            # STEP 1: Apply prediction lookahead - skip first N predictions
            if self.prediction_skip_count > 0:
                if len(new_predictions) > self.prediction_skip_count:
                    # Skip first N predictions, use the "future" predictions
                    future_predictions = new_predictions[self.prediction_skip_count:]
                    skipped_info = f"Skipped first {self.prediction_skip_count} predictions"
                else:
                    # Safety: if not enough predictions, use all but log warning
                    future_predictions = new_predictions
                    skipped_info = f"WARNING: Only {len(new_predictions)} predictions, using all"
            else:
                # No skipping, use all predictions
                future_predictions = new_predictions
                skipped_info = "No prediction skipping"
            
            # STEP 2: Apply safety limits to future predictions
            safe_predictions = np.clip(future_predictions, -self.max_angular_velocity, self.max_angular_velocity)
            
            # STEP 3: Add future predictions to queue
            for pred in safe_predictions:
                self.prediction_queue.append(float(pred))
            
            # STEP 4: Update confidence
            self.queue_confidence = confidence
            
            # STEP 5: Log the lookahead effect
            self.get_logger().debug(
                f"Queue updated: {skipped_info}, "
                f"using {len(safe_predictions)} future predictions, "
                f"confidence: {confidence:.3f}"
            )
            
            # STEP 6: Detailed debug logging every few updates
            if hasattr(self, 'inference_count') and self.inference_count % 10 == 0:
                if self.prediction_skip_count > 0 and len(new_predictions) > self.prediction_skip_count:
                    skipped_preds = new_predictions[:self.prediction_skip_count]
                    used_preds = new_predictions[self.prediction_skip_count:self.prediction_skip_count+3]  # Show first 3 used
                    self.get_logger().info(
                        f"🔮 Lookahead: Skipped {skipped_preds[:3]} → Using {used_preds} "
                        f"(queue size: {len(self.prediction_queue)})"
                    )
    
    def get_next_prediction(self):
        """Get the next prediction from the queue"""
        with self.queue_lock:
            if len(self.prediction_queue) > 0:
                return self.prediction_queue.popleft(), self.queue_confidence
            else:
                return None, 0.0
    
    def inference_callback(self):
        """Inference callback - runs at inference_rate to generate new predictions"""
        if not self.model_loaded or not self.enable_autonomous:
            return
        
        # Start timing the inference
        inference_start_time = time.time()
        
        try:
            # Check if we have a recent image
            with self.inference_lock:
                if self.latest_image is None:
                    return
                
                # Check image age
                current_time = self.get_clock().now()
                if self.latest_image_time is None:
                    return
                
                image_age = (current_time - self.latest_image_time).nanoseconds / 1e9
                if image_age > self.safety_timeout:
                    self.get_logger().warn(f"Image too old: {image_age:.2f}s")
                    return
                
                # Copy image for processing
                image_to_process = self.latest_image.copy()
            
            # Time the preprocessing step
            preprocess_start = time.time()
            image_tensor = self.preprocess_image(image_to_process)
            if image_tensor is None:
                return
            preprocess_time = time.time() - preprocess_start
            
            # Time the model prediction step
            prediction_start = time.time()
            predictions = self.predict_angular_velocity(image_tensor)
            if predictions is None:
                self.error_count += 1
                return
            prediction_time = time.time() - prediction_start
            
            # Calculate confidence
            confidence = self.calculate_confidence(predictions)
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                self.get_logger().warn(f"Low confidence: {confidence:.3f} < {self.confidence_threshold}")
                return
            
            # Update prediction queue with raw model predictions
            self.update_prediction_queue(predictions, confidence)
            
            self.inference_count += 1
            
            # Calculate total inference time
            total_inference_time = time.time() - inference_start_time
            
            # Log timing information every inference (you can adjust frequency)
            self.get_logger().info(
                f"Inference #{self.inference_count}: "
                f"total_time={total_inference_time*1000:.1f}ms, "
                f"preprocess={preprocess_time*1000:.1f}ms, "
                f"prediction={prediction_time*1000:.1f}ms, "
                f"confidence={confidence:.3f}"
            )
            
            # Log more detailed periodic status
            if self.inference_count % 20 == 0:  # Log every 20 inferences
                with self.queue_lock:
                    queue_size = len(self.prediction_queue)
                self.get_logger().info(
                    f"=== Inference Stats #{self.inference_count} === "
                    f"avg_inference_time={total_inference_time*1000:.1f}ms, "
                    f"confidence={confidence:.3f}, queue_size={queue_size}, errors={self.error_count}"
                )
            
        except Exception as e:
            total_inference_time = time.time() - inference_start_time
            self.get_logger().error(f"Error in inference callback after {total_inference_time*1000:.1f}ms: {e}")
            self.error_count += 1
    
    def publish_callback(self):
        """Publishing callback - runs at publish_rate to publish from queue"""
        if not self.enable_autonomous:
            return
        
        try:
            # Get next prediction from queue
            angular_velocity, confidence = self.get_next_prediction()
            
            if angular_velocity is None:
                # No predictions available, publish stop command with burst control
                controlled_linear, controlled_angular = self.burst_controller.process(0.0, 0.0)
                self.publish_command(controlled_linear, controlled_angular, confidence)
                return
            
            # Apply burst control to the prediction
            controlled_linear, controlled_angular = self.burst_controller.process(
                self.max_linear_velocity, angular_velocity)
            
            # Apply smoothing
            smooth_linear, smooth_angular = self.command_smoother.smooth(controlled_linear, controlled_angular)

            # Publish the command
            self.publish_command(smooth_linear, smooth_angular, confidence)
            
            self.publish_count += 1
            
            # Log periodic status
            if self.publish_count % 100 == 0:  # Log every 100 publishes
                stats = self.burst_controller.get_statistics()
                self.get_logger().info(
                    f"Publish #{self.publish_count}: final_angular={smooth_angular:.3f}, "
                    f"original_angular={angular_velocity:.3f}, confidence={confidence:.3f}, "
                    f"burst_phase={stats['current_phase']}, forced_remaining={stats['forced_straight_remaining']}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error in publish callback: {e}")
            self.publish_stop_command()
    
    def publish_command(self, linear, angular, confidence):
        """Publish command and related topics"""
        # Create and publish control command
        cmd_msg = Twist()
        cmd_msg.linear.x = linear
        cmd_msg.angular.z = angular
        
        self.cmd_vel_pub.publish(cmd_msg)
        
        # Publish additional info
        angular_msg = Float32()
        angular_msg.data = angular
        self.angular_prediction_pub.publish(angular_msg)
        
        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)
        
        # Publish queue size
        with self.queue_lock:
            queue_size = len(self.prediction_queue)
        queue_size_msg = Float32()
        queue_size_msg.data = float(queue_size)
        self.queue_size_pub.publish(queue_size_msg)
    
    def publish_stop_command(self):
        """Publish stop command for safety"""
        controlled_linear, controlled_angular = self.burst_controller.process(0.0, 0.0)
        self.publish_command(controlled_linear, controlled_angular, 0.0)
        
        # Publish status
        status_msg = String()
        status_msg.data = "STOPPED - No predictions available"
        self.status_pub.publish(status_msg)
    
    def publish_burst_stats(self):
        """Publish burst control statistics"""
        try:
            stats = self.burst_controller.get_statistics()
            
            stats_str = (
                f"Burst Stats: {stats['total_commands']} total, "
                f"{stats['steer_percentage']:.1f}% steer, "
                f"{stats['zero_percentage']:.1f}% zero, "
                f"{stats['forced_percentage']:.1f}% forced straight, "
                f"Bursts: {stats['burst_count']}, "
                f"Recent steer bursts: {stats['recent_steer_bursts']}, "
                f"Phase: {stats['current_phase']}, "
                f"Step: {stats['burst_step']}, "
                f"Forced remaining: {stats['forced_straight_remaining']}"
            )
            
            # Publish stats
            stats_msg = String()
            stats_msg.data = stats_str
            self.burst_stats_pub.publish(stats_msg)
            
            # Log periodically
            self.get_logger().info(f"📊 {stats_str}")
            
            # Log recent commands for debugging
            recent_commands = self.burst_controller.get_recent_commands(10)
            if recent_commands:
                cmd_types = [cmd['type'] for cmd in recent_commands]
                cmd_angulars = [f"{cmd['angular']:.2f}" for cmd in recent_commands]
                self.get_logger().debug(f"Recent commands: {cmd_types}")
                self.get_logger().debug(f"Recent angulars: {cmd_angulars}")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing burst stats: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = InferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in inference node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()