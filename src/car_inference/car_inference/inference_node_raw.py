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

class InferenceNode(Node):
    """ROS2 node for autonomous car control using trained neural network with queue-based publishing"""
    
    def __init__(self):
        super().__init__('car_inference_node')
        
        # Declare parameters
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
        
        # Get parameters
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
        self.command_smoother = CommandSmoother(alpha=self.smoothing_alpha)

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
        self.prediction_queue = deque(maxlen=self.sequence_length)
        self.queue_confidence = 0.0
        self.last_inference_time = time.time()
        
        # Removed predictions_history since we're not smoothing between inference runs
        
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
        
        # Create separate timers for inference and publishing
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate,
            self.inference_callback
        )
        
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_callback
        )
        
        # Status monitoring
        self.inference_count = 0
        self.publish_count = 0
        self.error_count = 0
        
        self.get_logger().info("Car Inference Node initialized with queue-based publishing")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Sequence length: {self.sequence_length}")
        self.get_logger().info(f"Inference rate: {self.inference_rate} Hz")
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz")
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
        """Update the prediction queue with new predictions, overriding what's left"""
        with self.queue_lock:
            # Clear existing queue and add new predictions
            self.prediction_queue.clear()
            
            # Apply safety limits to all predictions
            safe_predictions = np.clip(new_predictions, -self.max_angular_velocity, self.max_angular_velocity)
            
            # Add all predictions to queue
            for pred in safe_predictions:
                self.prediction_queue.append(float(pred))
            
            # Update confidence
            self.queue_confidence = confidence
            
            self.get_logger().debug(f"Queue updated with {len(self.prediction_queue)} predictions, confidence: {confidence:.3f}")
    
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
                # No predictions available, publish stop command
                self.publish_stop_command()
                return
            
            # Apply smoothing
            smooth_linear, smooth_angular = self.command_smoother.smooth(self.max_linear_velocity, angular_velocity)

            # Create and publish control command
            cmd_msg = Twist()
            cmd_msg.linear.x = smooth_linear  # Constant forward speed
            cmd_msg.angular.z = smooth_angular
            
            self.cmd_vel_pub.publish(cmd_msg)
            
            # Publish additional info
            angular_msg = Float32()
            angular_msg.data = smooth_angular
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
            
            self.publish_count += 1
            
            # Log periodic status
            if self.publish_count % 100 == 0:  # Log every 100 publishes (5 seconds at 20Hz)
                self.get_logger().info(
                    f"Publish #{self.publish_count}: angular_z={smooth_angular:.3f}, "
                    f"confidence={confidence:.3f}, queue_size={queue_size}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error in publish callback: {e}")
            self.publish_stop_command()
    
    def publish_stop_command(self):
        """Publish stop command for safety"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = "STOPPED - No predictions available"
        self.status_pub.publish(status_msg)


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