#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool, Trigger
import torch
import os
import json
import threading
from datetime import datetime


class ModelManager(Node):
    """Model management node for switching between different trained models"""
    
    def __init__(self):
        super().__init__('model_manager')
        
        # Declare parameters
        self.declare_parameter('models_directory', '/home/anton/jetson/models')
        self.declare_parameter('default_model', 'latest_model.pth')
        self.declare_parameter('auto_load_latest', True)
        self.declare_parameter('model_validation', True)
        
        # Get parameters
        self.models_directory = self.get_parameter('models_directory').get_parameter_value().string_value
        self.default_model = self.get_parameter('default_model').get_parameter_value().string_value
        self.auto_load_latest = self.get_parameter('auto_load_latest').get_parameter_value().bool_value
        self.model_validation = self.get_parameter('model_validation').get_parameter_value().bool_value
        
        # Initialize state
        self.current_model_path = None
        self.current_model_info = {}
        self.available_models = {}
        self.model_lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create publishers
        self.model_status_pub = self.create_publisher(String, '/car/model_status', 10)
        self.model_list_pub = self.create_publisher(String, '/car/available_models', 10)
        
        # Create services
        self.load_model_srv = self.create_service(
            SetBool,
            '/car/load_model',
            self.load_model_callback
        )
        
        self.scan_models_srv = self.create_service(
            Trigger,
            '/car/scan_models',
            self.scan_models_callback
        )
        
        self.get_model_info_srv = self.create_service(
            Trigger,
            '/car/get_model_info',
            self.get_model_info_callback
        )
        
        # Create timer for periodic status updates
        self.status_timer = self.create_timer(5.0, self.publish_status)
        
        # Initialize model scanning
        self.scan_available_models()
        
        # Auto-load default model if enabled
        if self.auto_load_latest:
            self.auto_load_best_model()
        
        self.get_logger().info("Model Manager initialized")
        self.get_logger().info(f"Models directory: {self.models_directory}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Found {len(self.available_models)} models")
    
    def scan_available_models(self):
        """Scan the models directory for available models"""
        self.available_models = {}
        
        if not os.path.exists(self.models_directory):
            self.get_logger().warn(f"Models directory does not exist: {self.models_directory}")
            return
        
        for root, dirs, files in os.walk(self.models_directory):
            for file in files:
                if file.endswith('.pth'):
                    model_path = os.path.join(root, file)
                    model_info = self.get_model_metadata(model_path)
                    
                    if model_info:
                        relative_path = os.path.relpath(model_path, self.models_directory)
                        self.available_models[relative_path] = model_info
        
        self.get_logger().info(f"Scanned models directory: found {len(self.available_models)} models")
        
        # Publish updated model list
        self.publish_model_list()
    
    def get_model_metadata(self, model_path):
        """Extract metadata from a model file"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            metadata = {
                'path': model_path,
                'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
            }
            
            # Extract training information if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                metadata.update({
                    'sequence_length': config.get('sequence_length', 'unknown'),
                    'experiment_name': config.get('experiment_name', 'unknown'),
                    'model_architecture': config.get('model_architecture', 'unknown'),
                    'training_epochs': config.get('num_epochs', 'unknown'),
                })
            
            if 'best_val_loss' in checkpoint:
                metadata['best_val_loss'] = float(checkpoint['best_val_loss'])
            
            if 'final_val_loss' in checkpoint:
                metadata['final_val_loss'] = float(checkpoint['final_val_loss'])
            
            # Validate model structure if enabled
            if self.model_validation:
                metadata['valid'] = self.validate_model_structure(checkpoint)
            else:
                metadata['valid'] = True
            
            return metadata
            
        except Exception as e:
            self.get_logger().warn(f"Could not read model metadata from {model_path}: {e}")
            return None
    
    def validate_model_structure(self, checkpoint):
        """Validate that the model has the expected structure"""
        try:
            # Check for required keys
            required_keys = ['model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    return False
            
            # Check model state dict structure
            state_dict = checkpoint['model_state_dict']
            
            # Verify it looks like a ResNet18 model
            expected_patterns = [
                'resnet18.fc.',  # Our custom head
                'resnet18.layer',  # ResNet layers
                'resnet18.conv1',  # First conv layer
            ]
            
            state_keys = list(state_dict.keys())
            for pattern in expected_patterns:
                if not any(pattern in key for key in state_keys):
                    self.get_logger().warn(f"Model validation failed: missing pattern {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            self.get_logger().warn(f"Model validation error: {e}")
            return False
    
    def auto_load_best_model(self):
        """Automatically load the best available model"""
        if not self.available_models:
            self.get_logger().warn("No models available for auto-loading")
            return
        
        # Find the model with lowest validation loss
        best_model = None
        best_loss = float('inf')
        
        for model_name, model_info in self.available_models.items():
            if not model_info.get('valid', False):
                continue
                
            val_loss = model_info.get('best_val_loss', model_info.get('final_val_loss', float('inf')))
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model_name
        
        if best_model:
            model_path = os.path.join(self.models_directory, best_model)
            self.load_model(model_path)
            self.get_logger().info(f"Auto-loaded best model: {best_model} (val_loss: {best_loss:.6f})")
        else:
            self.get_logger().warn("No valid models found for auto-loading")
    
    def load_model(self, model_path):
        """Load a specific model"""
        try:
            with self.model_lock:
                # Verify file exists
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Load and validate checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if 'model_state_dict' not in checkpoint:
                    raise ValueError("Invalid model file: missing model_state_dict")
                
                # Update current model info
                self.current_model_path = model_path
                self.current_model_info = self.get_model_metadata(model_path) or {}
                
                self.get_logger().info(f"Model loaded successfully: {os.path.basename(model_path)}")
                
                # Publish updated status
                self.publish_status()
                
                return True
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model {model_path}: {e}")
            return False
    
    def load_model_callback(self, request, response):
        """Service callback for loading a model"""
        # Extract model name from request (using request.data as string)
        model_name = request.data  # Assuming the client sends model name as data
        
        if model_name in self.available_models:
            model_path = os.path.join(self.models_directory, model_name)
            success = self.load_model(model_path)
            response.success = success
            response.message = f"Model {model_name} {'loaded' if success else 'failed to load'}"
        else:
            response.success = False
            response.message = f"Model {model_name} not found in available models"
        
        return response
    
    def scan_models_callback(self, request, response):
        """Service callback for scanning models directory"""
        try:
            old_count = len(self.available_models)
            self.scan_available_models()
            new_count = len(self.available_models)
            
            response.success = True
            response.message = f"Models scanned: found {new_count} models ({new_count - old_count:+d} change)"
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to scan models: {e}"
        
        return response
    
    def get_model_info_callback(self, request, response):
        """Service callback for getting current model information"""
        try:
            with self.model_lock:
                if self.current_model_path:
                    info_str = json.dumps(self.current_model_info, indent=2)
                    response.success = True
                    response.message = f"Current model: {os.path.basename(self.current_model_path)}\n{info_str}"
                else:
                    response.success = False
                    response.message = "No model currently loaded"
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to get model info: {e}"
        
        return response
    
    def publish_status(self):
        """Publish current model status"""
        try:
            with self.model_lock:
                if self.current_model_path:
                    model_name = os.path.basename(self.current_model_path)
                    val_loss = self.current_model_info.get('best_val_loss', 'unknown')
                    sequence_length = self.current_model_info.get('sequence_length', 'unknown')
                    
                    status = f"LOADED: {model_name} | Val Loss: {val_loss} | Seq Len: {sequence_length}"
                else:
                    status = "NO MODEL LOADED"
                
                status_msg = String()
                status_msg.data = status
                self.model_status_pub.publish(status_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing model status: {e}")
    
    def publish_model_list(self):
        """Publish list of available models"""
        try:
            model_list = []
            for model_name, model_info in self.available_models.items():
                val_loss = model_info.get('best_val_loss', 'unknown')
                valid = model_info.get('valid', False)
                status = "✓" if valid else "✗"
                model_list.append(f"{status} {model_name} (val_loss: {val_loss})")
            
            list_msg = String()
            list_msg.data = "\n".join(model_list) if model_list else "No models found"
            self.model_list_pub.publish(list_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing model list: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ModelManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in model manager: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()