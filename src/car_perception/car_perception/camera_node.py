#!/usr/bin/env python3
"""
Robust Camera Node for Jetson Xavier NX with CSI Camera
Handles ROS-specific timing and resource issues
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import time
import subprocess
import os

# Try importing cv_bridge with fallback
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except (ImportError, SystemError) as e:
    print(f"cv_bridge import failed: {e}")
    print("Using manual conversion instead")
    CV_BRIDGE_AVAILABLE = False

class CvBridgeManual:
    """Manual implementation of cv_bridge functionality"""
    
    def cv2_to_imgmsg(self, cv_image, encoding='bgr8'):
        """Convert OpenCV image to ROS Image message manually"""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
        img_msg.data = cv_image.tobytes()
        return img_msg

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Use cv_bridge if available, otherwise use manual implementation
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
            self.get_logger().info('Using cv_bridge')
        else:
            self.bridge = CvBridgeManual()
            self.get_logger().info('Using manual cv_bridge implementation')
        
        # Publishers
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # Parameters
        self.declare_parameter('output_width', 640)
        self.declare_parameter('output_height', 480)
        self.declare_parameter('framerate', 30)
        
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value
        self.framerate = self.get_parameter('framerate').value
        
        # Initialize camera with retry logic
        self.cap = None
        self.initialize_camera()
        
        if self.cap is None:
            self.get_logger().error('Failed to initialize camera after all attempts')
            return
        
        # Timer for capturing frames
        timer_period = 1.0 / self.framerate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Camera Node Started Successfully')
        self.get_logger().info(f'Publishing to /camera/image_raw at {self.framerate}fps')
        self.get_logger().info(f'Output resolution: {self.output_width}x{self.output_height}')
    
    def check_camera_availability(self):
        """Check if camera is available and kill any processes using it"""
        try:
            # Check for processes using the camera
            result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True)
            if result.returncode == 0:
                self.get_logger().warn('Camera is being used by another process')
                self.get_logger().info(f'Processes using camera: {result.stdout}')
            
            # Check nvargus daemon status
            result = subprocess.run(['pgrep', '-f', 'nvargus'], capture_output=True, text=True)
            if result.returncode != 0:
                self.get_logger().warn('nvargus daemon not running, trying to restart...')
                subprocess.run(['sudo', 'systemctl', 'restart', 'nvargus-daemon'], 
                             capture_output=True, text=True)
                time.sleep(2)  # Wait for daemon to start
                
        except Exception as e:
            self.get_logger().debug(f'Camera availability check failed: {e}')
    
    def initialize_camera(self):
        """Initialize camera with robust retry logic"""
        self.get_logger().info('Initializing camera...')
        
        # Check camera availability first
        self.check_camera_availability()
        
        # Wait a moment for any previous camera processes to release resources
        time.sleep(1)
        
        # Try the exact working pipeline from diagnostic with retries
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink max-buffers=1 drop=true"
        )
        
        # Try multiple times with increasing delays
        for attempt in range(5):
            self.get_logger().info(f'Camera initialization attempt {attempt + 1}/5')
            
            try:
                # Add delay between attempts
                if attempt > 0:
                    delay = min(attempt * 2, 5)  # 0, 2, 4, 5, 5 seconds
                    self.get_logger().info(f'Waiting {delay} seconds before retry...')
                    time.sleep(delay)
                
                self.get_logger().info('Opening CSI camera...')
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if cap.isOpened():
                    self.get_logger().info('Camera opened, testing capture...')
                    
                    # Test capture with timeout
                    for test_attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.get_logger().info(f'✓ Camera working! Frame shape: {frame.shape}')
                            
                            # Set buffer size to reduce latency
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            self.cap = cap
                            return
                        else:
                            self.get_logger().warn(f'Frame capture test {test_attempt + 1}/3 failed')
                            time.sleep(0.5)
                    
                    # If we get here, capture failed
                    self.get_logger().warn('Camera opened but frame capture failed')
                    cap.release()
                else:
                    self.get_logger().warn(f'Failed to open camera on attempt {attempt + 1}')
                    
            except Exception as e:
                self.get_logger().error(f'Exception on attempt {attempt + 1}: {e}')
            
            # Clean up any partial initialization
            try:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
            except:
                pass
        
        self.get_logger().error('All camera initialization attempts failed')
        self.cap = None
    
    def timer_callback(self):
        """Capture and publish camera frames"""
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            try:
                # Resize frame if needed
                if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
                    frame = cv2.resize(frame, (self.output_width, self.output_height))
                
                # Convert OpenCV image to ROS Image message
                msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_link'
                
                # Publish the image
                self.image_publisher.publish(msg)
                
            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')
        else:
            self.get_logger().warn('Failed to capture frame from camera')
            
            # Try to reinitialize camera if capture fails consistently
            if not hasattr(self, '_failed_captures'):
                self._failed_captures = 0
            self._failed_captures += 1
            
            if self._failed_captures > 10:
                self.get_logger().warn('Too many failed captures, trying to reinitialize camera...')
                self.destroy_camera()
                time.sleep(2)
                self.initialize_camera()
                self._failed_captures = 0
    
    def destroy_camera(self):
        """Clean up camera resources"""
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.get_logger().info('Camera released')
    
    def destroy_node(self):
        """Clean up camera resources"""
        self.destroy_camera()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        camera_node = CameraNode()
        if camera_node.cap is not None:
            rclpy.spin(camera_node)
        else:
            camera_node.get_logger().error('Camera initialization failed, exiting')
    except KeyboardInterrupt:
        camera_node.get_logger().info('Camera node interrupted')
    except Exception as e:
        print(f"Failed to start camera node: {e}")
    finally:
        if 'camera_node' in locals():
            camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()