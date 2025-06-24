#!/usr/bin/env python3
"""
Fixed Camera Node for Jetson Xavier NX with CSI Camera
Addresses resolution and pipeline issues
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np

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
        
        # Parameters - using camera's native resolution
        self.declare_parameter('camera_width', 1280)  # Match camera's native resolution
        self.declare_parameter('camera_height', 720)   # Match camera's native resolution
        self.declare_parameter('output_width', 1280)    # Desired output resolution
        self.declare_parameter('output_height', 720)   # Desired output resolution
        self.declare_parameter('framerate', 60)
        self.declare_parameter('flip_method', 2)  # 180 degree flip
        self.declare_parameter('camera_id', 0)
        
        self.camera_width = self.get_parameter('camera_width').value
        self.camera_height = self.get_parameter('camera_height').value
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value
        self.framerate = self.get_parameter('framerate').value
        self.flip_method = self.get_parameter('flip_method').value
        self.camera_id = self.get_parameter('camera_id').value
        
        # Build GStreamer pipeline for CSI camera
        self.gst_pipeline = self.build_gstreamer_pipeline()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera with GStreamer pipeline')
            self.get_logger().error(f'Pipeline: {self.gst_pipeline}')
            # Try simplified pipeline
            self.get_logger().info('Trying simplified pipeline...')
            self.gst_pipeline = self.build_simple_pipeline()
            self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                # Try fallback to regular camera
                self.get_logger().info('Trying fallback to regular camera...')
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    self.get_logger().error('Failed to open any camera')
                    return
                else:
                    self.get_logger().info('Using regular camera (not CSI)')
            else:
                self.get_logger().info('Successfully opened CSI camera with simplified pipeline')
        else:
            self.get_logger().info('Successfully opened CSI camera')
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Timer for capturing frames
        timer_period = 1.0 / self.framerate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Camera Node Started')
        self.get_logger().info(f'Camera Resolution: {self.camera_width}x{self.camera_height}')
        self.get_logger().info(f'Output Resolution: {self.output_width}x{self.output_height} @ {self.framerate}fps')
        self.get_logger().info(f'Flip method: {self.flip_method}')
        self.get_logger().info(f'Publishing to: /camera/image_raw')
    
    def build_gstreamer_pipeline(self):
        """Build GStreamer pipeline for CSI camera"""
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.camera_width}, height=(int){self.camera_height}, "
            f"format=(string)NV12, framerate=(fraction)30/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.output_width}, height=(int){self.output_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        )
        return pipeline
    
    def build_simple_pipeline(self):
        """Build simplified GStreamer pipeline"""
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            f"video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! "
            f"appsink max-buffers=1 drop=true"
        )
        return pipeline
    
    def timer_callback(self):
        """Capture and publish camera frames"""
        ret, frame = self.cap.read()
        
        if ret:
            try:
                # Resize frame if needed
                if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
                    frame = cv2.resize(frame, (self.output_width, self.output_height))
                
                # Apply manual flip if using regular camera (not CSI)
                if self.cap.get(cv2.CAP_PROP_BACKEND) != cv2.CAP_GSTREAMER:
                    if self.flip_method == 2:  # 180 degree flip
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                
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
    
    def destroy_node(self):
        """Clean up camera resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info('Camera released')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Camera node interrupted')
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()