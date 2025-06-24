#!/usr/bin/env python3
"""
Simple Image Viewer Node
Subscribes to camera images and displays them using OpenCV
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
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding='bgr8'):
        """Convert ROS Image message to OpenCV image manually"""
        if img_msg.encoding == 'bgr8':
            dtype = np.uint8
            channels = 3
        elif img_msg.encoding == 'rgb8':
            dtype = np.uint8
            channels = 3
        elif img_msg.encoding == 'mono8':
            dtype = np.uint8
            channels = 1
        else:
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
        
        # Convert bytes to numpy array
        cv_image = np.frombuffer(img_msg.data, dtype=dtype)
        
        if channels == 1:
            cv_image = cv_image.reshape((img_msg.height, img_msg.width))
        else:
            cv_image = cv_image.reshape((img_msg.height, img_msg.width, channels))
        
        # Convert RGB to BGR if needed
        if img_msg.encoding == 'rgb8' and desired_encoding == 'bgr8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        return cv_image

class ImageViewerNode(Node):
    def __init__(self):
        super().__init__('image_viewer_node')
        
        # Use cv_bridge if available, otherwise use manual implementation
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
            self.get_logger().info('Using cv_bridge')
        else:
            self.bridge = CvBridgeManual()
            self.get_logger().info('Using manual cv_bridge implementation')
        
        # Subscribe to camera images
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info('Image Viewer Node Started')
        self.get_logger().info('Subscribing to: /camera/image_raw')
        self.get_logger().info('Press ESC or Ctrl+C to exit')
        
        # Frame counter for FPS calculation
        self.frame_count = 0
        self.last_time = self.get_clock().now()
    
    def image_callback(self, msg):
        """Process received camera images"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Calculate FPS
            self.frame_count += 1
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_time).nanoseconds / 1e9
            
            if time_diff >= 1.0:  # Update FPS every second
                fps = self.frame_count / time_diff
                self.get_logger().info(f'Receiving images at {fps:.1f} FPS - Resolution: {cv_image.shape[1]}x{cv_image.shape[0]}')
                self.frame_count = 0
                self.last_time = current_time
            
            # Add FPS text to image
            fps_text = f'FPS: {self.frame_count/max(time_diff, 0.001):.1f}'
            cv2.putText(cv_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add resolution text
            res_text = f'Resolution: {cv_image.shape[1]}x{cv_image.shape[0]}'
            cv2.putText(cv_image, res_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow('Camera Feed', cv_image)
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                self.get_logger().info('ESC pressed, shutting down...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def destroy_node(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        image_viewer_node = ImageViewerNode()
        rclpy.spin(image_viewer_node)
    except KeyboardInterrupt:
        print('\nImage viewer interrupted')
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()