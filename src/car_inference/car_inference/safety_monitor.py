#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String
from sensor_msgs.msg import Image
import time
import threading


class SafetyMonitor(Node):
    """Safety monitor for autonomous car operation"""
    
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Declare parameters
        self.declare_parameter('max_angular_velocity', 1.0)  # rad/s
        self.declare_parameter('max_linear_velocity', 0.5)   # m/s
        self.declare_parameter('min_confidence', 0.1)
        self.declare_parameter('max_no_image_time', 2.0)     # seconds
        self.declare_parameter('max_no_inference_time', 1.0) # seconds
        self.declare_parameter('emergency_stop_enabled', True)
        
        # Get parameters
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.max_no_image_time = self.get_parameter('max_no_image_time').get_parameter_value().double_value
        self.max_no_inference_time = self.get_parameter('max_no_inference_time').get_parameter_value().double_value
        self.emergency_stop_enabled = self.get_parameter('emergency_stop_enabled').get_parameter_value().bool_value
        
        # Initialize state variables
        self.last_image_time = None
        self.last_inference_time = None
        self.last_confidence = 0.0
        self.autonomous_enabled = False
        self.safety_lock = threading.Lock()
        self.violation_count = 0
        self.total_commands = 0
        
        # Safety flags
        self.image_timeout = False
        self.inference_timeout = False
        self.low_confidence = False
        self.velocity_violation = False
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.autonomous_cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel_autonomous',
            self.autonomous_cmd_callback,
            10
        )
        
        self.confidence_sub = self.create_subscription(
            Float32,
            '/car/inference_confidence',
            self.confidence_callback,
            10
        )
        
        self.autonomous_enable_sub = self.create_subscription(
            Bool,
            '/car/autonomous/enable',
            self.autonomous_enable_callback,
            10
        )
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.safety_status_pub = self.create_publisher(String, '/car/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/car/emergency_stop', 10)
        self.safety_override_pub = self.create_publisher(Bool, '/car/safety_override', 10)
        
        # Create timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_monitor_callback)  # 10 Hz
        
        self.get_logger().info("Safety Monitor initialized")
        self.get_logger().info(f"Max angular velocity: {self.max_angular_velocity} rad/s")
        self.get_logger().info(f"Max linear velocity: {self.max_linear_velocity} m/s")
        self.get_logger().info(f"Min confidence: {self.min_confidence}")
        self.get_logger().info(f"Emergency stop enabled: {self.emergency_stop_enabled}")
    
    def image_callback(self, msg):
        """Track last image reception time"""
        with self.safety_lock:
            self.last_image_time = time.time()
            self.image_timeout = False
    
    def autonomous_cmd_callback(self, msg):
        """Monitor and potentially override autonomous commands"""
        with self.safety_lock:
            self.last_inference_time = time.time()
            self.inference_timeout = False
            self.total_commands += 1
            
            # Check velocity limits
            linear_violation = abs(msg.linear.x) > self.max_linear_velocity
            angular_violation = abs(msg.angular.z) > self.max_angular_velocity
            self.velocity_violation = linear_violation or angular_violation
            
            # Create output command
            output_cmd = Twist()
            
            # Apply safety filtering if autonomous is enabled
            if self.autonomous_enabled and not self.is_safe():
                # Safety violation - stop the car
                output_cmd.linear.x = 0.0
                output_cmd.angular.z = 0.0
                self.violation_count += 1
                
                self.get_logger().warn(f"Safety violation detected - stopping car (#{self.violation_count})")
                self.publish_emergency_stop(True)
                
            elif self.autonomous_enabled:
                # Safe to proceed - apply velocity limits as final safety check
                output_cmd.linear.x = max(-self.max_linear_velocity, 
                                        min(self.max_linear_velocity, msg.linear.x))
                output_cmd.angular.z = max(-self.max_angular_velocity, 
                                         min(self.max_angular_velocity, msg.angular.z))
                
                # If we had to limit velocities, log it
                if (abs(output_cmd.linear.x - msg.linear.x) > 0.001 or 
                    abs(output_cmd.angular.z - msg.angular.z) > 0.001):
                    self.get_logger().warn(
                        f"Velocity limited: linear {msg.linear.x:.3f}->{output_cmd.linear.x:.3f}, "
                        f"angular {msg.angular.z:.3f}->{output_cmd.angular.z:.3f}"
                    )
                
                self.publish_emergency_stop(False)
            
            else:
                # Autonomous not enabled - stop
                output_cmd.linear.x = 0.0
                output_cmd.angular.z = 0.0
            
            # Publish the filtered command
            self.cmd_vel_pub.publish(output_cmd)
    
    def confidence_callback(self, msg):
        """Track confidence levels"""
        with self.safety_lock:
            self.last_confidence = msg.data
            self.low_confidence = msg.data < self.min_confidence
    
    def autonomous_enable_callback(self, msg):
        """Track autonomous enable status"""
        with self.safety_lock:
            self.autonomous_enabled = msg.data
            
        if msg.data:
            self.get_logger().info("Autonomous mode ENABLED - Safety monitoring active")
        else:
            self.get_logger().info("Autonomous mode DISABLED - All commands stopped")
            self.publish_stop_command()
    
    def safety_monitor_callback(self):
        """Periodic safety monitoring"""
        current_time = time.time()
        
        with self.safety_lock:
            # Check image timeout
            if self.last_image_time is not None:
                image_age = current_time - self.last_image_time
                self.image_timeout = image_age > self.max_no_image_time
            else:
                self.image_timeout = True
            
            # Check inference timeout
            if self.last_inference_time is not None:
                inference_age = current_time - self.last_inference_time
                self.inference_timeout = inference_age > self.max_no_inference_time
            else:
                self.inference_timeout = True
            
            # Publish safety status
            self.publish_safety_status()
            
            # Emergency stop if not safe and autonomous is enabled
            if self.autonomous_enabled and not self.is_safe():
                if self.emergency_stop_enabled:
                    self.publish_stop_command()
                    self.publish_emergency_stop(True)
    
    def is_safe(self):
        """Check if current state is safe for autonomous operation"""
        # List of safety conditions
        safety_checks = [
            (not self.image_timeout, "Image timeout"),
            (not self.inference_timeout, "Inference timeout"),
            (not self.low_confidence, f"Low confidence ({self.last_confidence:.3f})"),
            (not self.velocity_violation, "Velocity violation"),
        ]
        
        # Log any violations (but only occasionally to avoid spam)
        if self.total_commands % 50 == 0:  # Every 50 commands
            violations = [msg for safe, msg in safety_checks if not safe]
            if violations:
                self.get_logger().warn(f"Safety violations: {', '.join(violations)}")
        
        return all(safe for safe, _ in safety_checks)
    
    def publish_safety_status(self):
        """Publish current safety status"""
        status_parts = []
        
        if self.autonomous_enabled:
            status_parts.append("AUTO")
        else:
            status_parts.append("MANUAL")
        
        if self.is_safe():
            status_parts.append("SAFE")
        else:
            status_parts.append("UNSAFE")
            
            # Add specific violations
            violations = []
            if self.image_timeout:
                violations.append("IMG_TIMEOUT")
            if self.inference_timeout:
                violations.append("INF_TIMEOUT")
            if self.low_confidence:
                violations.append(f"LOW_CONF({self.last_confidence:.2f})")
            if self.velocity_violation:
                violations.append("VEL_VIOLATION")
            
            status_parts.extend(violations)
        
        status_msg = String()
        status_msg.data = " | ".join(status_parts)
        self.safety_status_pub.publish(status_msg)
    
    def publish_stop_command(self):
        """Publish stop command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
    
    def publish_emergency_stop(self, emergency):
        """Publish emergency stop status"""
        emergency_msg = Bool()
        emergency_msg.data = emergency
        self.emergency_stop_pub.publish(emergency_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SafetyMonitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in safety monitor: {e}")
    finally:
        # Send stop command on shutdown
        try:
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            node.cmd_vel_pub.publish(cmd_msg)
        except:
            pass
        
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()