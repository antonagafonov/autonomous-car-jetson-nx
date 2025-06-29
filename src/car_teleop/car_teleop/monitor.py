#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import sys

class PS4JoystickMonitor(Node):
    def __init__(self):
        super().__init__('ps4_joystick_monitor')
        
        # Subscribe to joy topic
        self.subscription = self.create_subscription(
            Joy,
            '/joy',  # Default joy topic name
            self.joy_callback,
            10
        )
        
        # PS4 controller button mapping (common layout)
        self.button_names = {
            0: "X (Cross)",
            1: "O (Circle)", 
            2: "△ (Triangle)",
            3: "□ (Square)",
            4: "L1",
            5: "R1",
            6: "L2",
            7: "R2",
            8: "Share",
            9: "Options",
            10: "PS Button",
            11: "L3 (Left Stick)",
            12: "R3 (Right Stick)",
            13: "D-Pad Up",
            14: "D-Pad Down",
            15: "D-Pad Left",
            16: "D-Pad Right"
        }
        
        # Axis mapping
        self.axis_names = {
            0: "Left Stick X",
            1: "Left Stick Y", 
            2: "Right Stick X",
            3: "Right Stick Y",
            4: "L2 Trigger",
            5: "R2 Trigger"
        }
        
        self.get_logger().info("PS4 Joystick Monitor Started")
        self.get_logger().info("Press buttons on your PS4 controller to see their indices")
        self.get_logger().info("=" * 50)
        
        # Track previous states to only show changes
        self.prev_buttons = []
        self.prev_axes = []
        
    def joy_callback(self, msg):
        # Check for button presses
        for i, button_state in enumerate(msg.buttons):
            # Show when button is pressed (state changes from 0 to 1)
            if i < len(self.prev_buttons):
                if button_state == 1 and self.prev_buttons[i] == 0:
                    button_name = self.button_names.get(i, f"Unknown Button {i}")
                    self.get_logger().info(f"BUTTON PRESSED - Index: {i:2d} | Name: {button_name}")
            
        # Check for significant axis changes (to avoid spam from small movements)
        for i, axis_value in enumerate(msg.axes):
            if i < len(self.prev_axes):
                # Only show if axis moved significantly
                if abs(axis_value - self.prev_axes[i]) > 0.1:
                    axis_name = self.axis_names.get(i, f"Unknown Axis {i}")
                    self.get_logger().info(f"AXIS MOVED    - Index: {i:2d} | Name: {axis_name:15s} | Value: {axis_value:6.3f}")
        
        # Update previous states
        self.prev_buttons = list(msg.buttons)
        self.prev_axes = list(msg.axes)
        
        # Optionally, print full state periodically (uncomment if needed)
        # self.print_full_state(msg)
    
    def print_full_state(self, msg):
        """Print complete joystick state - useful for debugging"""
        print("\n" + "="*60)
        print("FULL JOYSTICK STATE:")
        print(f"Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        
        print("\nBUTTONS:")
        for i, button in enumerate(msg.buttons):
            button_name = self.button_names.get(i, f"Button_{i}")
            status = "PRESSED" if button else "released"
            print(f"  [{i:2d}] {button_name:15s}: {status}")
        
        print("\nAXES:")
        for i, axis in enumerate(msg.axes):
            axis_name = self.axis_names.get(i, f"Axis_{i}")
            print(f"  [{i:2d}] {axis_name:15s}: {axis:6.3f}")
        print("="*60)

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node
    joystick_monitor = PS4JoystickMonitor()
    
    try:
        print("\n🎮 PS4 Joystick Button Indices Monitor")
        print("=" * 40)
        print("Instructions:")
        print("1. Make sure your PS4 controller is connected")
        print("2. Start the joy_node: ros2 run joy joy_node")
        print("3. Press buttons to see their indices")
        print("4. Press Ctrl+C to exit")
        print("=" * 40)
        
        # Spin the node
        rclpy.spin(joystick_monitor)
        
    except KeyboardInterrupt:
        print("\n\nShutting down PS4 Joystick Monitor...")
        
    finally:
        # Clean shutdown
        joystick_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()