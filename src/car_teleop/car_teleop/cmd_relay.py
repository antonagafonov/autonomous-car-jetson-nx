#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class CmdRelay(Node):
    def __init__(self):
        super().__init__('cmd_relay')
        
        # State variable - ALWAYS start in manual mode
        self.autonomous_mode = False
        self.mode_initialized = False
        self.last_manual_cmd = Twist()
        self.last_autonomous_cmd = Twist()
        
        # Publish to motor controller
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribe to manual commands (ALWAYS active)
        self.manual_subscription = self.create_subscription(
            Twist,
            'cmd_vel_manual',
            self.manual_cmd_callback,
            10)
        
        # Subscribe to autonomous commands
        self.autonomous_subscription = self.create_subscription(
            Twist,
            'cmd_vel_autonomous',
            self.autonomous_cmd_callback,
            10)
        
        # Subscribe to mode control (this comes after manual sub to ensure manual works first)
        self.mode_subscription = self.create_subscription(
            Bool,
            'autonomous_mode',
            self.mode_callback,
            10)
        
        self.get_logger().info('Command Relay Node Started')
        self.get_logger().info('🎮 Mode: MANUAL - Relaying /cmd_vel_manual to /cmd_vel')
        self.get_logger().info('✅ Joystick control is ACTIVE')
    
    def mode_callback(self, msg):
        """Handle autonomous mode changes"""
        prev_mode = self.autonomous_mode
        self.autonomous_mode = msg.data
        self.mode_initialized = True
        
        # Only log if mode actually changed
        if prev_mode != self.autonomous_mode:
            mode_str = "AUTONOMOUS" if self.autonomous_mode else "MANUAL"
            source_topic = "/cmd_vel_autonomous" if self.autonomous_mode else "/cmd_vel_manual"
            
            self.get_logger().info(f'🔄 Mode switched to {mode_str} - Relaying {source_topic} to /cmd_vel')
            
            # When switching modes, send a stop command to ensure clean transition
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.publisher.publish(stop_cmd)
    
    def manual_cmd_callback(self, msg):
        """Handle manual commands from joystick"""
        self.last_manual_cmd = msg
        
        # ALWAYS relay manual commands if not in autonomous mode
        # This ensures joystick works even before mode is explicitly set
        if not self.autonomous_mode:
            self.publisher.publish(msg)
            # Only log non-zero commands to avoid spam
            if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                self.get_logger().debug(f'🎮 Manual: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}')
        else:
            # In autonomous mode, still log that manual commands are being ignored
            if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                self.get_logger().debug('🎮 Manual command ignored (in autonomous mode)')
    
    def autonomous_cmd_callback(self, msg):
        """Handle autonomous commands from inference node"""
        self.last_autonomous_cmd = msg
        
        # Only relay if in autonomous mode
        if self.autonomous_mode:
            self.publisher.publish(msg)
            # Only log non-zero commands to avoid spam
            if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                self.get_logger().debug(f'🤖 Autonomous: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}')
        else:
            # In manual mode, still log that autonomous commands are being ignored
            if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                self.get_logger().debug('🤖 Autonomous command ignored (in manual mode)')

def main(args=None):
    rclpy.init(args=args)
    cmd_relay = CmdRelay()
    rclpy.spin(cmd_relay)
    cmd_relay.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()