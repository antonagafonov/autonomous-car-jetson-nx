#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdRelay(Node):
    def __init__(self):
        super().__init__('cmd_relay')
        
        # Subscribe to manual commands
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel_manual',
            self.cmd_callback,
            10)
        
        # Publish to motor controller
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.get_logger().info('Command Relay Node Started - Relaying /cmd_vel_manual to /cmd_vel')
    
    def cmd_callback(self, msg):
        # Simply relay the message
        self.publisher.publish(msg)
        self.get_logger().debug(f'Relaying command: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    cmd_relay = CmdRelay()
    rclpy.spin(cmd_relay)
    cmd_relay.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()