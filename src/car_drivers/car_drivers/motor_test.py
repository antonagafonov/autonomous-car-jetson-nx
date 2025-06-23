# car_drivers/car_drivers/motor_test.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MotorTest(Node):
    def __init__(self):
        super().__init__('motor_test')
        
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Test sequence timer
        self.test_timer = self.create_timer(2.0, self.run_test_sequence)
        self.test_step = 0
        
        self.get_logger().info('Motor Test Node Started - Running test sequence...')
    
    def run_test_sequence(self):
        """Run a sequence of motor tests"""
        twist = Twist()
        
        if self.test_step == 0:
            # Stop
            self.get_logger().info('Test 0: Stop')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        elif self.test_step == 1:
            # Forward slow
            self.get_logger().info('Test 1: Forward slow')
            twist.linear.x = 0.3
            twist.angular.z = 0.0
            
        elif self.test_step == 2:
            # Backward slow
            self.get_logger().info('Test 2: Backward slow')
            twist.linear.x = -0.3
            twist.angular.z = 0.0
            
        elif self.test_step == 3:
            # Turn left
            self.get_logger().info('Test 3: Turn left')
            twist.linear.x = 0.0
            twist.angular.z = 0.5
            
        elif self.test_step == 4:
            # Turn right
            self.get_logger().info('Test 4: Turn right')
            twist.linear.x = 0.0
            twist.angular.z = -0.5
            
        elif self.test_step == 5:
            # Forward with left turn
            self.get_logger().info('Test 5: Forward with left turn')
            twist.linear.x = 0.3
            twist.angular.z = 0.3
            
        elif self.test_step == 6:
            # Forward with right turn
            self.get_logger().info('Test 6: Forward with right turn')
            twist.linear.x = 0.3
            twist.angular.z = -0.3
            
        else:
            # Stop and end
            self.get_logger().info('Test complete - stopping')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.test_timer.cancel()
        
        self.cmd_vel_publisher.publish(twist)
        self.test_step += 1

def main(args=None):
    rclpy.init(args=args)
    motor_test = MotorTest()
    rclpy.spin(motor_test)
    motor_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()