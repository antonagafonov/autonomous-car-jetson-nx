# car_drivers/car_drivers/motor_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO
import threading
import time

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        # Declare parameters FIRST
        self.declare_parameter('base_speed_scale', 80)  # Speed scale factor (0-100)
        self.declare_parameter('steering_offset', 0.0)  # Steering calibration offset
        self.declare_parameter('max_speed', 1.0)        # Maximum speed (m/s)
        self.declare_parameter('pin_mode', 'BOARD')     # GPIO pin mode: 'BOARD' or 'BCM'
        
        # Get parameter values
        self.base_speed_scale = self.get_parameter('base_speed_scale').value
        self.steering_offset = self.get_parameter('steering_offset').value
        self.max_speed = self.get_parameter('max_speed').value
        self.pin_mode = self.get_parameter('pin_mode').value
        
        # Initialize motor control
        self.setup_gpio()
        
        # Subscribe to cmd_vel commands
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # State variables
        self.startup = True
        self.my_speed = 0
        self.straight_count = 0
        self.cleanup_done = False
        self.cleanup_lock = threading.Lock()
        
        self.get_logger().info('Motor Controller Node Started (Jetson Xavier NX with RPi.GPIO)')
        self.get_logger().info(f'GPIO Model: {GPIO.model}')
        self.get_logger().info(f'GPIO Mode: {self.pin_mode}')
        self.get_logger().info(f'Base speed scale: {self.base_speed_scale}%')
        self.get_logger().info(f'Steering offset: {self.steering_offset}')
    
    def setup_gpio(self):
        """Setup GPIO pins for motor control - Jetson Xavier NX with RPi.GPIO"""
        
        # Check if we're on a supported Jetson board
        try:
            board_model = GPIO.model
            self.get_logger().info(f'Detected board: {board_model}')
        except:
            self.get_logger().warn('Could not detect board model')
        
        # Motor pin mappings for Jetson Xavier NX
        if self.pin_mode == 'BCM':
            # Using BCM numbering - map your RPi pins to equivalent Jetson pins
            self.in1a = 24  # Motor A Direction pin 1
            self.in2a = 23  # Motor A Direction pin 2  
            self.en_a = 25  # Motor A PWM enable pin
            self.in1b = 20  # Motor B Direction pin 1
            self.in2b = 16  # Motor B Direction pin 2
            self.en_b = 21  # Motor B PWM enable pin
            GPIO.setmode(GPIO.BCM)
        else:
            # Using BOARD numbering for Jetson Xavier NX (recommended)
            # These pins are PWM-capable on Xavier NX
            self.in1a = 18  # Pin 18 - Motor A Direction pin 1
            self.in2a = 16  # Pin 16 - Motor A Direction pin 2
            self.en_a = 33  # Pin 33 - Motor A PWM enable pin (Xavier NX PWM pin)
            self.in1b = 37  # Pin 37 - Motor B Direction pin 1  
            self.in2b = 35  # Pin 35 - Motor B Direction pin 2
            self.en_b = 32  # Pin 32 - Motor B PWM enable pin (Xavier NX PWM pin)
            GPIO.setmode(GPIO.BOARD)
        
        try:
            # Setup GPIO pins
            GPIO.setup([self.in1a, self.in2a, self.en_a, self.in1b, self.in2b, self.en_b], GPIO.OUT, initial=GPIO.LOW)
            
            # Initialize PWM at 100Hz (matching your original frequency)
            self.pwm_a = GPIO.PWM(self.en_a, 100)
            self.pwm_b = GPIO.PWM(self.en_b, 100)
            self.pwm_a.start(0)
            self.pwm_b.start(0)
            
            self.get_logger().info(f'GPIO setup completed successfully')
            self.get_logger().info(f'Motor A pins: {self.in1a}, {self.in2a}, {self.en_a}')
            self.get_logger().info(f'Motor B pins: {self.in1b}, {self.in2b}, {self.en_b}')
            
        except Exception as e:
            self.get_logger().error(f'GPIO setup failed: {e}')
            self.get_logger().error('Make sure you have proper permissions and the pins are available')
            raise
    
    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands"""
        if self.startup:
            self.stop_motors()
            self.startup = False
        
        # Extract linear and angular velocities
        speed = msg.linear.x / self.max_speed  # Normalize to 0-1 range
        turn = msg.angular.z  # Angular velocity (rad/s)
        
        # Use your original move function logic
        self.move(speed=speed, turn=turn, boost=0, t=0, 
                 steering_offset=self.steering_offset, s=self.base_speed_scale)
    
    def move(self, speed=0.5, turn=0, boost=0, t=0.05, steering_offset=0.0, s=80):
        """
        Move the vehicle with specified speed and turn rate
        Args:
            speed: Forward speed (0-1)
            turn: Turn rate (-1 to 1, negative=left, positive=right)
            boost: Speed boost factor (0-1)
            t: Sleep time after movement
            steering_offset: Calibration offset for straight driving
            s: Base speed scale (0-100)
        """
        # Track straight driving for potential calibration
        if abs(turn) < 0.1:
            self.straight_count += 1
        else:
            self.straight_count = 0
        
        # Apply steering offset when driving straight
        if speed > 0.05:
            if abs(turn) < 0.1:
                turn += steering_offset
        
        # Scale speed with boost
        speed = round(speed * (s + (100-s) * boost), 1)
        turn = round(turn * 100, 1)
        
        # Calculate differential drive speeds
        left_speed = speed - turn
        right_speed = speed + turn
        
        # Clamp speeds to [-100, 100]
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        try:
            # Set PWM duty cycles
            self.pwm_a.ChangeDutyCycle(abs(left_speed))
            self.pwm_b.ChangeDutyCycle(abs(right_speed))
            
            # Control left motor (Motor A) direction
            if left_speed > 0:
                GPIO.output(self.in1a, GPIO.HIGH)
                GPIO.output(self.in2a, GPIO.LOW)
            elif left_speed < 0:
                GPIO.output(self.in1a, GPIO.LOW)
                GPIO.output(self.in2a, GPIO.HIGH)
            else:
                GPIO.output(self.in1a, GPIO.LOW)
                GPIO.output(self.in2a, GPIO.LOW)
            
            # Control right motor (Motor B) direction
            if right_speed > 0:
                GPIO.output(self.in1b, GPIO.HIGH)
                GPIO.output(self.in2b, GPIO.LOW)
            elif right_speed < 0:
                GPIO.output(self.in1b, GPIO.LOW)
                GPIO.output(self.in2b, GPIO.HIGH)
            else:
                GPIO.output(self.in1b, GPIO.LOW)
                GPIO.output(self.in2b, GPIO.LOW)
                
        except Exception as e:
            self.get_logger().error(f'Motor control error: {e}')
        
        # Optional sleep (usually not needed in ROS callback)
        if t > 0:
            time.sleep(t)
        
        # Debug logging
        self.get_logger().debug(
            f'Speed: {speed:.1f}, Turn: {turn:.1f}, Left: {left_speed:.1f}, Right: {right_speed:.1f}'
        )
    
    def stop_motors(self, t=0):
        """Stop the motors"""
        try:
            self.pwm_a.ChangeDutyCycle(0)
            self.pwm_b.ChangeDutyCycle(0)
            GPIO.output([self.in1a, self.in2a, self.in1b, self.in2b], GPIO.LOW)
            self.my_speed = 0
            if t > 0:
                time.sleep(t)
            self.get_logger().info('Motors stopped')
        except Exception as e:
            self.get_logger().error(f'Error stopping motors: {e}')
    
    def set_speed(self, speed):
        """Set the current speed value"""
        self.my_speed = speed
    
    def cleanup_gpio(self):
        """Clean up GPIO resources"""
        with self.cleanup_lock:
            if not self.cleanup_done:
                try:
                    self.get_logger().info("Stopping PWM and cleaning up GPIO...")
                    self.stop_motors()
                    self.pwm_a.stop()
                    self.pwm_b.stop()
                    GPIO.cleanup()
                    self.cleanup_done = True
                    self.get_logger().info("GPIO cleanup completed")
                except Exception as e:
                    self.get_logger().error(f"GPIO cleanup error: {e}")
    
    def destroy_node(self):
        """Override destroy_node to ensure GPIO cleanup"""
        self.cleanup_gpio()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    
    try:
        rclpy.spin(motor_controller)
    except KeyboardInterrupt:
        pass
    finally:
        motor_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# # Source ROS2 and workspace
# source /opt/ros/humble/setup.bash
# source ~/car_ws/install/setup.bash

# # Test 1: Stop (just to be safe)
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once

# # Test 2: Very slow forward (start conservative)
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once

# # Test 3: Stop
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once

# # Test 4: Slow turn left
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}' --once

# # Test 5: Slow turn right  
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.3}}' --once

# # Test 6: Stop
# ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once

# Adjust parameters

# # Reduce speed for testing
# ros2 param set /motor_controller base_speed_scale 30

# # Reduce max speed
# ros2 param set /motor_controller max_speed 0.3

# # Check current parameters
# ros2 param list /motor_controller
# ros2 param get /motor_controller base_speed_scale

# Your Pin Configuration
# Based on the output, your motors are connected to:

# Motor A (Left):

# Direction: Pins 18, 16
# PWM: Pin 33


# Motor B (Right):

# Direction: Pins 37, 35
# PWM: Pin 32