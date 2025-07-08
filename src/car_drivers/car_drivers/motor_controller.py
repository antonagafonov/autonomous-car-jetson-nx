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
        
        # Motor pin mappings for Jetson Xavier NX (updated to match working test)
        if self.pin_mode == 'BCM':
            # Using BCM numbering - convert from BOARD numbers
            self.in1_left = 18   # Left side direction 1 (BOARD pin 18)
            self.in2_left = 16   # Left side direction 2 (BOARD pin 16)
            self.en_left = 15    # Left side PWM enable (BOARD pin 15)
            self.in1_right = 35  # Right side direction 1 (BOARD pin 35)
            self.in2_right = 37  # Right side direction 2 (BOARD pin 37)
            self.en_right = 32   # Right side PWM enable (BOARD pin 32)
            GPIO.setmode(GPIO.BCM)
        else:
            # Using BOARD numbering for Jetson Xavier NX (matches your working test)
            self.in1_left = 18   # Left side direction 1 (Pin 18)
            self.in2_left = 16   # Left side direction 2 (Pin 16)  
            self.en_left = 15    # Left side PWM enable (Pin 15) - working PWM
            self.in1_right = 35  # Right side direction 1 (Pin 35)
            self.in2_right = 37  # Right side direction 2 (Pin 37)
            self.en_right = 32   # Right side PWM enable (Pin 32) - now working PWM
            GPIO.setmode(GPIO.BOARD)
        
        try:
            # Setup GPIO pins
            GPIO.setup([self.in1_left, self.in2_left, self.en_left, 
                       self.in1_right, self.in2_right, self.en_right], GPIO.OUT, initial=GPIO.LOW)
            
            # Initialize PWM at 100Hz (matching your original frequency)
            self.pwm_left = GPIO.PWM(self.en_left, 100)   # Left side PWM (Pin 15)
            self.pwm_right = GPIO.PWM(self.en_right, 100) # Right side PWM (Pin 32)
            self.pwm_left.start(0)
            self.pwm_right.start(0)
            
            self.get_logger().info(f'GPIO setup completed successfully')
            self.get_logger().info(f'Left side pins: IN1={self.in1_left}, IN2={self.in2_left}, PWM={self.en_left}')
            self.get_logger().info(f'Right side pins: IN1={self.in1_right}, IN2={self.in2_right}, PWM={self.en_right}')
            
        except Exception as e:
            self.get_logger().error(f'GPIO setup failed: {e}')
            self.get_logger().error('Make sure PWM pins are enabled in device tree')
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
            self.pwm_left.ChangeDutyCycle(abs(left_speed))
            self.pwm_right.ChangeDutyCycle(abs(right_speed))
            
            # Control left side motors direction
            if left_speed > 0:
                GPIO.output(self.in1_left, GPIO.HIGH)
                GPIO.output(self.in2_left, GPIO.LOW)
            elif left_speed < 0:
                GPIO.output(self.in1_left, GPIO.LOW)
                GPIO.output(self.in2_left, GPIO.HIGH)
            else:
                GPIO.output(self.in1_left, GPIO.LOW)
                GPIO.output(self.in2_left, GPIO.LOW)
            
            # Control right side motors direction
            if right_speed > 0:
                GPIO.output(self.in1_right, GPIO.HIGH)
                GPIO.output(self.in2_right, GPIO.LOW)
            elif right_speed < 0:
                GPIO.output(self.in1_right, GPIO.LOW)
                GPIO.output(self.in2_right, GPIO.HIGH)
            else:
                GPIO.output(self.in1_right, GPIO.LOW)
                GPIO.output(self.in2_right, GPIO.LOW)
                
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
            self.pwm_left.ChangeDutyCycle(0)
            self.pwm_right.ChangeDutyCycle(0)
            GPIO.output([self.in1_left, self.in2_left, self.in1_right, self.in2_right], GPIO.LOW)
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
                    self.pwm_left.stop()
                    self.pwm_right.stop()
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