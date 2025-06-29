# car_teleop/car_teleop/joystick_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

class JoystickController(Node):
    def __init__(self):
        super().__init__('joystick_controller')
        
        # Subscribers
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel_manual', 10)
        self.autonomous_mode_publisher = self.create_publisher(Bool, 'autonomous_mode', 10)
        self.recording_trigger_publisher = self.create_publisher(Bool, 'recording_trigger', 10)
        
        # Parameters
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('deadzone', 0.1)
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.deadzone = self.get_parameter('deadzone').value
        
        # Button mapping (Xbox/PS4 controller)
        self.button_mapping = {
            'autonomous_toggle': 0,  # A button (Xbox) / X button (PS4)
            'emergency_stop': 1,     # B button (Xbox) / Circle button (PS4)
            'recording_toggle': 2,   # X button (Xbox) / Square button (PS4) - NEW
            'slow_mode': 4,          # LB button (Xbox) / L1 button (PS4)
            'turbo_mode': 5          # RB button (Xbox) / R1 button (PS4)
        }
        
        # Axis mapping
        self.axis_mapping = {
            'linear': 1,    # Left stick vertical
            'angular': 3    # Right stick horizontal
        }
        
        # State variables
        self.autonomous_mode = False
        self.emergency_stop = False
        self.slow_mode = False
        self.turbo_mode = False
        self.recording_active = False  # NEW: Recording state
        self.last_button_state = {}
        
        self.get_logger().info('Joystick Controller Node Started')
        self.get_logger().info(f'Max speeds - Linear: {self.max_linear_speed}, Angular: {self.max_angular_speed}')
        self.get_logger().info('Controls:')
        self.get_logger().info('  A/X button: Toggle autonomous mode')
        self.get_logger().info('  B/Circle button: Emergency stop')
        self.get_logger().info('  X/Square button: Toggle recording')  # NEW
        self.get_logger().info('  LB/L1 button: Slow mode (hold)')
        self.get_logger().info('  RB/R1 button: Turbo mode (hold)')
    
    def joy_callback(self, msg):
        # Handle button presses
        self.handle_buttons(msg.buttons)
        
        # Handle movement only if not in autonomous mode and not emergency stopped
        if not self.autonomous_mode and not self.emergency_stop:
            self.handle_movement(msg.axes)
    
    def handle_buttons(self, buttons):
        # Check for button press events (rising edge)
        current_buttons = {}
        for i, button in enumerate(buttons):
            current_buttons[i] = button
        
        # Autonomous mode toggle (on button press, not hold)
        if len(buttons) > self.button_mapping['autonomous_toggle']:
            button_idx = self.button_mapping['autonomous_toggle']
            if (buttons[button_idx] == 1 and 
                self.last_button_state.get(button_idx, 0) == 0):
                self.autonomous_mode = not self.autonomous_mode
                mode_msg = Bool()
                mode_msg.data = self.autonomous_mode
                self.autonomous_mode_publisher.publish(mode_msg)
                
                mode_str = "AUTONOMOUS" if self.autonomous_mode else "MANUAL"
                self.get_logger().info(f'Switched to {mode_str} mode')
        
        # Emergency stop toggle
        if len(buttons) > self.button_mapping['emergency_stop']:
            button_idx = self.button_mapping['emergency_stop']
            if (buttons[button_idx] == 1 and 
                self.last_button_state.get(button_idx, 0) == 0):
                self.emergency_stop = not self.emergency_stop
                if self.emergency_stop:
                    self.stop_robot()
                    # Stop recording if emergency stop is activated
                    if self.recording_active:
                        self.recording_active = False
                        recording_msg = Bool()
                        recording_msg.data = False
                        self.recording_trigger_publisher.publish(recording_msg)
                        self.get_logger().warn('Recording stopped due to emergency stop')
                    self.get_logger().warn('EMERGENCY STOP ACTIVATED')
                else:
                    self.get_logger().info('Emergency stop deactivated')
        
        # NEW: Recording toggle (X/Square button)
        if len(buttons) > self.button_mapping['recording_toggle']:
            button_idx = self.button_mapping['recording_toggle']
            if (buttons[button_idx] == 1 and 
                self.last_button_state.get(button_idx, 0) == 0):
                self.recording_active = not self.recording_active
                recording_msg = Bool()
                recording_msg.data = self.recording_active
                self.recording_trigger_publisher.publish(recording_msg)
                
                status_str = "STARTED" if self.recording_active else "STOPPED"
                self.get_logger().info(f'🎬 Recording {status_str}')
        
        # Speed modifiers (hold buttons)
        if len(buttons) > self.button_mapping['slow_mode']:
            self.slow_mode = buttons[self.button_mapping['slow_mode']] == 1
        
        if len(buttons) > self.button_mapping['turbo_mode']:
            self.turbo_mode = buttons[self.button_mapping['turbo_mode']] == 1
        
        # Store button states for next iteration
        self.last_button_state = current_buttons
    
    def handle_movement(self, axes):
        # Get axis values
        if len(axes) > max(self.axis_mapping.values()):
            linear_axis = axes[self.axis_mapping['linear']]
            angular_axis = axes[self.axis_mapping['angular']]
            
            # Apply deadzone
            linear_axis = 0.0 if abs(linear_axis) < self.deadzone else linear_axis
            angular_axis = 0.0 if abs(angular_axis) < self.deadzone else angular_axis
            
            # Calculate speeds
            linear_speed = linear_axis * self.max_linear_speed
            angular_speed = angular_axis * self.max_angular_speed  # Do not Invert for intuitive control
            
            # Apply speed modifiers
            if self.slow_mode:
                linear_speed *= 0.3
                angular_speed *= 0.3
                if abs(linear_speed) > 0.01 or abs(angular_speed) > 0.01:
                    self.get_logger().debug('SLOW MODE active')
            elif self.turbo_mode:
                linear_speed *= 1.5
                angular_speed *= 1.5
                if abs(linear_speed) > 0.01 or abs(angular_speed) > 0.01:
                    self.get_logger().debug('TURBO MODE active')
            
            # Create and publish Twist message
            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
            
            self.cmd_vel_publisher.publish(twist)
    
    def stop_robot(self):
        """Send stop command to robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    joystick_controller = JoystickController()
    rclpy.spin(joystick_controller)
    joystick_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()