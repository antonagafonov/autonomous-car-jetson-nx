from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments for configuration
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'max_linear_speed',
            default_value='1.0',
            description='Maximum linear speed (m/s)'
        ),
        DeclareLaunchArgument(
            'max_angular_speed', 
            default_value='2.0',
            description='Maximum angular speed (rad/s)'
        ),
        DeclareLaunchArgument(
            'min_linear_speed',
            default_value='0.4', 
            description='Minimum linear speed (m/s)'
        ),
        DeclareLaunchArgument(
            'base_speed_scale',
            default_value='80',
            description='Motor speed scale (0-100%)'
        ),
        
        # Node 1: Motor Controller (start first)
        Node(
            package='car_drivers',
            executable='motor_controller',
            name='motor_controller',
            output='screen',
            parameters=[{
                'base_speed_scale': LaunchConfiguration('base_speed_scale'),
                'steering_offset': 0.0,
                'max_speed': LaunchConfiguration('max_linear_speed'),
                'pin_mode': 'BOARD'
            }],
            respawn=True,
            respawn_delay=2.0
        ),
        
        # Node 2: Joy Node (start after small delay)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='joy',
                    executable='joy_node',
                    name='joy_node',
                    output='screen',
                    parameters=[{
                        'device_id': 0,
                        'deadzone': 0.1,
                        'autorepeat_rate': 20.0
                    }],
                    respawn=True,
                    respawn_delay=2.0
                )
            ]
        ),
        
        # Node 3: Joystick Controller (start after joy node)
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='car_teleop',
                    executable='joystick_controller',
                    name='joystick_controller',
                    output='screen',
                    parameters=[{
                        'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                        'max_angular_speed': LaunchConfiguration('max_angular_speed'),
                        'min_linear_speed': LaunchConfiguration('min_linear_speed'),
                        'deadzone': 0.1
                    }],
                    respawn=True,
                    respawn_delay=2.0
                )
            ]
        ),
        
        # Node 4: Command Relay (start last)
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='car_teleop',
                    executable='cmd_relay',
                    name='cmd_relay',
                    output='screen',
                    respawn=True,
                    respawn_delay=2.0
                )
            ]
        )
    ])
