from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
import launch.conditions

def generate_launch_description():
    # Launch arguments for configuration
    return LaunchDescription([
        # ================== RECORDER ARGUMENT (NOW DEFAULT TRUE) ==================
        DeclareLaunchArgument(
            'enable_recorder',
            default_value='true',  # CHANGED: Now enabled by default
            description='Enable the smart data recorder (set to false to disable)'
        ),
        
        # Motor control arguments
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
        
        # Camera arguments
        DeclareLaunchArgument(
            'enable_camera',
            default_value='true',  # CHANGED: Now enabled by default
            description='Enable camera node (set to false to disable camera)'
        ),
        DeclareLaunchArgument(
            'camera_width',
            default_value='1280',
            description='Camera sensor width resolution'
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='720',
            description='Camera sensor height resolution'
        ),
        DeclareLaunchArgument(
            'output_width',
            default_value='320',
            description='Output image width resolution'
        ),
        DeclareLaunchArgument(
            'output_height',
            default_value='240',
            description='Output image height resolution'
        ),
        DeclareLaunchArgument(
            'framerate',
            default_value='20',
            description='Camera framerate in fps'
        ),
        DeclareLaunchArgument(
            'flip_method',
            default_value='2',
            description='Camera flip method (0=none, 1=90CW, 2=180°, 3=90CCW, 4=H-flip, 5=V-flip, 6=both)'
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Camera sensor ID'
        ),
        DeclareLaunchArgument(
            'enable_image_viewer',
            default_value='false',
            description='Enable image viewer node'
        ),
        
        # ================== WEB STREAMER ARGUMENTS ==================
        DeclareLaunchArgument(
            'enable_web_streamer',
            default_value='false',  # NEW: Enabled by default
            description='Enable web image streamer for network viewing'
        ),
        DeclareLaunchArgument(
            'web_port',
            default_value='8080',
            description='Port for web image streamer'
        ),
        DeclareLaunchArgument(
            'web_quality',
            default_value='80',
            description='JPEG quality for web stream (0-100)'
        ),
        DeclareLaunchArgument(
            'web_max_width',
            default_value='1280',
            description='Maximum width for web stream'
        ),
        DeclareLaunchArgument(
            'web_max_height',
            default_value='720',
            description='Maximum height for web stream'
        ),
        
        # ================== RECORDER NODE (START EARLY) ==================
        # Start the recorder early with a small delay to ensure it's ready
        # NOTE: No respawn to allow HOME button shutdown
        TimerAction(
            period=1.0,  # Start after 1 second
            actions=[
                Node(
                    package='data_collect',
                    executable='bag_collect',
                    name='bag_collect',
                    output='screen',
                    condition=launch.conditions.IfCondition(LaunchConfiguration('enable_recorder')),
                    # REMOVED: respawn=True, respawn_delay=2.0
                )
            ]
        ),

        # Node 1: Motor Controller (start first)
        # NOTE: No respawn to allow HOME button shutdown
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
            # REMOVED: respawn=True, respawn_delay=2.0
        ),
        
        # Node 2: Camera Node (start with motor controller)
        # NOTE: No respawn to allow HOME button shutdown
        Node(
            package='car_perception',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'camera_width': LaunchConfiguration('camera_width'),
                'camera_height': LaunchConfiguration('camera_height'),
                'output_width': LaunchConfiguration('output_width'),
                'output_height': LaunchConfiguration('output_height'),
                'framerate': LaunchConfiguration('framerate'),
                'flip_method': LaunchConfiguration('flip_method'),
                'camera_id': LaunchConfiguration('camera_id'),
            }],
            # REMOVED: respawn=True, respawn_delay=2.0
            condition=launch.conditions.IfCondition(LaunchConfiguration('enable_camera')),
        ),
        
        # Node 3: Joy Node (start after small delay)
        # NOTE: No respawn to allow HOME button shutdown
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
                    # REMOVED: respawn=True, respawn_delay=2.0
                )
            ]
        ),
        
        # Node 4: Joystick Controller (start after joy node)
        # NOTE: No respawn for joystick controller so HOME button can shutdown system
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
                    # REMOVED: respawn=True, respawn_delay=2.0
                    # This allows HOME button to shut down the system
                )
            ]
        ),
        
        # Node 5: Command Relay (start after joystick controller)
        # NOTE: No respawn to allow HOME button shutdown
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='car_teleop',
                    executable='cmd_relay',
                    name='cmd_relay',
                    output='screen',
                    # REMOVED: respawn=True, respawn_delay=2.0
                )
            ]
        ),
        
        # ================== WEB STREAMER NODE (START AFTER CAMERA) ==================
        # Start web streamer after camera is stable
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='car_web_streamer',  # NEW PACKAGE
                    executable='web_image_streamer',
                    name='web_image_streamer',
                    output='screen',
                    parameters=[{
                        'port': LaunchConfiguration('web_port'),
                        'quality': LaunchConfiguration('web_quality'),
                        'max_width': LaunchConfiguration('web_max_width'),
                        'max_height': LaunchConfiguration('web_max_height'),
                    }],
                    condition=launch.conditions.IfCondition(LaunchConfiguration('enable_web_streamer')),
                    # NOTE: No respawn to allow HOME button shutdown
                )
            ]
        ),
        
        # Node 6: Image Viewer (optional, start after camera is stable)
        # NOTE: No respawn to allow HOME button shutdown
        TimerAction(
            period=10.0,  # Moved to after web streamer
            actions=[
                Node(
                    package='car_perception',
                    executable='image_viewer',
                    name='image_viewer',
                    output='screen',
                    # REMOVED: respawn=True, respawn_delay=2.0
                    condition=launch.conditions.IfCondition(LaunchConfiguration('enable_image_viewer')),
                )
            ]
        )
    ])