#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/toon/car_ws/models/best_model.pth',
        description='Path to the trained model file'
    )
    
    # Test Inference Node (without safety systems)
    test_inference_node = Node(
        package='car_inference',
        executable='inference_node',
        name='test_inference_node',
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'sequence_length': 10,
                'inference_rate': 10.0,  # Lower rate for testing
                'use_vertical_crop': False,
                'crop_pixels': 100,
                'max_angular_velocity': 0.5,  # Conservative limits for testing
                'max_linear_velocity': 0.3,
                'confidence_threshold': 0.05,  # Lower threshold for testing
                'safety_timeout': 3.0,
                'enable_autonomous': True,  # Start disabled
                'device': 'cuda',
            }
        ],
        output='screen',
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/cmd_vel_autonomous', '/cmd_vel_test'),  # Different topic for testing
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        test_inference_node,
    ])