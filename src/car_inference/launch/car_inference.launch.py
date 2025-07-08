#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('car_inference'),
            'config',
            'inference_config.yaml'
        ]),
        description='Path to configuration file'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/anton/jetson/models/best_model.pth',
        description='Path to the trained model file'
    )
    
    enable_autonomous_arg = DeclareLaunchArgument(
        'enable_autonomous',
        default_value='false',
        description='Enable autonomous mode on startup (safety: default false)'
    )
    
    use_cuda_arg = DeclareLaunchArgument(
        'use_cuda',
        default_value='true',
        description='Use CUDA for inference if available'
    )
    
    inference_rate_arg = DeclareLaunchArgument(
        'inference_rate',
        default_value='20.0',
        description='Inference rate in Hz'
    )
    
    enable_safety_monitor_arg = DeclareLaunchArgument(
        'enable_safety_monitor',
        default_value='true',
        description='Enable safety monitor node'
    )
    
    enable_model_manager_arg = DeclareLaunchArgument(
        'enable_model_manager',
        default_value='true',
        description='Enable model manager node'
    )
    
    # Model Manager Node
    model_manager_node = Node(
        package='car_inference',
        executable='model_manager',
        name='model_manager',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_model_manager'))
    )
    
    # Inference Node
    inference_node = Node(
        package='car_inference',
        executable='inference_node',
        name='car_inference_node',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'model_path': LaunchConfiguration('model_path'),
                'enable_autonomous': LaunchConfiguration('enable_autonomous'),
                'device': 'cuda' if LaunchConfiguration('use_cuda') else 'cpu',
                'inference_rate': LaunchConfiguration('inference_rate'),
            }
        ],
        output='screen',
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/cmd_vel_autonomous', '/cmd_vel_autonomous'),
        ]
    )
    
    # Safety Monitor Node
    safety_monitor_node = Node(
        package='car_inference',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/cmd_vel_autonomous', '/cmd_vel_autonomous'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_safety_monitor'))
    )
    
    # Group all nodes
    car_inference_group = GroupAction([
        model_manager_node,
        inference_node,
        safety_monitor_node,
    ])
    
    return LaunchDescription([
        config_file_arg,
        model_path_arg,
        enable_autonomous_arg,
        use_cuda_arg,
        inference_rate_arg,
        enable_safety_monitor_arg,
        enable_model_manager_arg,
        car_inference_group,
    ])