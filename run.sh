ros2 run car_inference inference_node --ros-args -p max_linear_velocity:=0.5 -p max_angular_velocity:=0.3 -p publish_rate:=40.0 X
ros2 run car_inference inference_node --ros-args -p max_linear_velocity:=0.5 -p max_angular_velocity:=0.7 -p publish_rate:=40.0

ros2 launch car_bringup car_manual_control.launch.py framerate:=10

ros2 topic echo /car/queue_size

ros2 topic echo /cmd_vel_autonomous