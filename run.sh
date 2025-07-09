ros2 run car_inference inference_node --ros-args \
  -p max_linear_velocity:=0.7 \
  -p max_angular_velocity:=2.5 \
  -p publish_rate:=40.0 \
  -p smoothing_alpha:=0.7 \
  -p prediction_skip_count:=2 \
  -p angular_command_threshold:=2.0 \
  -p burst_steer_count:=3 \
  -p burst_zero_count:=3 \
  -p max_steer_bursts:=5

ros2 launch car_bringup car_manual_control.launch.py framerate:=10

ros2 topic echo /car/queue_size

ros2 topic echo /cmd_vel_autonomous

