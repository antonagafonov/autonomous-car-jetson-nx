# 🚗 Autonomous Car Project - Jetson Xavier NX

![ROS2](https://img.shields.io/badge/ROS2-Foxy-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Xavier%20NX-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

A comprehensive autonomous car project built with ROS2 Foxy on NVIDIA Jetson Xavier NX. Features real-time joystick control, differential drive motor control, CSI camera integration, and expandable architecture for autonomous navigation capabilities.

![Autonomous Car](images/4.jpeg)
*The completed autonomous car with Jetson Xavier NX, 4-motor differential drive, CSI camera, and Bluetooth joystick control*

## 🎯 Project Overview

This project transforms a basic RC car into an intelligent autonomous vehicle using:
- **NVIDIA Jetson Xavier NX** for high-performance edge computing
- **ROS2 Foxy** for robust robotics software architecture
- **Differential drive control** for precise movement
- **CSI Camera integration** with hardware-accelerated pipeline
- **Bluetooth joystick** for intuitive manual control
- **Modular design** ready for autonomous navigation features

## ✨ Features

- ✅ **Manual Control**: Bluetooth joystick teleoperation with Xbox/PS4 controller support
- ✅ **Motor Control**: Precise PWM-based differential drive control with 4-motor setup
- ✅ **Camera Integration**: CSI camera with GStreamer hardware acceleration and 180° flip
- ✅ **Real-time Vision**: Live camera feed publishing to ROS2 topics at 30fps
- ✅ **One-Command Launch**: Complete system startup with single launch file
- ✅ **Safety Systems**: Emergency stop, speed limiting, and mode switching with minimum speed threshold
- ✅ **Real-time Performance**: Low-latency control loop for responsive driving
- ✅ **Modular Architecture**: Clean ROS2 package structure for easy expansion
- ✅ **PWM Configuration**: Hardware PWM on both motor sides with device tree optimization
- 🚧 **Lane Detection**: OpenCV-based computer vision (in development)
- 🚧 **Autonomous Navigation**: PID control and path planning (planned)

## 🔧 Hardware Requirements

### Core Components
- **NVIDIA Jetson Xavier NX** Developer Kit
- **CSI Camera** (IMX219 or compatible) or USB webcam
- **4 DC Motors** (2 per side) with H-bridge motor drivers
- **Bluetooth Joystick** (Xbox One, PS4, or compatible)
- **Power Supply** (12V for motors, 5V for Jetson)

### Optional Components
- **IMU Sensor** for orientation tracking
- **Ultrasonic Sensors** for obstacle detection
- **Servo Motor** for camera gimbal

## 📋 Pin Configuration

### Jetson Xavier NX GPIO (BOARD numbering)

| Component | Pin | Function |
|-----------|-----|----------|
| **Left Side Motors** | 18, 16 | Direction Control |
| | 15 | PWM Speed Control |
| **Right Side Motors** | 35, 37 | Direction Control |
| | 32 | PWM Speed Control |
| **CSI Camera** | CSI Connector | Camera Interface |

> **Note**: Pins 15 and 32 are hardware PWM-capable pins on the Jetson Xavier NX. Enable PWM in device tree with `sudo /opt/nvidia/jetson-io/jetson-io.py`

## 🚀 Installation

### Prerequisites

```bash
# Install ROS2 Foxy (if not already installed)
sudo apt update
sudo apt install ros-foxy-desktop

# Install joystick support
sudo apt install ros-foxy-joy ros-foxy-teleop-twist-joy

# Install camera and vision packages
sudo apt install ros-foxy-cv-bridge ros-foxy-image-transport
sudo apt install python3-opencv

# Install GPIO library
sudo apt install python3-rpi.gpio

# Install development tools
sudo apt install python3-colcon-common-extensions
```

### Setup GPIO and PWM Configuration

```bash
# Enable PWM pins in device tree
sudo /opt/nvidia/jetson-io/jetson-io.py
# Select "Configure for compatible hardware" and enable all PWM functions

# Add user to GPIO group
sudo usermod -a -G gpio $USER

# Logout and login again, or reboot
sudo reboot
```

### Clone and Build

```bash
# Create workspace
mkdir -p ~/car_ws/src
cd ~/car_ws

# Clone repository
git clone https://github.com/yourusername/autonomous-car-jetson-nx.git src/

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build --symlink-install

# Source workspace
source install/setup.bash
echo "source ~/car_ws/install/setup.bash" >> ~/.bashrc
```

## 🚀 Usage

### Quick Start - One Command Launch

**Launch entire system with camera:**
```bash
source ~/car_ws/install/setup.bash
ros2 launch car_bringup car_full_system.launch.py
```

**Launch camera system only:**
```bash
ros2 launch car_bringup car_camera.launch.py
```

**Launch manual control only (no camera - default):**
```bash
ros2 launch car_bringup car_manual_control.launch.py
```

**Launch manual control with camera:**
```bash
ros2 launch car_bringup car_manual_control.launch.py enable_camera:=true
```

**Launch with custom parameters:**
```bash
# Manual control only (motors + joystick, no camera)
ros2 launch car_bringup car_manual_control.launch.py

# Manual control with camera enabled
ros2 launch car_bringup car_manual_control.launch.py enable_camera:=true

# Manual control with camera and viewer for debugging
ros2 launch car_bringup car_manual_control.launch.py \
  enable_camera:=true enable_image_viewer:=true

# Conservative settings for testing with camera
ros2 launch car_bringup car_manual_control.launch.py \
  enable_camera:=true max_linear_speed:=0.5 base_speed_scale:=40 \
  output_width:=320 output_height:=240

# Performance settings with higher resolution
ros2 launch car_bringup car_full_system.launch.py \
  max_linear_speed:=0.8 camera_width:=1920 camera_height:=1080 \
  output_width:=960 output_height:=540

# Camera only with viewer
ros2 launch car_bringup car_camera.launch.py \
  camera_width:=1280 camera_height:=720 enable_viewer:=true

# Full system with image viewer enabled
ros2 launch car_bringup car_full_system.launch.py \
  enable_image_viewer:=true
```

### Camera System

**Start camera node separately:**
```bash
# Default 640x480 output from 1280x720 camera
ros2 run car_perception camera_node

# Custom resolution and flip settings
ros2 run car_perception camera_node --ros-args -p output_width:=800 -p output_height:=600 -p flip_method:=0

# High resolution mode
ros2 run car_perception camera_node --ros-args -p camera_width:=1920 -p camera_height:=1080 -p output_width:=960 -p output_height:=540
```

**View camera feed:**
```bash
# Simple image viewer node (displays camera feed in OpenCV window)
ros2 run car_perception image_viewer

# Or run directly with Python
python3 src/car_perception/car_perception/image_viewer.py

# Or use RQT image viewer
ros2 run rqt_image_view rqt_image_view

# Or use rviz2
rviz2
```

**Check camera status:**
```bash
# List camera topics
ros2 topic list | grep camera

# Monitor camera feed rate
ros2 topic hz /camera/image_raw

# Check image info
ros2 topic info /camera/image_raw
```

### Manual Launch (Individual Nodes)

**Terminal 1: Motor Controller**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_drivers motor_controller
```

**Terminal 2: Camera Node**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_perception camera_node
```

**Terminal 3: Joystick Input**
```bash
source ~/car_ws/install/setup.bash
ros2 run joy joy_node
```

**Terminal 4: Joystick Controller**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_teleop joystick_controller
```

**Terminal 5: Command Relay**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_teleop cmd_relay
```

**Terminal 6: Image Viewer**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_perception image_viewer
```

### Joystick Controls

| Control | Action | Description |
|---------|--------|-------------|
| **Left Stick ↕** | Forward/Backward | Move car forward or reverse (min 0.4 m/s) |
| **Left Stick ↔** | Turn Left/Right | Steer the car |
| **A Button** | Mode Toggle | Switch between manual/autonomous |
| **B Button** | Emergency Stop | Immediate stop with toggle |
| **LB/L1** | Slow Mode | Reduce speed to 40% |
| **RB/R1** | Turbo Mode | Increase speed to 150% |

### Command Line Control

```bash
# Move forward at 20cm/s
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.2}, angular: {z: 0.0}}' --once

# Stop the car
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0}, angular: {z: 0.0}}' --once

# Turn left while moving forward
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.2}, angular: {z: 0.5}}' --once
```

## 📡 ROS2 Architecture

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joy` | `sensor_msgs/Joy` | Raw joystick input |
| `/cmd_vel_manual` | `geometry_msgs/Twist` | Manual control commands |
| `/cmd_vel` | `geometry_msgs/Twist` | Final motor commands |
| `/autonomous_mode` | `std_msgs/Bool` | Current operation mode |
| `/camera/image_raw` | `sensor_msgs/Image` | Live camera feed (640x480 @ 30fps) |

### Nodes

| Node | Package | Function |
|------|---------|----------|
| `motor_controller` | `car_drivers` | GPIO motor control |
| `camera_node` | `car_perception` | CSI camera interface with GStreamer |
| `image_viewer` | `car_perception` | Live camera feed display with OpenCV |
| `joystick_controller` | `car_teleop` | Joystick input processing |
| `cmd_relay` | `car_teleop` | Command routing |
| `joy_node` | `joy` | Joystick hardware interface |

## ⚙️ Configuration

### Motor Controller Parameters

```bash
# Adjust speed scaling (0-100%)
ros2 param set /motor_controller base_speed_scale 60

# Set maximum speed (m/s)
ros2 param set /motor_controller max_speed 0.8

# Calibrate steering offset
ros2 param set /motor_controller steering_offset 0.05

# Change GPIO mode (BOARD/BCM)
ros2 param set /motor_controller pin_mode BOARD
```

### Camera Node Parameters

```bash
# Set camera resolution (native sensor resolution)
ros2 param set /camera_node camera_width 1280
ros2 param set /camera_node camera_height 720

# Set output resolution (published image size)
ros2 param set /camera_node output_width 640
ros2 param set /camera_node output_height 480

# Set frame rate
ros2 param set /camera_node framerate 30

# Set flip method (0=none, 1=90°CW, 2=180°, 3=90°CCW)
ros2 param set /camera_node flip_method 2

# Change camera ID (for multiple cameras)
ros2 param set /camera_node camera_id 0
```

### Joystick Controller Parameters

```bash
# Set maximum speeds
ros2 param set /joystick_controller max_linear_speed 1.0
ros2 param set /joystick_controller max_angular_speed 2.0

# Adjust deadzone sensitivity
ros2 param set /joystick_controller deadzone 0.1
```

## 📁 Project Structure

```
car_ws/
├── src/
│   ├── car_bringup/              # Launch files and system integration
│   │   ├── car_bringup/
│   │   │   └── __init__.py
│   │   ├── launch/
│   │   │   ├── car_manual_control.launch.py # Motor + Joystick launch
│   │   │   ├── car_full_system.launch.py    # Complete system with camera
│   │   │   └── car_camera.launch.py         # Camera system only
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── test/
│   ├── car_control/              # Control algorithms (future)
│   │   ├── car_control/
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   └── setup.py
│   ├── car_description/          # Robot description files (future)
│   │   ├── car_description/
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   └── setup.py
│   ├── car_drivers/              # Hardware interface nodes
│   │   ├── car_drivers/
│   │   │   ├── motor_controller.py    # Main motor control
│   │   │   ├── motor_test.py          # Motor testing utilities
│   │   │   ├── debug.py               # Debug utilities
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── test/
│   ├── car_msgs/                 # Custom message definitions (future)
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── include/
│   │   └── src/
│   ├── car_navigation/           # Path planning (future)
│   │   ├── car_navigation/
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   └── setup.py
│   ├── car_perception/           # Computer vision and sensors
│   │   ├── car_perception/
│   │   │   ├── camera_node.py         # CSI camera with GStreamer
│   │   │   ├── image_viewer.py        # Camera feed viewer
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── test/
│   ├── car_teleop/               # Manual control
│   │   ├── car_teleop/
│   │   │   ├── joystick_controller.py # Joystick input handling
│   │   │   ├── cmd_relay.py           # Command routing
│   │   │   └── __init__.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── test/
│   └── vision_opencv/            # OpenCV integration
│       ├── cv_bridge/            # ROS-OpenCV bridge
│       ├── image_geometry/       # Camera geometry utilities
│       ├── opencv_tests/         # OpenCV test utilities
│       └── vision_opencv/        # Meta-package
├── build/                        # Build artifacts (auto-generated)
├── install/                      # Installation files (auto-generated)
├── log/                          # Build and runtime logs
├── images/                       # Project documentation images
│   ├── 1.jpeg
│   ├── 2.jpeg
│   ├── 3.jpeg
│   └── 4.jpeg
├── env.yaml                      # Environment configuration
├── README.md
└── tree.txt                      # Project structure reference
```

## 🔍 Monitoring and Debugging

### Check System Status

```bash
# List active nodes
ros2 node list

# Monitor topic data
ros2 topic echo /cmd_vel

# Check topic frequencies
ros2 topic hz /joy
ros2 topic hz /camera/image_raw

# View node information
ros2 node info /motor_controller
ros2 node info /camera_node
```

### Debug Camera Issues

```bash
# Test camera with GStreamer directly
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv flip-method=2 ! xvimagesink

# Check available camera modes
v4l2-ctl --list-formats-ext

# Monitor camera topics
ros2 topic list | grep camera
ros2 topic info /camera/image_raw

# Check camera parameters
ros2 param list /camera_node
```

### Debug Joystick Issues

```bash
# Check joystick device
ls /dev/input/js*

# Test joystick directly
jstest /dev/input/js0

# Monitor joystick data
ros2 topic echo /joy
```

### Debug Motor Issues

```bash
# Check GPIO permissions
ls -la /dev/gpiomem

# Test motor parameters
ros2 param list /motor_controller

# Monitor motor commands
ros2 topic echo /cmd_vel
```

## 🚧 Troubleshooting

### Common Issues

**Camera not working:**
```bash
# Check if camera is detected
ls /dev/video*

# Test basic camera functionality
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink

# Check for CSI connection issues
dmesg | grep -i camera
```

**Camera pipeline errors:**
```bash
# Try different resolutions
ros2 run car_perception camera_node --ros-args -p camera_width:=1920 -p camera_height:=1080

# Disable flip if causing issues
ros2 run car_perception camera_node --ros-args -p flip_method:=0

# Use fallback pipeline
ros2 run car_perception camera_node --ros-args -p camera_id:=0
```

**cv_bridge import errors:**
```bash
# Install cv_bridge
sudo apt install ros-foxy-cv-bridge python3-opencv

# Rebuild workspace
cd ~/car_ws
colcon build --packages-select car_perception
```

**Joystick not detected:**
```bash
# Reconnect Bluetooth joystick
sudo bluetoothctl
scan on
pair XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
```

**GPIO permission denied:**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Reboot required
```

**PWM pins not working:**
```bash
# Enable PWM in device tree
sudo /opt/nvidia/jetson-io/jetson-io.py
# Configure for compatible hardware and enable PWM functions
sudo reboot
```

**Motors not responding:**
- Check motor driver connections to correct pins (15, 16, 18, 32, 35, 37)
- Verify power supply (motors need adequate current)
- Confirm 4-motor setup: 2 motors per side connected in parallel
- Check minimum speed threshold (0.4 m/s required for movement)

**Build errors:**
```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Motor control implementation with 4-motor differential drive
- [x] Joystick teleoperation with minimum speed threshold
- [x] Safety systems (emergency stop, slow mode, speed limiting)
- [x] Complete ROS2 architecture with launch files
- [x] Hardware PWM configuration on Jetson Xavier NX

### Phase 2: Vision ✅
- [x] CSI camera integration with GStreamer
- [x] Hardware-accelerated image pipeline
- [x] Real-time camera feed at 30fps
- [x] Image processing foundation
- [ ] Lane detection algorithms
- [ ] Object detection

### Phase 3: Autonomy
- [ ] PID-based control
- [ ] Path planning
- [ ] Obstacle avoidance
- [ ] SLAM integration

### Phase 4: Advanced Features
- [ ] Web-based control interface
- [ ] Machine learning integration
- [ ] Multi-sensor fusion
- [ ] Fleet management

## 🎯 Camera Configuration Examples

### CSI Camera Modes Available

Your Jetson Xavier NX CSI camera supports these modes:
- **3280 x 2464** @ 21fps (Max resolution)
- **3280 x 1848** @ 28fps (Wide format)
- **1920 x 1080** @ 30fps (Full HD)
- **1640 x 1232** @ 30fps (4:3 format)
- **1280 x 720** @ 60fps (HD, high framerate)

### Recommended Camera Settings

**For development/testing:**
```bash
ros2 run car_perception camera_node --ros-args \
  -p camera_width:=1280 -p camera_height:=720 \
  -p output_width:=640 -p output_height:=480 \
  -p framerate:=30 -p flip_method:=2
```

**For high quality:**
```bash
ros2 run car_perception camera_node --ros-args \
  -p camera_width:=1920 -p camera_height:=1080 \
  -p output_width:=960 -p output_height:=540 \
  -p framerate:=30 -p flip_method:=2
```

**For maximum performance:**
```bash
ros2 run car_perception camera_node --ros-args \
  -p camera_width:=1280 -p camera_height:=720 \
  -p output_width:=320 -p output_height:=240 \
  -p framerate:=30 -p flip_method:=2
```

## 🚀 Launch Commands

### Launch System
```bash
# Launch complete system (motors + camera + joystick)
ros2 launch car_bringup car_full_system.launch.py

# Launch camera system only (camera + viewer)
ros2 launch car_bringup car_camera.launch.py

# Launch with custom camera settings
ros2 launch car_bringup car_full_system.launch.py \
  camera_width:=1280 camera_height:=720 \
  output_width:=640 output_height:=480

# Launch manual control only (no camera)
ros2 launch car_bringup car_manual_control.launch.py

# Launch with performance settings
ros2 launch car_bringup car_full_system.launch.py \
  max_linear_speed:=0.8 base_speed_scale:=80 \
  camera_width:=1920 camera_height:=1080 \
  enable_image_viewer:=true
```

### Monitor Launch
```bash
# Check all nodes are running
ros2 node list

# Monitor topics
ros2 topic list

# Check camera feed
ros2 topic hz /camera/image_raw

# View live camera feed
ros2 run car_perception image_viewer

# Or directly with Python
python3 src/car_perception/car_perception/image_viewer.py
```

### Stop System
```bash
# Stop all nodes
Ctrl+C in the launch terminal

# Or kill specific processes
pkill -f "ros2 launch"
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow ROS2 coding standards
- Add comprehensive documentation
- Include unit tests where applicable
- Test on actual hardware before submitting
- Test camera functionality on Jetson Xavier NX

## 📚 Documentation

- [ROS2 Foxy Documentation](https://docs.ros.org/en/foxy/)
- [Jetson Xavier NX Developer Guide](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **NVIDIA** for the Jetson Xavier NX platform and GStreamer integration
- **Open Robotics** for ROS2 framework
- **Raspberry Pi Foundation** for GPIO libraries
- The **open-source robotics community**

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/autonomous-car-jetson-nx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autonomous-car-jetson-nx/discussions)
- **Email**: your.email@example.com

---

**⭐ If this project helps you, please give it a star on GitHub!**

Made with ❤️ for the robotics community