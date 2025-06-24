# 🚗 Autonomous Car Project - Jetson Xavier NX

![ROS2](https://img.shields.io/badge/ROS2-Foxy-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Xavier%20NX-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

A comprehensive autonomous car project built with ROS2 Foxy on NVIDIA Jetson Xavier NX. Features real-time joystick control, differential drive motor control, CSI camera integration, **bag data collection**, and expandable architecture for autonomous navigation capabilities.

![Autonomous Car](images/4.jpeg)
*The completed autonomous car with Jetson Xavier NX, 4-motor differential drive, CSI camera, and Bluetooth joystick control*

## 🎯 Project Overview

This project transforms a basic RC car into an intelligent autonomous vehicle using:
- **NVIDIA Jetson Xavier NX** for high-performance edge computing
- **ROS2 Foxy** for robust robotics software architecture
- **Differential drive control** for precise movement
- **CSI Camera integration** with hardware-accelerated pipeline
- **Bluetooth joystick** for intuitive manual control
- **Data collection system** for machine learning and behavior cloning
- **Modular design** ready for autonomous navigation features

## ✨ Features

- ✅ **Manual Control**: Bluetooth joystick teleoperation with Xbox/PS4 controller support
- ✅ **Motor Control**: Precise PWM-based differential drive control with 4-motor setup
- ✅ **Camera Integration**: CSI camera with GStreamer hardware acceleration and 180° flip
- ✅ **Real-time Vision**: Live camera feed publishing to ROS2 topics at 30fps
- ✅ **Data Collection**: Comprehensive bag recording and extraction for ML training
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

# Install data collection dependencies
sudo apt install ros-foxy-rosbag2* ros-foxy-image-view
pip3 install zstandard  # For compressed bag support
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
git clone https://github.com/antonagafonov/autonomous-car-jetson-nx.git src/

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

### 🎒 Data Collection for Machine Learning

The project includes a comprehensive data collection system for training autonomous driving models using behavior cloning and imitation learning.

#### Record Training Data

**Start recording while driving manually:**
```bash
# Launch car with camera (required for ML data)
ros2 launch car_bringup car_full_system.launch.py

# In another terminal, start recording
mkdir -p ~/training_data
cd ~/training_data

# Record all topics for comprehensive dataset
ros2 bag record -o drive_session_001 \
  /camera/image_raw \
  /cmd_vel \
  /cmd_vel_manual \
  /joy \
  /autonomous_mode

# Or record specific topics only
ros2 bag record -o drive_session_001 \
  /camera/image_raw /cmd_vel_manual
```

**Record with automatic naming:**
```bash
# Create session with timestamp
SESSION_NAME="drive_$(date +%Y%m%d_%H%M%S)"
ros2 bag record -o $SESSION_NAME \
  /camera/image_raw /cmd_vel_manual /joy
```

#### Extract Training Data

Use the integrated bag data extractor to convert ROS2 bags into ML-ready format:

```bash
# Extract from bag directory
python3 bag_collect.py ~/training_data/drive_session_001

# Extract from specific .db3 file
python3 bag_collect.py ~/training_data/drive_session_001/rosbag2_2024_01_15-10_30_00_0.db3

# Specify custom output directory
python3 bag_collect.py ~/training_data/drive_session_001 -o ~/ml_datasets/session_001

# Extract multiple sessions
for bag in ~/training_data/drive_session_*; do
    python3 bag_collect.py "$bag"
done
```

#### Extracted Data Structure

The extractor creates an organized dataset:

```
drive_session_001_extracted/
├── images/               # Camera images for visual input
│   ├── image_000000.png  # Sequential PNG images (640x480)
│   ├── image_000001.png
│   └── ...
├── data/                 # Command and sensor data
│   ├── images.csv        # Image metadata with timestamps
│   ├── cmd_vel.csv       # Motor commands (actual outputs)
│   ├── cmd_vel_manual.csv # Manual commands (training labels)
│   ├── joy.csv           # Raw joystick input
│   └── *.json           # Same data in JSON format
└── metadata/
    └── extraction_summary.json  # Dataset statistics
```

#### Data Format Details

**Images (PNG files):**
- Format: 640x480 BGR images ready for OpenCV/ML frameworks
- Naming: Sequential `image_XXXXXX.png`
- Synchronized with command timestamps

**Command Data (CSV format):**
```csv
timestamp,linear_x,linear_y,linear_z,angular_x,angular_y,angular_z
1642248600123456789,0.2,0.0,0.0,0.0,0.0,0.1
```

**Image Metadata (CSV format):**
```csv
filename,timestamp,width,height,encoding,frame_id,seq
image_000000.png,1642248600123456789,640,480,bgr8,camera_link,0
```

#### ML Training Ready

The extracted data is ready for:
- **Behavior Cloning**: Use `cmd_vel_manual` as labels, images as input
- **Imitation Learning**: Train neural networks to mimic human driving
- **Computer Vision**: Lane detection, object recognition training
- **Reinforcement Learning**: State-action pairs for training

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

### Data Collection Tools

| Tool | Function |
|------|----------|
| `ros2 bag record` | Record ROS2 topics to bag files |
| `bag_collect.py` | Extract and organize bag data for ML |
| **Bag Data Extractor** | Convert bags to training-ready datasets |

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

### Data Collection Parameters

```bash
# Set recording quality vs storage trade-off
ros2 param set /camera_node output_width 320   # Lower for storage
ros2 param set /camera_node output_width 640   # Standard quality
ros2 param set /camera_node output_width 960   # High quality

# Adjust recording frequency
ros2 topic hz /camera/image_raw               # Check current rate
ros2 param set /camera_node framerate 15     # Lower for storage
ros2 param set /camera_node framerate 30     # Standard rate
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
├── bag_collect.py               # **NEW: ML data extraction tool**
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

### Debug Data Collection

```bash
# Check bag recording status
ros2 bag info <bag_directory>

# List available topics for recording
ros2 topic list

# Monitor recording rate
ros2 topic hz /camera/image_raw
watch "ros2 topic hz /cmd_vel_manual"

# Check bag file size during recording
ls -lh ~/training_data/

# Test data extraction
python3 bag_collect.py --help
python3 bag_collect.py <test_bag> -o /tmp/test_extraction
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

**Data collection issues:**
```bash
# Check available disk space
df -h

# Verify bag recording permissions
ls -la ~/training_data/

# Test bag extraction on small file first
ros2 bag record -o test_bag /joy --duration 10
python3 bag_collect.py test_bag

# Check zstandard installation for compressed bags
python3 -c "import zstandard; print('zstandard OK')"
```

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

### Phase 3: Data Collection ✅
- [x] **ROS2 bag recording system**
- [x] **Integrated data extraction tool**
- [x] **ML-ready dataset organization**
- [x] **Image and command synchronization**
- [x] **Support for compressed bags**

### Phase 4: Machine Learning 🚧
- [ ] **Behavior cloning implementation**
- [ ] **Neural network training pipeline**
- [ ] **Imitation learning framework**
- [ ] **End-to-end driving model**

### Phase 5: Autonomy
- [ ] PID-based control
- [ ] Path planning
- [ ] Obstacle avoidance
- [ ] SLAM integration

### Phase 6: Advanced Features
- [ ] Web-based control interface
- [ ] Multi-sensor fusion
- [ ] Fleet management
- [ ] Model deployment optimization

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

**For data collection (recommended):**
```bash
ros2 run car_perception camera_node --ros-args \
  -p camera_width:=1280 -p camera_height:=720 \
  -p output_width:=640 -p output_height:=480 \
  -p framerate:=30 -p flip_method:=2
```

## 🎓 Machine Learning Workflow

### 1. Data Collection
```bash
# Start car system
ros2 launch car_bringup car_full_system.launch.py

# Record training session
mkdir -p ~/ml_data/training_sessions
cd ~/ml_data/training_sessions

# Record comprehensive dataset
ros2 bag record -o session_$(date +%H%M%S) \
  /camera/image_raw /cmd_vel_manual /joy

# Drive manually for 10-30 minutes with varied scenarios:
# - Straight driving
# - Turning left/right
# - Different speeds
# - Various lighting conditions
```

### 2. Data Extraction
```bash
# Extract all sessions
for session in ~/ml_data/training_sessions/session_*; do
    echo "Extracting $session..."
    python3 bag_collect.py "$session"
done

# Verify extracted data
ls ~/ml_data/training_sessions/session_*_extracted/
```

### 3. Data Analysis
```bash
# Check dataset statistics
python3 -c "
import json
with open('session_123456_extracted/metadata/extraction_summary.json') as f:
    summary = json.load(f)
    print(f'Duration: {summary[\"timing_analysis\"][\"duration_seconds\"]:.1f}s')
    print(f'Images: {summary[\"data_summary\"][\"total_images\"]}')
    print(f'Commands: {summary[\"data_summary\"][\"total_cmd_vel_manual\"]}')
    print(f'Frequency: {summary[\"timing_analysis\"][\"command_frequency\"]:.1f} Hz')
"
```

### 4. Training Preparation
The extracted data is ready for popular ML frameworks:

**PyTorch Example:**
```python
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.commands = pd.read_csv(f"{data_dir}/data/cmd_vel_manual.csv")
        self.images = pd.read_csv(f"{data_dir}/data/images.csv")