# 🚗 Autonomous Car Project 
# Jetson Xavier NX - ROS2 - Pytorch

![ROS2](https://img.shields.io/badge/ROS2-Foxy-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Xavier%20NX-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

A comprehensive autonomous car project built with ROS2 Foxy on NVIDIA Jetson Xavier NX. Features real-time joystick control, differential drive motor control, and expandable architecture for autonomous navigation capabilities.

## 🎯 Project Overview

This project transforms a basic RC car into an intelligent autonomous vehicle using:
- **NVIDIA Jetson Xavier NX** for high-performance edge computing
- **ROS2 Foxy** for robust robotics software architecture
- **Differential drive control** for precise movement
- **Bluetooth joystick** for intuitive manual control
- **Modular design** ready for camera integration and autonomous features

## ✨ Features

- ✅ **Manual Control**: Bluetooth joystick teleoperation with Xbox/PS4 controller support
- ✅ **Motor Control**: Precise PWM-based differential drive control
- ✅ **Safety Systems**: Emergency stop, speed limiting, and mode switching
- ✅ **Real-time Performance**: Low-latency control loop for responsive driving
- ✅ **Modular Architecture**: Clean ROS2 package structure for easy expansion
- 🚧 **Camera Integration**: CSI camera support (planned)
- 🚧 **Lane Detection**: OpenCV-based computer vision (planned)
- 🚧 **Autonomous Navigation**: PID control and path planning (planned)

## 🔧 Hardware Requirements

### Core Components
- **NVIDIA Jetson Xavier NX** Developer Kit
- **Dual DC Motors** with H-bridge motor driver
- **Bluetooth Joystick** (Xbox One, PS4, or compatible)
- **Power Supply** (12V for motors, 5V for Jetson)

### Optional Components
- **CSI Camera** or USB webcam
- **IMU Sensor** for orientation tracking
- **Ultrasonic Sensors** for obstacle detection
- **Servo Motor** for camera gimbal

## 📋 Pin Configuration

### Jetson Xavier NX GPIO (BOARD numbering)

| Component | Pin | Function |
|-----------|-----|----------|
| **Motor A (Left)** | 18, 16 | Direction Control |
| | 33 | PWM Speed Control |
| **Motor B (Right)** | 37, 35 | Direction Control |
| | 32 | PWM Speed Control |

> **Note**: Pins 33 and 32 are hardware PWM-capable pins on the Jetson Xavier NX

## 🚀 Installation

### Prerequisites

```bash
# Install ROS2 Foxy (if not already installed)
sudo apt update
sudo apt install ros-foxy-desktop

# Install joystick support
sudo apt install ros-foxy-joy ros-foxy-teleop-twist-joy

# Install GPIO library
sudo apt install python3-rpi.gpio

# Install development tools
sudo apt install python3-colcon-common-extensions
```

### Setup GPIO Permissions

```bash
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

## 🎮 Usage

### Quick Start - Manual Control

**Terminal 1: Motor Controller**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_drivers motor_controller
```

**Terminal 2: Joystick Input**
```bash
source ~/car_ws/install/setup.bash
ros2 run joy joy_node
```

**Terminal 3: Joystick Controller**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_teleop joystick_controller
```

**Terminal 4: Command Relay**
```bash
source ~/car_ws/install/setup.bash
ros2 run car_teleop cmd_relay
```

### Joystick Controls

| Control | Action | Description |
|---------|--------|-------------|
| **Left Stick ↕** | Forward/Backward | Move car forward or reverse |
| **Left Stick ↔** | Turn Left/Right | Steer the car |
| **A Button** | Mode Toggle | Switch between manual/autonomous |
| **B Button** | Emergency Stop | Immediate stop with toggle |
| **LB/L1** | Slow Mode | Reduce speed to 30% |
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
| `/camera/image_raw` | `sensor_msgs/Image` | Camera feed (planned) |

### Nodes

| Node | Package | Function |
|------|---------|----------|
| `motor_controller` | `car_drivers` | GPIO motor control |
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
│   ├── car_drivers/              # Hardware interface nodes
│   │   ├── car_drivers/
│   │   │   ├── motor_controller.py    # Main motor control
│   │   │   └── camera_node.py         # Camera interface
│   │   ├── package.xml
│   │   └── setup.py
│   ├── car_teleop/               # Manual control
│   │   ├── car_teleop/
│   │   │   ├── joystick_controller.py # Joystick input handling
│   │   │   └── cmd_relay.py           # Command routing
│   │   ├── package.xml
│   │   └── setup.py
│   ├── car_control/              # Control algorithms (planned)
│   │   ├── car_control/
│   │   │   ├── pid_controller.py      # PID implementation
│   │   │   └── lane_follower.py       # Autonomous control
│   │   └── ...
│   ├── car_perception/           # Computer vision (planned)
│   ├── car_navigation/           # Path planning (planned)
│   └── car_bringup/             # Launch files (planned)
├── README.md
├── LICENSE
└── .gitignore
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

# View node information
ros2 node info /motor_controller
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

**Motors not responding:**
- Check motor driver connections
- Verify power supply (motors need adequate current)
- Confirm pin assignments match your hardware

**Build errors:**
```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build --symlink-install
```

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Motor control implementation
- [x] Joystick teleoperation
- [x] Safety systems
- [x] Basic ROS2 architecture

### Phase 2: Vision (In Progress)
- [ ] CSI camera integration
- [ ] Image processing pipeline
- [ ] Lane detection algorithms
- [ ] Object detection

### Phase 3: Autonomy
- [ ] PID-based control
- [ ] Path planning
- [ ] Obstacle avoidance



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

## 📚 Documentation

- [ROS2 Foxy Documentation](https://docs.ros.org/en/foxy/)
- [Jetson Xavier NX Developer Guide](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit)
- [RPi.GPIO Documentation](https://sourceforge.net/p/raspberry-gpio-python/wiki/Home/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **NVIDIA** for the Jetson Xavier NX platform
- **Open Robotics** for ROS2 framework
- **Raspberry Pi Foundation** for GPIO libraries
- The **open-source robotics community**

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/autonomous-car-jetson-nx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autonomous-car-jetson-nx/discussions)
- **Email**: fake@email.com

---

**⭐ If this project helps you, please give it a star on GitHub!**

Made with ❤️ for the robotics community

# GPIO PinOut

https://jetsonhacks.com/nvidia-jetson-xavier-nx-gpio-header-pinout/
```
3V3   (1)  (2)   5V
GPIO2  (3)  (4)   5V
GPIO3  (5)  (6)   GND
GPIO4  (7)  (8)   GPIO14
GND   (9) (10)   GPIO15
GPIO17 (11) (12)   GPIO18    ← Motor A Direction 1
GPIO27 (13) (14)   GND
GPIO22 (15) (16)   GPIO23    ← Motor A Direction 2
3V3  (17) (18)   GPIO24
GPIO10 (19) (20)   GND
GPIO9 (21) (22)   GPIO25
GPIO11 (23) (24)   GPIO8
GND  (25) (26)   GPIO7
GPIO0 (27) (28)   GPIO1
GPIO5 (29) (30)   GND
GPIO6 (31) (32)   GPIO12    ← Motor B PWM (ENB)
GPIO13 (33) (34)   GND       ← Motor A PWM (ENA) at pin 33
GPIO19 (35) (36)   GPIO16    ← Motor B Direction 2 at pin 35
GPIO26 (37) (38)   GPIO20    ← Motor B Direction 1 at pin 37
GND  (39) (40)   GPIO21
```
