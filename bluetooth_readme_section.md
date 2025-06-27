## 📶 Bluetooth Configuration

### Wireless Controller Setup

The project supports Bluetooth joystick control for manual operation. Follow these steps to connect your wireless controller.

#### Supported Controllers
- **Xbox One/Series Controllers** - Full support with haptic feedback
- **PlayStation 4/5 Controllers** - Complete button mapping
- **Nintendo Pro Controller** - Basic functionality
- **Generic Bluetooth Controllers** - May require button remapping

#### Pairing Process (Headless/SSH)

**1. Start Bluetooth service:**
```bash
sudo systemctl start bluetooth
sudo systemctl enable bluetooth
```

**2. Enter pairing mode:**
```bash
bluetoothctl
power on
agent on
default-agent
discoverable on
pairable on
scan on
```

**3. Put controller in pairing mode:**
- **Xbox**: Hold Xbox + pairing button until light flashes rapidly
- **PS4/PS5**: Hold Share + PS button until light bar flashes
- **Nintendo Pro**: Hold sync button until lights scroll

**4. Pair the controller:**
```bash
# Look for your controller in scan results
# Example: [NEW] Device 0C:67:94:01:88:E8 Wireless Controller

pair 0C:67:94:01:88:E8
trust 0C:67:94:01:88:E8
connect 0C:67:94:01:88:E8
quit
```

#### Auto-Connect on Boot

Create a systemd service for automatic controller connection:

```bash
# Create auto-connect service
sudo nano /etc/systemd/system/controller-autoconnect.service
```

Add this content (replace MAC address with your controller's):
```ini
[Unit]
Description=Auto-connect Bluetooth Controller
After=bluetooth.service
Wants=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/bin/bluetoothctl connect 0C:67:94:01:88:E8
RemainAfterExit=yes
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl enable controller-autoconnect.service
sudo systemctl start controller-autoconnect.service
```

#### Verify Connection

```bash
# Check connected devices
bluetoothctl devices Connected

# Test controller input
sudo apt install joystick
jstest /dev/input/js0

# Monitor joystick data in ROS2
ros2 topic echo /joy
```

#### Troubleshooting

**Controller not connecting:**
```bash
# Restart Bluetooth service
sudo systemctl restart bluetooth

# Check Bluetooth status
sudo systemctl status bluetooth

# Manual connection attempt
bluetoothctl connect [MAC_ADDRESS]
```

**Xbox Controller specific issues:**
```bash
# Install xpadneo driver for better Xbox support
sudo apt update
sudo apt install dkms git
git clone https://github.com/atar-axis/xpadneo.git
cd xpadneo
sudo ./install.sh
sudo reboot
```

**Permission issues:**
```bash
# Add user to input group
sudo usermod -a -G input $USER
sudo reboot
```

#### GUI Mode Bluetooth Setup

If you need to switch to GUI mode temporarily for easier Bluetooth setup:

```bash
# Switch to GUI mode
sudo systemctl set-default graphical.target
sudo reboot

# Use GUI Bluetooth manager, then switch back
sudo systemctl set-default multi-user.target
sudo reboot
```