# Camera Debug Solution for Jetson Xavier NX

## Overview

This document provides a systematic approach to debug camera issues on Jetson Xavier NX when using CSI cameras with OpenCV and ROS 2. The most common issue is when GStreamer works from command line but OpenCV fails to capture frames in Python scripts.

## Quick Diagnosis Checklist

- [ ] GStreamer pipeline works from command line
- [ ] OpenCV built with GStreamer support
- [ ] nvargus-daemon is running
- [ ] Sufficient swap space (6-8GB)
- [ ] Camera physically connected
- [ ] No resource conflicts

## Prerequisites Check

### 1. System Information
```bash
# Install and run jtop for system overview
pip3 install jtop
sudo apt install python3-smbus
jtop
```

**Expected Values:**
- Platform: NVIDIA Jetson Xavier NX
- JetPack: 5.1.3
- Python: 3.8.10
- CUDA Arch BIN: 7.2
- OpenCV: YES (this is what we're fixing if it shows NO/MISSING)

### 2. Check Swap Space
```bash
# Check current swap
free -h

# If less than 6GB, create swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Verify nvargus-daemon
```bash
# Check daemon status
sudo systemctl status nvargus-daemon

# Restart if needed
sudo systemctl restart nvargus-daemon
```

## Step-by-Step Debugging

### Step 1: Test GStreamer Pipeline (Hardware Verification)

```bash
# Basic camera test
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
'video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1' ! \
nvvidconv ! nveglglessink

# If above fails, try lower resolution
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
'video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1' ! \
nvvidconv ! nveglglessink
```

**Expected:** Live camera window opens

**Troubleshooting:**
- No window: Check physical camera connection, try different sensor-id (0, 1, etc.)
- Plugin errors: Verify JetPack installation
- Resource busy: Reboot system

### Step 2: Verify OpenCV GStreamer Support

```bash
# Check if OpenCV has GStreamer support
/usr/bin/python3.8 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
```

**Expected Output:**
```
GStreamer:                   YES (1.16.3)
```

**If NO or empty:** OpenCV needs to be recompiled with GStreamer support.

### Step 3: OpenCV Compilation (If GStreamer Support Missing)

```bash
# Remove existing OpenCV installations
sudo apt-get purge libopencv*
sudo apt-get purge python3-opencv
pip3 uninstall opencv-python opencv-contrib-python

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libjpeg8-dev libtiff5-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y python3-dev python3-numpy
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libtbb2 libtbb-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2

# Clone OpenCV
cd ~/
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.9.0
cd ../opencv_contrib
git checkout 4.9.0

# Build OpenCV
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -D WITH_OPENCL=OFF \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=7.2 \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUDNN=ON \
      -D WITH_CUBLAS=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_NEON=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENMP=ON \
      -D BUILD_TIFF=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_TBB=ON \
      -D BUILD_TBB=ON \
      -D BUILD_TESTS=OFF \
      -D WITH_EIGEN=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D PYTHON3_EXECUTABLE=/usr/bin/python3.8 \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
      -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.8/dist-packages \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF ..

# Verify GStreamer is enabled in CMake output
# Look for: GStreamer: YES

# Compile (this takes 2-4 hours)
make -j$(nproc)

# Install
sudo make install
sudo ldconfig

# Reboot
sudo reboot
```

### Step 4: Test Python OpenCV Camera Access

Create `test_camera.py`:

```python
#!/usr/bin/env python3

import cv2
import sys
import time

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
    sensor_id=0
):
    return (
        "nvarguscamerasrc sensor-id={} ! "
        "video/x-raw(memory:NVMM), width=(int){}, height=(int){}, framerate=(fraction){}/1, format=(string)NV12 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        .format(
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def test_camera():
    print("OpenCV Version:", cv2.__version__)
    print("Testing GStreamer support...")
    
    # Check GStreamer support
    build_info = cv2.getBuildInformation()
    if "GStreamer" in build_info and "YES" in build_info:
        print("✓ GStreamer support detected")
    else:
        print("✗ GStreamer support NOT detected")
        print("OpenCV needs to be recompiled with GStreamer support")
        return False
    
    pipeline = gstreamer_pipeline()
    print(f"\nTesting pipeline:\n{pipeline}\n")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("✗ ERROR: Failed to open camera with OpenCV!")
        print("\nTroubleshooting steps:")
        print("1. Check camera physical connection")
        print("2. Verify nvargus-daemon is running: sudo systemctl status nvargus-daemon")
        print("3. Try rebooting to clear resource locks")
        print("4. Test with different sensor-id (0, 1, etc.)")
        return False
    
    print("✓ Camera opened successfully!")
    print("Reading frames (Press 'q' to quit)...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ ERROR: Failed to grab frame")
                break
            
            if frame is None or frame.size == 0:
                print("⚠ WARNING: Empty frame received")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Display FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frames: {frame_count}, FPS: {fps:.2f}")
            
            cv2.imshow("Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released successfully")
        return True

if __name__ == "__main__":
    print("=" * 50)
    print("CAMERA DEBUG TEST")
    print("=" * 50)
    
    success = test_camera()
    
    if success:
        print("\n✓ CAMERA TEST PASSED!")
        print("Your camera setup is working correctly.")
    else:
        print("\n✗ CAMERA TEST FAILED!")
        print("Follow the troubleshooting steps above.")
    
    sys.exit(0 if success else 1)
```

Run the test:
```bash
chmod +x test_camera.py
/usr/bin/python3.8 test_camera.py
```

### Step 5: ROS 2 Integration

#### Check ROS 2 Environment

```bash
# Source your workspace
source ~/car_ws/install/setup.bash

# Verify OpenCV in ROS 2 environment
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
```

#### Create Symlink (if needed)

```bash
# Find installed cv2.so
INSTALLED_CV2_SO=$(find /usr/local/lib/python3.8/ -name "cv2.cpython-38-aarch64-linux-gnu.so" 2>/dev/null)
echo "Installed cv2.so: $INSTALLED_CV2_SO"

# Find ROS 2 site-packages
ROS2_VENV_SITE_PACKAGES=$(find ~/car_ws/install/ -path "*python3.8/site-packages" 2>/dev/null | head -n 1)
echo "ROS 2 site-packages: $ROS2_VENV_SITE_PACKAGES"

# Create symlink if both paths exist
if [ -n "$INSTALLED_CV2_SO" ] && [ -n "$ROS2_VENV_SITE_PACKAGES" ]; then
    echo "Creating symlink..."
    sudo ln -sf "$INSTALLED_CV2_SO" "$ROS2_VENV_SITE_PACKAGES/cv2.so"
    echo "✓ Symlink created"
else
    echo "✗ Cannot create symlink - check paths"
fi

# Rebuild workspace
cd ~/car_ws
colcon build --symlink-install --continue-on-error
```

#### Test ROS 2 Camera Node

```bash
# Run with debug logging
source ~/car_ws/install/setup.bash
ros2 run your_package_name camera_node --ros-args --log-level camera_node:=debug

# In another terminal, check topics
ros2 topic list | grep camera
ros2 topic info /camera/image_raw
ros2 topic echo /camera/image_raw --once

# Visual check (if GUI available)
rqt_image_view
```

## Common Issues and Solutions

### Issue: "No such element 'nvarguscamerasrc'"
**Solution:** JetPack installation incomplete. Re-flash or reinstall JetPack.

### Issue: Camera opens but black/frozen frames
**Solution:** 
1. Reboot system
2. Check if another process is using camera: `sudo lsof | grep video`
3. Try different pipeline parameters

### Issue: "Resource busy" error
**Solution:**
1. Reboot system
2. Kill processes: `sudo pkill -f nvargus`
3. Restart daemon: `sudo systemctl restart nvargus-daemon`

### Issue: ROS 2 can't find cv2 module
**Solution:** Create symlink as shown in Step 5

### Issue: Low FPS or dropped frames
**Solution:**
1. Reduce capture resolution
2. Increase buffer size in pipeline
3. Check system load with `jtop`

## Debug Scripts

Save these scripts in your repository for quick debugging:

### `scripts/check_system.sh`
```bash
#!/bin/bash
echo "=== SYSTEM CHECK ==="
echo "Platform: $(cat /proc/device-tree/model)"
echo "JetPack: $(dpkg -l | grep nvidia-jetpack | awk '{print $3}')"
echo "Python: $(python3 --version)"
echo "Swap: $(free -h | grep Swap)"
echo "nvargus-daemon: $(systemctl is-active nvargus-daemon)"
echo "OpenCV GStreamer: $(python3 -c 'import cv2; print(cv2.getBuildInformation())' | grep GStreamer | head -1)"
```

### `scripts/test_gstreamer.sh`
```bash
#!/bin/bash
echo "Testing GStreamer pipeline..."
timeout 10 gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
'video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1' ! \
nvvidconv ! fakesink
echo "GStreamer test completed"
```

Make them executable:
```bash
chmod +x scripts/*.sh
```

## Support

If issues persist after following this guide:

1. Check Jetson forums: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
2. Verify hardware with different camera if available
3. Consider re-flashing JetPack
4. Check for hardware defects

---

**Last Updated:** $(date)
**Compatible with:** JetPack 5.1.3, Ubuntu 20.04, Python 3.8.10