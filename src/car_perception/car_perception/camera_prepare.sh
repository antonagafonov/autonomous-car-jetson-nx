#!/bin/bash
# Camera preparation script for Jetson Xavier NX
# Run this before starting ROS camera node if having issues

echo "Preparing camera for ROS use..."

# Kill any processes that might be using the camera
echo "Checking for processes using camera..."
sudo pkill -f nvargus || true
sudo pkill -f gst-launch || true
sudo pkill -f opencv || true

# Wait for processes to terminate
sleep 2

# Restart nvargus daemon
echo "Restarting nvargus daemon..."
sudo systemctl restart nvargus-daemon

# Wait for daemon to be ready
sleep 3

# Check camera availability
echo "Checking camera availability..."
if [ -e /dev/video0 ]; then
    echo "✓ /dev/video0 exists"
    ls -la /dev/video0
else
    echo "✗ /dev/video0 not found"
fi

# Check nvargus daemon status
if systemctl is-active --quiet nvargus-daemon; then
    echo "✓ nvargus daemon is running"
else
    echo "✗ nvargus daemon is not running"
    sudo systemctl status nvargus-daemon
fi

# Test camera briefly
echo "Testing camera briefly..."
python3 -c "
import cv2
gst_pipeline = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✓ Camera test successful')
    else:
        print('✗ Camera opened but failed to capture')
    cap.release()
else:
    print('✗ Camera test failed to open')
"

echo "Camera preparation complete!"
echo "You can now start the ROS camera node."