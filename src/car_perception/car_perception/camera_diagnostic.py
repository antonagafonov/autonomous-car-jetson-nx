#!/usr/bin/env python3
"""
Camera diagnostic script to test CSI camera on Jetson Xavier NX
"""

import cv2
import subprocess
import os

def run_command(cmd):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_camera_devices():
    """Check available camera devices"""
    print("=== Checking Camera Devices ===")
    
    # Check for video devices
    video_devices = run_command("ls -la /dev/video*")
    print(f"Video devices: {video_devices}")
    
    # Check for CSI camera
    csi_info = run_command("dmesg | grep -i csi")
    print(f"CSI info: {csi_info}")
    
    # Check camera modules
    camera_modules = run_command("lsmod | grep -i camera")
    print(f"Camera modules: {camera_modules}")
    
    # Check nvargus daemon
    nvargus_status = run_command("ps aux | grep nvargus")
    print(f"Nvargus daemon: {nvargus_status}")

def test_gstreamer_pipelines():
    """Test different GStreamer pipelines"""
    print("\n=== Testing GStreamer Pipelines ===")
    
    # Pipeline 1: Your working cv2_cam.py pipeline
    pipeline1 = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    # Pipeline 2: Alternative with sensor-id
    pipeline2 = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    # Pipeline 3: Simplified version
    pipeline3 = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )
    
    pipelines = [
        ("Working cv2_cam.py pipeline", pipeline1),
        ("Alternative with sensor-id", pipeline2),
        ("Simplified pipeline", pipeline3)
    ]
    
    for name, pipeline in pipelines:
        print(f"\nTesting: {name}")
        print(f"Pipeline: {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("✓ Pipeline opened successfully")
                
                # Try to capture a frame
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Frame captured successfully - Shape: {frame.shape}")
                else:
                    print("✗ Failed to capture frame")
                
                cap.release()
            else:
                print("✗ Failed to open pipeline")
                
        except Exception as e:
            print(f"✗ Exception: {e}")

def test_regular_camera():
    """Test regular USB/built-in camera"""
    print("\n=== Testing Regular Camera ===")
    
    for i in range(5):  # Check video0 to video4
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✓ /dev/video{i} opened successfully")
                
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  Resolution: {width}x{height}, FPS: {fps}")
                
                # Try to capture a frame
                ret, frame = cap.read()
                if ret:
                    print(f"  ✓ Frame captured - Shape: {frame.shape}")
                else:
                    print("  ✗ Failed to capture frame")
                
                cap.release()
            else:
                print(f"✗ /dev/video{i} failed to open")
                
        except Exception as e:
            print(f"✗ /dev/video{i} exception: {e}")

def main():
    print("Camera Diagnostic Tool for Jetson Xavier NX")
    print("=" * 50)
    
    check_camera_devices()
    test_gstreamer_pipelines()
    test_regular_camera()
    
    print("\n=== Recommendations ===")
    print("1. If CSI camera is not detected, check physical connection")
    print("2. If nvargus daemon is not running, try: sudo systemctl restart nvargus-daemon")
    print("3. If no pipelines work, the camera might need different drivers")
    print("4. Check dmesg for camera-related errors: dmesg | grep -i camera")

if __name__ == '__main__':
    main()