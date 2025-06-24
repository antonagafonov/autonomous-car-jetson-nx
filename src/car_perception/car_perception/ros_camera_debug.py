#!/usr/bin/env python3
"""
Debug script to understand why camera works standalone but not in ROS2
"""

import cv2
import os
import sys
import subprocess
import time

def print_environment():
    """Print environment variables that might affect camera"""
    print("=== Environment Variables ===")
    camera_vars = ['GST_PLUGIN_PATH', 'GST_PLUGIN_SCANNER', 'LD_LIBRARY_PATH', 
                   'PYTHONPATH', 'ROS_DISTRO', 'ROS_VERSION']
    
    for var in camera_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    print(f"USER: {os.environ.get('USER', 'unknown')}")
    print(f"HOME: {os.environ.get('HOME', 'unknown')}")
    print(f"PWD: {os.environ.get('PWD', 'unknown')}")

def check_permissions():
    """Check file permissions and user groups"""
    print("\n=== Permissions Check ===")
    
    # Check video device permissions
    try:
        stat_info = os.stat('/dev/video0')
        print(f"/dev/video0 permissions: {oct(stat_info.st_mode)[-3:]}")
    except Exception as e:
        print(f"Error checking /dev/video0: {e}")
    
    # Check user groups
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        print(f"User groups: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error checking groups: {e}")
    
    # Check if user is in video group
    try:
        result = subprocess.run(['groups', os.environ.get('USER', '')], capture_output=True, text=True)
        if 'video' in result.stdout:
            print("✓ User is in video group")
        else:
            print("✗ User is NOT in video group")
    except Exception as e:
        print(f"Error checking video group: {e}")

def test_gstreamer_directly():
    """Test GStreamer command line"""
    print("\n=== Testing GStreamer Command Line ===")
    
    gst_cmd = [
        'gst-launch-1.0',
        'nvarguscamerasrc', 'num-buffers=1', '!',
        'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12', '!',
        'nvvidconv', '!',
        'video/x-raw,format=BGRx', '!',
        'videoconvert', '!',
        'video/x-raw,format=BGR', '!',
        'filesink', 'location=/tmp/test_frame.raw'
    ]
    
    try:
        result = subprocess.run(gst_cmd, capture_output=True, text=True, timeout=10)
        print(f"GStreamer command exit code: {result.returncode}")
        if result.returncode == 0:
            print("✓ GStreamer command line test successful")
        else:
            print("✗ GStreamer command line test failed")
            print(f"stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("GStreamer command timed out")
    except Exception as e:
        print(f"Error running GStreamer command: {e}")

def test_opencv_verbose():
    """Test OpenCV with verbose output"""
    print("\n=== Testing OpenCV with Verbose Output ===")
    
    # Set OpenCV debug environment
    os.environ['OPENCV_LOG_LEVEL'] = 'DEBUG'
    os.environ['GST_DEBUG'] = '3'
    
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    print(f"Pipeline: {gst_pipeline}")
    print("Attempting to open camera...")
    
    try:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        print(f"Camera opened: {cap.isOpened()}")
        
        if cap.isOpened():
            print("Testing frame capture...")
            ret, frame = cap.read()
            print(f"Frame captured: {ret}")
            if ret:
                print(f"Frame shape: {frame.shape}")
            cap.release()
            return True
        else:
            print("Failed to open camera")
            cap.release()
            return False
            
    except Exception as e:
        print(f"Exception during camera test: {e}")
        return False

def compare_with_working_script():
    """Run the known working script and compare"""
    print("\n=== Running Known Working Script ===")
    
    working_script = '''
import cv2
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("WORKING_SCRIPT_SUCCESS")
    else:
        print("WORKING_SCRIPT_CAPTURE_FAILED")
    cap.release()
else:
    print("WORKING_SCRIPT_OPEN_FAILED")
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', working_script], 
                              capture_output=True, text=True, timeout=15)
        print(f"Working script result: {result.stdout.strip()}")
        print(f"Working script stderr: {result.stderr}")
        return "WORKING_SCRIPT_SUCCESS" in result.stdout
    except Exception as e:
        print(f"Error running working script: {e}")
        return False

def test_without_ros():
    """Test camera in current Python session (mimicking ROS environment)"""
    print("\n=== Testing Camera in Current Session ===")
    return test_opencv_verbose()

def main():
    print("Camera Debug Tool - ROS2 vs Standalone")
    print("=" * 50)
    
    print_environment()
    check_permissions()
    
    # Test 1: GStreamer command line
    test_gstreamer_directly()
    
    # Test 2: Working script as subprocess
    working_subprocess = compare_with_working_script()
    
    # Test 3: Camera in current session
    working_current = test_without_ros()
    
    print("\n=== Summary ===")
    print(f"Working script (subprocess): {'✓' if working_subprocess else '✗'}")
    print(f"Current session: {'✓' if working_current else '✗'}")
    
    if working_subprocess and not working_current:
        print("\n🔍 DIAGNOSIS: Camera works in subprocess but not current session")
        print("This suggests an environment or initialization issue in the current Python session")
        print("\nPossible solutions:")
        print("1. ROS2 might be interfering with GStreamer")
        print("2. Environment variables might be different")
        print("3. Library loading order might be different")
        print("4. User permissions might be different in ROS context")
        
        print("\nTrying solutions:")
        
        # Try adding user to video group
        print("- Check if you need to add user to video group:")
        print("  sudo usermod -a -G video $USER")
        print("  (then logout and login again)")
        
        # Try running with different environment
        print("- Try running ROS node with minimal environment:")
        print("  env -i HOME=$HOME USER=$USER /opt/ros/foxy/bin/ros2 run car_perception camera_node")
    
    elif not working_subprocess and not working_current:
        print("\n🔍 DIAGNOSIS: Camera not working in any Python context")
        print("This suggests a system-level issue")
        print("Check camera connection and nvargus daemon")
    
    elif working_current:
        print("\n✓ Camera working in current session - ROS issue might be elsewhere")

if __name__ == '__main__':
    main()