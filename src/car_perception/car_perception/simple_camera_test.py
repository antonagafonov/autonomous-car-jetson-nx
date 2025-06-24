#!/usr/bin/env python3
"""
Simple camera test - no ROS imports at all
This will help identify if ROS is causing the issue
"""

import cv2
import time
import sys

def test_camera():
    """Test camera exactly like the working script"""
    print("Testing camera without any ROS imports...")
    
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
    
    try:
        print("Opening camera...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        print("✅ Camera opened successfully")
        
        print("Testing frame capture...")
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to capture frame")
            cap.release()
            return False
        
        print(f"✅ Frame captured successfully - Shape: {frame.shape}")
        
        # Save frame to verify it works
        cv2.imwrite("/tmp/ros_test_frame.png", frame)
        print("✅ Frame saved to /tmp/ros_test_frame.png")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_camera_in_loop():
    """Test camera in a loop like ROS would do"""
    print("\nTesting camera in continuous loop (like ROS timer)...")
    
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )
    
    try:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("❌ Failed to open camera for loop test")
            return False
        
        print("✅ Camera opened for loop test")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Capture frames for 5 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else:
                print(f"❌ Frame capture failed at frame {frame_count}")
                break
            
            time.sleep(1/30)  # 30 FPS
        
        cap.release()
        
        print(f"✅ Loop test completed - Captured {frame_count} frames in 5 seconds")
        return frame_count > 0
        
    except Exception as e:
        print(f"❌ Loop test exception: {e}")
        return False

def main():
    print("Simple Camera Test (No ROS)")
    print("=" * 40)
    
    # Test 1: Basic camera test
    basic_success = test_camera()
    
    # Test 2: Loop test (simulating ROS timer)
    loop_success = test_camera_in_loop()
    
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"Basic test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Loop test:  {'✅ PASS' if loop_success else '❌ FAIL'}")
    
    if basic_success and loop_success:
        print("\n🎉 Camera works perfectly without ROS!")
        print("The issue is likely ROS-related. Try these solutions:")
        print("1. Check if importing ROS packages affects camera")
        print("2. Run ROS node with minimal environment")
        print("3. Check ROS-specific environment variables")
    elif basic_success and not loop_success:
        print("\n⚠️  Camera works for single capture but not continuous")
        print("This might be a buffering or timing issue")
    else:
        print("\n❌ Camera not working even without ROS")
        print("This suggests a system-level issue")

if __name__ == '__main__':
    main()