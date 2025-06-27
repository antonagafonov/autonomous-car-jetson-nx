import cv2
import time

# Replace this with your exact GStreamer pipeline for your camera
# Example for Jetson CSI camera:
gst_pipeline = (
    "nvarguscamerasrc sensor_id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true sync=false"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream or file.")
    exit()

print("Camera opened successfully!")

# Try to read a few frames to confirm
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {i}")
        break
    print(f"Read frame {i}: {frame.shape}") # Should show (480, 640, 3) or similar

cap.release()
print("Camera released.")