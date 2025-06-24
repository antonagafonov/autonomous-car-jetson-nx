import cv2

# GStreamer pipeline string
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

# Open the camera
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera with GStreamer pipeline.")

# Capture one frame
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to capture image.")

# Save as PNG
cv2.imwrite("capture.png", frame)

print("Image saved as capture.png")
