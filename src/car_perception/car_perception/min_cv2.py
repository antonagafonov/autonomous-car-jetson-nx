import cv2
import sys
import time

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=640, # For display window
    display_height=480, # For display window
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1, format=(string)NV12 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

pipeline = gstreamer_pipeline(
    capture_width=1920, capture_height=1080, # Native resolution of camera
    display_width=640, display_height=480,  # Resolution for display
    framerate=30,
    flip_method=0 # Adjust if your camera is mounted upside down etc.
)

print(f"GStreamer Pipeline: {pipeline}")

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    print("Check if nvargus-daemon is running: 'sudo systemctl status nvargus-daemon'")
    print("Check permissions for /dev/video0 (though nvarguscamerasrc typically bypasses this for CSI)")
    print("Check GStreamer plugins: 'gst-inspect-1.0 nvarguscamerasrc'")
    print("Try simpler pipeline directly with gst-launch-1.0 from terminal.")
    sys.exit(1)

print("Camera opened successfully. Reading frames...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        if frame is None or frame.size == 0:
            print("Warning: Captured empty frame. Retrying...")
            time.sleep(0.1) # Small delay
            continue

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")