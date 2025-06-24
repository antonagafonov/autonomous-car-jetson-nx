#!/usr/bin/env python3
# Save as test_ros_imports.py

print("Testing camera before ROS imports...")
import cv2
gst_pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
print(f"Before ROS: Camera opened = {cap.isOpened()}")
if cap.isOpened():
    cap.release()

print("Now importing ROS...")
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

print("Testing camera after ROS imports...")
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
print(f"After ROS: Camera opened = {cap.isOpened()}")
if cap.isOpened():
    cap.release()