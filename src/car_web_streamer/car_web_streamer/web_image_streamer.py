#!/usr/bin/env python3
"""
Web Image Streamer Node
Subscribes to camera/image_raw and serves it over HTTP for network viewing
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import socket

# Try importing cv_bridge with fallback
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except (ImportError, SystemError) as e:
    print(f"cv_bridge import failed: {e}")
    print("Using manual conversion instead")
    CV_BRIDGE_AVAILABLE = False

class CvBridgeManual:
    """Manual implementation of cv_bridge functionality"""
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding='bgr8'):
        """Convert ROS Image message to OpenCV image manually"""
        if img_msg.encoding == 'bgr8':
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8)
            cv_image = cv_image.reshape((img_msg.height, img_msg.width, 3))
            return cv_image
        elif img_msg.encoding == 'rgb8':
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8)
            cv_image = cv_image.reshape((img_msg.height, img_msg.width, 3))
            return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        else:
            # For other encodings, try basic conversion
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8)
            if len(cv_image) == img_msg.height * img_msg.width * 3:
                cv_image = cv_image.reshape((img_msg.height, img_msg.width, 3))
                return cv_image
            else:
                # Fallback - create black image
                return np.zeros((img_msg.height, img_msg.width, 3), dtype=np.uint8)

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for streaming video"""
    
    def __init__(self, *args, streamer_node=None, **kwargs):
        self.streamer_node = streamer_node
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_html_page()
        elif self.path == '/stream.mjpg':
            self.send_mjpeg_stream()
        else:
            self.send_error(404)
    
    def send_html_page(self):
        """Send HTML page with video stream"""
        # Use string concatenation to avoid formatting issues
        content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stream-container {
            margin: 20px 0;
            border: 2px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            display: inline-block;
        }
        .stream-img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .info {
            margin: 10px 0;
            color: #666;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.online {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.offline {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            margin: 0 10px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .refresh {
            background-color: #28a745;
        }
        .refresh:hover {
            background-color: #1e7e34;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Robot Camera Stream</h1>
        <div class="info">
            <p>Server: ''' + str(self.streamer_node.get_local_ip()) + ''' | Port: ''' + str(self.streamer_node.port) + '''</p>
            <p>Resolution: ''' + str(self.streamer_node.current_width) + '''x''' + str(self.streamer_node.current_height) + '''</p>
        </div>
        
        <div id="status" class="status online">
            📡 Stream Active
        </div>
        
        <div class="stream-container">
            <img id="stream" class="stream-img" src="/stream.mjpg" 
                 alt="Camera Stream" 
                 onerror="showOfflineStatus()"
                 onload="showOnlineStatus()">
        </div>
        
        <div class="controls">
            <button onclick="location.reload()">🔄 Refresh Page</button>
            <button class="refresh" onclick="refreshStream()">📹 Refresh Stream</button>
        </div>
        
        <div class="info">
            <p>💡 <strong>Tips:</strong></p>
            <p>• The stream auto-refreshes if connection is lost</p>
            <p>• Use Ctrl+F5 for hard refresh if needed</p>
            <p>• Check robot network connection if stream fails</p>
        </div>
    </div>
    
    <script>
        function showOfflineStatus() {
            const status = document.getElementById('status');
            status.className = 'status offline';
            status.innerHTML = '❌ Stream Offline - Attempting to reconnect...';
            
            // Try to reconnect after 3 seconds
            setTimeout(refreshStream, 3000);
        }
        
        function showOnlineStatus() {
            const status = document.getElementById('status');
            status.className = 'status online';
            status.innerHTML = '📡 Stream Active';
        }
        
        function refreshStream() {
            const img = document.getElementById('stream');
            const currentSrc = img.src;
            img.src = '';
            setTimeout(() => {
                img.src = currentSrc + '?t=' + new Date().getTime();
            }, 100);
        }
        
        // Auto-refresh every 30 seconds to keep connection alive
        setInterval(() => {
            refreshStream();
        }, 30000);
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content.encode('utf-8'))))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def send_mjpeg_stream(self):
        """Send MJPEG stream"""
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        
        try:
            while True:
                if self.streamer_node.current_frame is not None:
                    # Encode frame as JPEG
                    ret, jpeg = cv2.imencode('.jpg', self.streamer_node.current_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_data = jpeg.tobytes()
                        
                        # Send frame
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame_data)}\r\n\r\n'.encode())
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            self.streamer_node.get_logger().debug(f'Client disconnected: {e}')

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Thread per request HTTP server"""
    allow_reuse_address = True

class WebImageStreamer(Node):
    def __init__(self):
        super().__init__('web_image_streamer')
        
        # Use cv_bridge if available, otherwise use manual implementation
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
            self.get_logger().info('Using cv_bridge')
        else:
            self.bridge = CvBridgeManual()
            self.get_logger().info('Using manual cv_bridge implementation')
        
        # Parameters
        self.declare_parameter('port', 8080)
        self.declare_parameter('quality', 80)
        self.declare_parameter('max_width', 1280)
        self.declare_parameter('max_height', 720)
        
        self.port = self.get_parameter('port').value
        self.quality = self.get_parameter('quality').value
        self.max_width = self.get_parameter('max_width').value
        self.max_height = self.get_parameter('max_height').value
        
        # Image storage
        self.current_frame = None
        self.current_width = 0
        self.current_height = 0
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        # Get local IP
        self.local_ip = self.get_local_ip()
        
        # Start HTTP server in separate thread
        self.start_http_server()
        
        # Timer for stats
        self.stats_timer = self.create_timer(10.0, self.log_stats)
        
        self.get_logger().info(f'🌐 Web Image Streamer Started!')
        self.get_logger().info(f'📱 Access from network: http://{self.local_ip}:{self.port}')
        self.get_logger().info(f'🏠 Access locally: http://localhost:{self.port}')
        self.get_logger().info(f'⚙️  Quality: {self.quality}%, Max size: {self.max_width}x{self.max_height}')
    
    def get_local_ip(self):
        """Get local IP address"""
        try:
            # Connect to a remote server to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except Exception:
            return "127.0.0.1"
    
    def image_callback(self, msg):
        """Handle incoming image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Resize if too large
            height, width = cv_image.shape[:2]
            if width > self.max_width or height > self.max_height:
                # Calculate scaling factor
                scale_w = self.max_width / width
                scale_h = self.max_height / height
                scale = min(scale_w, scale_h)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                cv_image = cv2.resize(cv_image, (new_width, new_height))
            
            # Update current frame
            self.current_frame = cv_image
            self.current_width = cv_image.shape[1]
            self.current_height = cv_image.shape[0]
            self.frame_count += 1
            self.last_frame_time = time.time()
            
            # Log first frame
            if self.frame_count == 1:
                self.get_logger().info(f'📸 First frame received: {self.current_width}x{self.current_height}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def start_http_server(self):
        """Start HTTP server in separate thread"""
        def run_server():
            try:
                # Create handler with node reference
                handler = lambda *args, **kwargs: StreamingHandler(*args, streamer_node=self, **kwargs)
                
                # Create server
                self.server = ThreadedHTTPServer(('0.0.0.0', self.port), handler)
                self.get_logger().info(f'🚀 HTTP server starting on port {self.port}')
                self.server.serve_forever()
            except Exception as e:
                self.get_logger().error(f'HTTP server error: {e}')
        
        # Start server thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def log_stats(self):
        """Log streaming statistics"""
        if self.frame_count > 0:
            time_since_last = time.time() - self.last_frame_time
            if time_since_last < 5.0:  # Only log if recent frames
                fps = self.frame_count / 10.0  # Over last 10 seconds
                self.get_logger().info(f'📊 Streaming: {fps:.1f} FPS, {self.current_width}x{self.current_height}, {self.frame_count} total frames')
            else:
                self.get_logger().warn(f'⚠️  No frames received for {time_since_last:.1f} seconds')
        else:
            self.get_logger().warn('⚠️  No frames received yet')
        
        # Reset frame count for next period
        self.frame_count = 0
    
    def destroy_node(self):
        """Clean up resources"""
        if hasattr(self, 'server'):
            self.get_logger().info('🛑 Shutting down HTTP server')
            self.server.shutdown()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    streamer = WebImageStreamer()
    
    try:
        rclpy.spin(streamer)
    except KeyboardInterrupt:
        streamer.get_logger().info('Web Image Streamer interrupted')
    finally:
        streamer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()