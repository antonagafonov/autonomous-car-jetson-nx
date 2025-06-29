#!/bin/bash
# ~/car_ws/kill_car_system.sh
# System-wide cleanup script for autonomous car

echo "🧹 Autonomous Car System Cleanup"
echo "================================="

# Function to kill processes by pattern
kill_pattern() {
    local pattern=$1
    local name=$2
    
    echo "🔍 Searching for $name processes..."
    
    # Get PIDs matching pattern
    PIDS=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -n "$PIDS" ]; then
        echo "📋 Found PIDs: $PIDS"
        
        # First try SIGTERM
        echo "🛑 Sending SIGTERM to $name..."
        pkill -f "$pattern" 2>/dev/null
        sleep 2
        
        # Check if still running
        REMAINING=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -n "$REMAINING" ]; then
            echo "⚡ Force killing $name (SIGKILL)..."
            pkill -9 -f "$pattern" 2>/dev/null
            sleep 1
        fi
        
        # Verify cleanup
        FINAL_CHECK=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -z "$FINAL_CHECK" ]; then
            echo "✅ $name processes cleaned up"
        else
            echo "❌ Some $name processes still running: $FINAL_CHECK"
        fi
    else
        echo "✅ No $name processes found"
    fi
    echo ""
}

# Kill specific car nodes (in dependency order)
kill_pattern "joy_node" "Joy Node"
kill_pattern "joystick_controller" "Joystick Controller"
kill_pattern "cmd_relay" "Command Relay"
kill_pattern "motor_controller" "Motor Controller"
kill_pattern "camera_node" "Camera Node"
kill_pattern "bag_collect" "Bag Collector"

# Kill by package patterns
kill_pattern "car_drivers" "Car Drivers"
kill_pattern "car_perception" "Car Perception"
kill_pattern "car_teleop" "Car Teleop"
kill_pattern "data_collect" "Data Collect"

# Kill any remaining launch processes
echo "🔍 Searching for launch processes..."
LAUNCH_PIDS=$(pgrep -f "launch.*car_manual_control" 2>/dev/null)
if [ -n "$LAUNCH_PIDS" ]; then
    echo "🛑 Killing launch processes: $LAUNCH_PIDS"
    pkill -f "launch.*car_manual_control" 2>/dev/null
    sleep 2
    pkill -9 -f "launch.*car_manual_control" 2>/dev/null
    echo "✅ Launch processes cleaned up"
else
    echo "✅ No launch processes found"
fi
echo ""

# Clean up GPIO resources
echo "🔧 Cleaning up GPIO resources..."
echo "15" | sudo tee /sys/class/gpio/unexport >/dev/null 2>&1 || true
echo "32" | sudo tee /sys/class/gpio/unexport >/dev/null 2>&1 || true
echo "✅ GPIO cleanup complete"
echo ""

# Final verification
echo "🔍 Final system check..."
REMAINING_NODES=$(ps aux | grep -E "(camera_node|motor_controller|bag_collect|joystick_controller|cmd_relay|joy_node)" | grep -v grep | wc -l)

if [ "$REMAINING_NODES" -eq 0 ]; then
    echo "✅ All car nodes successfully terminated"
    echo "🎉 System cleanup complete!"
else
    echo "⚠️  Some processes may still be running:"
    ps aux | grep -E "(camera_node|motor_controller|bag_collect|joystick_controller|cmd_relay|joy_node)" | grep -v grep
fi

# Check ROS2 nodes
echo ""
echo "🔍 Checking active ROS2 nodes..."
ACTIVE_NODES=$(ros2 node list 2>/dev/null | wc -l)
if [ "$ACTIVE_NODES" -eq 0 ]; then
    echo "✅ No active ROS2 nodes"
else
    echo "⚠️  Active ROS2 nodes still found:"
    ros2 node list 2>/dev/null || true
fi

echo ""
echo "🏁 Cleanup script complete!"
echo "================================="
echo ""