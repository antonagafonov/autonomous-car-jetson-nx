#!/bin/bash
# ROS Dependencies Report Generator
# Saves comprehensive ROS package information to car_ws directory

cd ~/car_ws

echo "🔍 Generating ROS Dependencies Report..."

# Create the report file
REPORT_FILE="ros_dependencies_report.txt"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Start the report
cat > $REPORT_FILE << EOF
🚗 ROS2 DEPENDENCIES REPORT
Generated: $DATE
Workspace: $(pwd)
ROS Distro: $ROS_DISTRO
Platform: $(uname -a)

===============================================
EOF

echo "📋 Collecting ROS2 packages..." 
echo "" >> $REPORT_FILE
echo "=== ROS2 PACKAGES AVAILABLE ===" >> $REPORT_FILE
echo "Total packages: $(ros2 pkg list | wc -l)" >> $REPORT_FILE
echo "" >> $REPORT_FILE
ros2 pkg list >> $REPORT_FILE

echo "🏗️  Collecting workspace info..."
echo "" >> $REPORT_FILE
echo "=== WORKSPACE PACKAGES ===" >> $REPORT_FILE
colcon list >> $REPORT_FILE

echo "📦 Collecting system packages..."
echo "" >> $REPORT_FILE
echo "=== SYSTEM ROS-FOXY PACKAGES ===" >> $REPORT_FILE
echo "Total ROS packages: $(dpkg -l | grep ros-foxy | wc -l)" >> $REPORT_FILE
echo "" >> $REPORT_FILE
dpkg -l | grep ros-foxy >> $REPORT_FILE

echo "🐍 Collecting Python packages..."
echo "" >> $REPORT_FILE
echo "=== PYTHON PACKAGES (ROS-RELATED) ===" >> $REPORT_FILE
pip3 list | grep -E "(ros|cv|numpy|opencv|pandas|sklearn|tensorflow|torch)" >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "=== ALL PYTHON PACKAGES ===" >> $REPORT_FILE
pip3 list >> $REPORT_FILE

echo "🔧 Collecting environment info..."
echo "" >> $REPORT_FILE
echo "=== ROS ENVIRONMENT VARIABLES ===" >> $REPORT_FILE
printenv | grep ROS >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "=== WORKSPACE DEPENDENCIES CHECK ===" >> $REPORT_FILE
rosdep check --from-paths src --ignore-src >> $REPORT_FILE 2>&1

echo "📊 Collecting package dependencies..."
echo "" >> $REPORT_FILE
echo "=== CUSTOM PACKAGE DEPENDENCIES ===" >> $REPORT_FILE

# Check dependencies for each workspace package
for pkg in $(colcon list --names-only); do
    echo "" >> $REPORT_FILE
    echo "--- $pkg dependencies ---" >> $REPORT_FILE
    ros2 pkg dependencies $pkg >> $REPORT_FILE 2>/dev/null || echo "No dependencies found" >> $REPORT_FILE
done

echo "🏥 Running ROS2 doctor..."
echo "" >> $REPORT_FILE
echo "=== ROS2 SYSTEM HEALTH ===" >> $REPORT_FILE
ros2 doctor >> $REPORT_FILE 2>&1

echo "" >> $REPORT_FILE
echo "=== SUMMARY ===" >> $REPORT_FILE
echo "Total ROS2 packages: $(ros2 pkg list | wc -l)" >> $REPORT_FILE
echo "System ROS packages: $(dpkg -l | grep ros-foxy | wc -l)" >> $REPORT_FILE
echo "Workspace packages: $(colcon list --names-only | wc -l)" >> $REPORT_FILE
echo "Python packages: $(pip3 list | wc -l)" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "Report saved to: $(pwd)/$REPORT_FILE" >> $REPORT_FILE

echo "✅ Report generated successfully!"
echo "📄 Saved to: $(pwd)/$REPORT_FILE"
echo "📊 File size: $(du -h $REPORT_FILE | cut -f1)"

# Also create a quick summary file
SUMMARY_FILE="ros_summary.txt"
cat > $SUMMARY_FILE << EOF
🚗 ROS DEPENDENCIES QUICK SUMMARY
Generated: $DATE

📦 Package Counts:
   ROS2 packages available: $(ros2 pkg list | wc -l)
   System ROS packages: $(dpkg -l | grep ros-foxy | wc -l)
   Workspace packages: $(colcon list --names-only | wc -l)
   Python packages: $(pip3 list | wc -l)

🏗️  Workspace Packages:
$(colcon list)

🔧 Key Dependencies:
$(pip3 list | grep -E "(ros|cv|numpy|opencv)" | head -10)

🌐 Environment:
   ROS_DISTRO: $ROS_DISTRO
   ROS_DOMAIN_ID: $ROS_DOMAIN_ID
   Workspace: $(pwd)

📄 Full report: $REPORT_FILE
EOF

echo "📋 Quick summary: $(pwd)/$SUMMARY_FILE"

# Make files readable
chmod 644 $REPORT_FILE $SUMMARY_FILE

echo ""
echo "🎯 To view the report:"
echo "   cat ~/car_ws/$REPORT_FILE"
echo "   cat ~/car_ws/$SUMMARY_FILE"
echo ""
echo "📤 To share the report:"
echo "   less ~/car_ws/$REPORT_FILE"
echo "   head -50 ~/car_ws/$REPORT_FILE"