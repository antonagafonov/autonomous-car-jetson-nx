#!/bin/bash

echo "🔍 Checking Current Build Environment"
echo "===================================="

# Check if we're in the right directory
if [[ ! -d "src" ]]; then
    echo "❌ Please run this from your ROS2 workspace root (~/car_ws)"
    exit 1
fi

echo "✅ Running from workspace: $(pwd)"
echo ""

# Check Python and pip versions
echo "🐍 Python Environment:"
echo "----------------------"
python3 --version
pip3 --version
echo ""

# Check critical Python packages
echo "📦 Critical Python Packages:"
echo "----------------------------"
check_package() {
    local package=$1
    python3 -c "import $package; print(f'$package: {$package.__version__}')" 2>/dev/null || echo "$package: NOT INSTALLED"
}

check_package setuptools
check_package packaging
check_package wheel
echo ""

# Check ROS2 environment
echo "🤖 ROS2 Environment:"
echo "-------------------"
if [ -n "$ROS_DISTRO" ]; then
    echo "ROS Distribution: $ROS_DISTRO"
else
    echo "⚠️  ROS not sourced"
fi

if command -v colcon >/dev/null 2>&1; then
    echo "Colcon: $(colcon version)"
else
    echo "❌ Colcon not found"
fi
echo ""

# Check workspace status
echo "🏗️  Workspace Status:"
echo "--------------------"
echo "Packages in src/:"
ls -1 src/ | sed 's/^/  - /'
echo ""

if [ -d "build" ]; then
    echo "Build directory exists: $(du -sh build | cut -f1)"
else
    echo "No build directory"
fi

if [ -d "install" ]; then
    echo "Install directory exists: $(du -sh install | cut -f1)"
else
    echo "No install directory"
fi

if [ -d "log" ]; then
    echo "Log directory exists: $(du -sh log | cut -f1)"
else
    echo "No log directory"
fi
echo ""

# Check for known problematic files
echo "🔧 Setup Files Analysis:"
echo "-----------------------"
find src/ -name "setup.py" -exec echo "Found: {}" \;
find src/ -name "setup.cfg" -exec echo "Found: {}" \;
echo ""

# Check for specific issues in setup.py files
echo "🚨 Potential Issues Found:"
echo "-------------------------"
issues_found=0

for setup_file in $(find src/ -name "setup.py"); do
    if grep -q "tests_require" "$setup_file"; then
        echo "❌ $setup_file contains deprecated 'tests_require'"
        issues_found=$((issues_found + 1))
    fi
done

for setup_cfg in $(find src/ -name "setup.cfg"); do
    if grep -q "script-dir\|install-scripts" "$setup_cfg"; then
        echo "❌ $setup_cfg contains deprecated dash-separated options"
        issues_found=$((issues_found + 1))
    fi
done

if [ $issues_found -eq 0 ]; then
    echo "✅ No obvious setup file issues found"
fi
echo ""

# Check what built successfully last time
echo "📋 Last Build Results:"
echo "---------------------"
if [ -f "log/latest_build/events.log" ]; then
    echo "Recent build events:"
    tail -10 log/latest_build/events.log 2>/dev/null || echo "Could not read build log"
else
    echo "No recent build log found"
fi
echo ""

# Check if packages are currently functional
echo "🧪 Package Status Check:"
echo "-----------------------"
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    
    # Check if packages are available
    for pkg in car_drivers car_teleop car_perception car_description data_collect car_inference; do
        if ros2 pkg list 2>/dev/null | grep -q "^${pkg}$"; then
            echo "✅ $pkg is built and available"
        else
            echo "❌ $pkg not available"
        fi
    done
else
    echo "⚠️  No install/setup.bash found - workspace not built"
fi
echo ""

# Recommendations
echo "🎯 Recommendations:"
echo "------------------"

# Check setuptools version specifically
setuptools_version=$(python3 -c "import setuptools; print(setuptools.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    # Compare versions (rough check)
    major_version=$(echo "$setuptools_version" | cut -d. -f1)
    if [ "$major_version" -gt 67 ]; then
        echo "⚠️  Setuptools version ($setuptools_version) might be too new"
        echo "   Consider downgrading to 67.8.0 for compatibility"
    elif [ "$major_version" -lt 60 ]; then
        echo "⚠️  Setuptools version ($setuptools_version) might be too old"
        echo "   Consider upgrading to 67.8.0"
    else
        echo "✅ Setuptools version ($setuptools_version) looks reasonable"
    fi
else
    echo "❌ Could not check setuptools version"
fi

echo ""
echo "🔒 Safety Check Complete!"
echo "========================"
echo ""
echo "📝 Next Steps:"
echo "1. Review the output above"
echo "2. If everything looks good, you can proceed with fixes"
echo "3. Make a backup of your workspace if concerned:"
echo "   cp -r ~/car_ws ~/car_ws_backup"
echo "4. Run the fix script when ready"
echo ""
echo "💡 To create a backup:"
echo "   cd ~"
echo "   cp -r car_ws car_ws_backup_$(date +%Y%m%d_%H%M%S)"
