#!/bin/bash

echo "🔧 Conservative Setuptools Fix for ROS2 Foxy"
echo "============================================"

# Check current versions first
echo "📋 Current versions:"
python3 -c "import setuptools; print(f'setuptools: {setuptools.__version__}')" 2>/dev/null || echo "setuptools: ERROR"
python3 -c "import packaging; print(f'packaging: {packaging.__version__}')" 2>/dev/null || echo "packaging: ERROR"

echo ""
echo "🎯 Target versions for ROS2 Foxy compatibility:"
echo "setuptools: 67.8.0 (stable, well-tested)"
echo "packaging: 21.3 (compatible with setuptools 67.8.0)"

echo ""
read -p "❓ Proceed with setuptools downgrade? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 1
fi

echo "⬇️  Downgrading setuptools and packaging..."

# Specific versions that work well with ROS2 Foxy
python3 -m pip install setuptools==67.8.0
python3 -m pip install packaging==21.3

echo ""
echo "✅ Updated versions:"
python3 -c "import setuptools; print(f'setuptools: {setuptools.__version__}')"
python3 -c "import packaging; print(f'packaging: {packaging.__version__}')"

echo ""
echo "🧪 Testing compatibility..."
python3 -c "
try:
    from packaging.version import Version
    from setuptools._core_metadata import _distribution_fullname
    print('✅ Core functions working')
except Exception as e:
    print(f'❌ Still has issues: {e}')
"

echo ""
echo "✅ Setuptools fix complete!"
echo "Next: Fix setup.py files, then try building"
