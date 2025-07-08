#!/usr/bin/env python3

import os
import re

def fix_setup_cfg_files():
    """Fix setup.cfg files to remove deprecated dash-separated options"""
    
    setup_cfg_files = []
    for root, dirs, files in os.walk('src'):
        if 'setup.cfg' in files:
            setup_cfg_files.append(os.path.join(root, 'setup.cfg'))
    
    print(f"🔧 Found {len(setup_cfg_files)} setup.cfg files to fix")
    
    for cfg_file in setup_cfg_files:
        try:
            with open(cfg_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Fix the dash-separated options
            content = re.sub(r'script-dir', 'script_dir', content)
            content = re.sub(r'install-scripts', 'install_scripts', content)
            
            if content != original_content:
                with open(cfg_file, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed: {cfg_file}")
            else:
                print(f"ℹ️  No changes needed: {cfg_file}")
                
        except Exception as e:
            print(f"❌ Error fixing {cfg_file}: {e}")

def fix_setup_py_files():
    """Fix setup.py files to remove deprecated tests_require"""
    
    setup_py_files = []
    for root, dirs, files in os.walk('src'):
        if 'setup.py' in files:
            setup_py_files.append(os.path.join(root, 'setup.py'))
    
    print(f"🔧 Found {len(setup_py_files)} setup.py files to fix")
    
    for py_file in setup_py_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Remove tests_require lines
            content = re.sub(r',?\s*tests_require=\[.*?\]', '', content, flags=re.DOTALL)
            content = re.sub(r'tests_require=\[.*?\],?', '', content, flags=re.DOTALL)
            
            # Clean up any double commas or trailing commas
            content = re.sub(r',\s*,', ',', content)
            content = re.sub(r',\s*\)', ')', content)
            
            if content != original_content:
                with open(py_file, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed: {py_file}")
            else:
                print(f"ℹ️  No changes needed: {py_file}")
                
        except Exception as e:
            print(f"❌ Error fixing {py_file}: {e}")

if __name__ == "__main__":
    print("🛠️  Quick Setup Files Fix")
    print("========================")
    
    if not os.path.exists('src'):
        print("❌ Please run from workspace root (~/car_ws)")
        exit(1)
    
    print("1. Fixing setup.cfg files...")
    fix_setup_cfg_files()
    
    print("\n2. Fixing setup.py files...")
    fix_setup_py_files()
    
    print("\n✅ All setup files processed!")
    print("🧪 Ready to test builds!")
