from setuptools import setup
import os
from glob import glob

package_name = 'car_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antonagafonov',
    maintainer_email='toonagafonov@gmail.com',
    description='Launch files for autonomous car',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    })