from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'car_web_streamer'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Web-based image streaming for robot camera',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'web_image_streamer = car_web_streamer.web_image_streamer:main',
        ],
    })