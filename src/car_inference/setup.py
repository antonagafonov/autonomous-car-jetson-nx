from setuptools import setup
import os
from glob import glob

package_name = 'car_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='Autonomous inference package for car control using trained neural networks',
    license='Apache-2.0',
    #,
    entry_points={
        'console_scripts': [
            'inference_node = car_inference.inference_node:main',
            'safety_monitor = car_inference.safety_monitor:main',
            'model_manager = car_inference.model_manager:main',
        ],
    })