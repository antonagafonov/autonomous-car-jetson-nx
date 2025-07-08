from setuptools import setup

package_name = 'car_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antonagafonov',
    maintainer_email='toonagafonov@gmail.com',
    description='Computer vision and perception for autonomous car',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = car_perception.camera_node:main',
            'image_viewer = car_perception.image_viewer:main',
        ],
    })