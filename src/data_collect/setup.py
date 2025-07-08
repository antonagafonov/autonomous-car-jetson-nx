from setuptools import setup

package_name = 'data_collect'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS2 package containing the bag collector node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bag_collect = data_collect.bag_collect:main',
        ],
    })