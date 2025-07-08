from setuptools import setup

package_name = 'car_drivers'

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
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Hardware drivers for autonomous car',
    license='MIT',
    entry_points={
        'console_scripts': [
            'motor_controller = car_drivers.motor_controller:main',
            'motor_test = car_drivers.motor_test:main',
        ],
    })