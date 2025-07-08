from setuptools import setup

package_name = 'car_teleop'

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
    maintainer='toon',
    maintainer_email='toon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'joystick_controller = car_teleop.joystick_controller:main',
            'cmd_relay = car_teleop.cmd_relay:main',
        ],
    })
