from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'PLC'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/PLC/', ['PLC/procssing_times_table.csv']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='heiberg',
    maintainer_email='heibergvha@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'tcp_node = PLC.Tcp_node:main',
        'process_node = PLC.process_node:main',
        'drone_cltside = PLC.Drone_Cltside:main',
        'DroneController = PLC.DroneController:main',
        'mission_planner_node = PLC.mission_planner_node:main',
        'vicon_node = PLC.vicon_node:main',
        'Lidar_collection = PLC.LidarCollection:main',
        'radio_node = PLC.radio_node:main',
        'plotter = PLC.plotter:main',
        ],
    },
)

