import glob
from setuptools import find_packages, setup

package_name = 'Cost_Map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob.glob('resource/*')),
        ('share/' + package_name + '/Cost_Map_Coordinates', glob.glob('Cost_Map_Coordinates/*')),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config/', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesper-kwame-jensen',
    maintainer_email='you@example.com',
    description='Coverts A point cloud to a cost map based on the slope of the terrain',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'lidar_costmap_node = Cost_Map.lidar_costmap_node:main',
        'lidar_template_matcher_node = Cost_Map.lidar_template_matcher_node:main',
        'pcd_publisher = Cost_Map.pcd_publisher:main',
        ],
    },
)
