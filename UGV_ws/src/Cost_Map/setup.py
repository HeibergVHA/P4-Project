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
        'lidar_costmap_node = Cost_Map.lidar_costmap_node:main',],
    },
)
