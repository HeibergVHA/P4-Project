import os
from glob import glob
from setuptools import setup

package_name = 'px4_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('resource/*rviz'))
        # (os.path.join('share', package_name), ['scripts/TerminatorScript.sh'])
    ],
    install_requires=['setuptools', 'numpy', 'control'],
    zip_safe=True,
    maintainer='Mathias Lykholt-Ustrup',
    maintainer_email='mustru23@student.aau.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'px4_planner_api = px4_simulation.px4_planner_api:main',
                'px4_planner_implemented = px4_simulation.px4_planner_implemented:main',
                'px4_planner_original = px4_simulation.px4_planner_original:main',
        ],
    },
)
