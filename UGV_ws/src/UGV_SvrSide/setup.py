from setuptools import find_packages, setup

package_name = 'UGV_SvrSide'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/UGV_SvrSide/', ['UGV_SvrSide/procssing_times_table.csv']),
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
        'tcp_node = UGV_SvrSide.Tcp_node:main',
        'process_node = UGV_SvrSide.process_node:main',
        'ugv_svrside = UGV_SvrSide.UGV_SvrSide:main'
        ],
    },
)

