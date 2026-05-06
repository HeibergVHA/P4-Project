import launch
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    ugv_svrside = Node(
        package='UGV_SvrSide',
        executable='ugv_svrside',
        name='ugv_svrside',
        output='screen'
    )

    livox_bag_reader = Node(
        package='UGV_SvrSide',
        executable='livox_bag_reader',
        name='livox_bag_reader',
        output='screen'
    )
    
    lidar_costmap_node = Node(
        package='Cost_Map',
        executable='lidar_costmap_node',
        name='lidar_costmap_node',
        output='screen'
    )


    fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('fast_lio'),
                'launch',
                'mapping.launch.py'
            )
        ),
        launch_arguments={'config_file': 'avia.yaml',
                          'rviz': 'false'}.items()
    )

    # Trigger start_reading after bag transfer window
    # Adjust delay to match your expected transfer time
    # Launch FAST-LIO shortly after reading start

    return LaunchDescription([
        ugv_svrside,
        livox_bag_reader,
        lidar_costmap_node
    ])