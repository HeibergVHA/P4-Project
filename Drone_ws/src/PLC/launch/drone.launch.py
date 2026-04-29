import launch
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    livox_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('livox_ros2_driver'),
                'launch',
                'livox_lidar_msg_launch.py'
            )
        )
    )
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('mavros'),
                'launch',
                'apm.launch'
            )
        ),
        launch_arguments={
            'fcu_url': '/dev/ttyAMA0:921600',
        }.items()
)

    lidar_collection = Node(
        package='PLC',
        executable='Lidar_collection',
        name='lidar_collection',
        output='screen'
    )

    drone_cltside = Node(
        package='PLC',
        executable='drone_cltside',
        name='drone_cltside',
        output='screen'
    )

    radio_node = Node(
        package='PLC',
        executable='radio_node',
        name='radio_node',
        output='screen'
    )

    mission_planner_node = Node(
        package='PLC',
        executable='mission_planner_node',
        name='mission_planner_node',
        output='screen'
    )

    DroneController = Node(
        package='PLC',
        executable='DroneController',
        name='DroneController',
        output='screen'
    )

    #Give stop recording some time to finish before sending the rosbag
    send_bag = TimerAction(
        period=30.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/send', 'std_srvs/srv/Trigger', '{}'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        # lidar_collection,
        # drone_cltside
        mavros,
        radio_node,
        mission_planner_node,
        DroneController
    ])

