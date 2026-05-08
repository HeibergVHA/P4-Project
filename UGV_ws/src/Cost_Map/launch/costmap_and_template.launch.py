from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    package_share = get_package_share_directory('Cost_Map')

    scene_file = os.path.join(
        package_share,
        'resource',
        'WALLR1.pcd'
    )

    template_file = os.path.join(
        package_share,
        'resource',
        'rc_car_template.pcd'
    )

    costmap_node = Node(
        package='Cost_Map',
        executable='lidar_costmap_node',
        name='lidar_costmap_node',
        output='screen',

        parameters=[
            {
                'pcd_file_path': 'src/Cost_Map/resource/WALLR1.pcd',
                'map_frame_id': 'map',
                'costmap_resolution': 0.05,
                'inflation_radius_meters': 0.05,
                'flat_caution_threshold': 0.03,
                'caution_obstacle_threshold': 0.08,
                'max_filter_size': 2,
                'min_filter_size': 2,
            }
        ]
    )

    template_matcher_node = Node(
        package='Cost_Map',
        executable='lidar_template_matcher_node',
        name='lidar_template_matcher_node',
        output='screen',

        parameters=[
            {
                'scene_file': scene_file,
                'template_file': template_file,
                'voxel_size': 0.05,
            }
]
    )

    return LaunchDescription([
        costmap_node,
        template_matcher_node
    ])