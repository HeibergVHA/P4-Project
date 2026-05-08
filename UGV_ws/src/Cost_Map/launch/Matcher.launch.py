from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
 
 
def generate_launch_description():
    # ---------- Arguments ----------
    template_path_arg = DeclareLaunchArgument(
        'template_path',
        default_value='src/Cost_Map/resource/rc_car_template.pcd',
        description='Absolute path to the template PCD file (rc_car_template.pcd)',
    )
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.05',
        description='Voxel downsampling leaf size in metres',
    )
    fitness_threshold_arg = DeclareLaunchArgument(
        'fitness_threshold',
        default_value='0.6',
        description='Maximum score (1-fitness) to publish a match',
    )
 
    # ---------- Node ----------
    matcher_node = Node(
        package='Cost_Map',
        executable='lidar_template_matcher_node',
        name='lidar_template_matcher_node',
        output='screen',
        parameters=[
            PathJoinSubstitution(['src/Cost_Map', 'config', 'params.yaml']),
            {
                'template_path':     LaunchConfiguration('template_path'),
                'voxel_size':        LaunchConfiguration('voxel_size'),
                'fitness_threshold': LaunchConfiguration('fitness_threshold'),
            },
        ],
    )

    pcd_publisher_node = Node(
        package='Cost_Map',
        executable='pcd_publisher',
        name='pcd_publisher',
        output='screen',
        parameters=[
            {
                'pcd_file_path': 'src/Cost_Map/resource/WALLR1.pcd',
            }
        ],
    )

    return LaunchDescription([
        template_path_arg,
        voxel_size_arg,
        fitness_threshold_arg,
        matcher_node,
        pcd_publisher_node,
    ])