import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 配置文件路径
    config_dir = os.path.join(get_package_share_directory('fastlio2_nav'), 'config')
    nav2_params_file = os.path.join(config_dir, 'nav2_params.yaml')
    
    # 新增：创建costmap_common_params，强制禁用static_layer
    costmap_common_params = {
        'global_costmap': {
            'ros__parameters': {
                'plugins': ['voxel_layer', 'inflation_layer'],
                'static_layer': {'enabled': False}
            }
        },
        'local_costmap': {
            'ros__parameters': {
                'plugins': ['voxel_layer', 'inflation_layer'],
                'static_layer': {'enabled': False}
            }
        }
    }

    # 启动Nav2（添加覆盖参数，彻底禁用static_layer）
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': nav2_params_file,
            'autostart': 'true',
            # 关键：不传入map参数，让map_server不启动，彻底断联static_layer
            'map': ''  
        }.items()
    )

    # 点云转激光（备用，可选）
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[('input', '/cloud_registered')],
        parameters=[{
            'min_height': -0.5,
            'max_height': 2.0,
            'angle_min': -3.1415,
            'angle_max': 3.1415,
            'range_min': 0.1,
            'range_max': 50.0,
            'scan_time': 0.1,
            'output_frame': 'base_link'
        }]
    )

    # RViz2可视化
    rviz_config = os.path.join(get_package_share_directory('nav2_bringup'), 'rviz', 'nav2_default_view.rviz')
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # 新增：强制覆盖costmap参数，禁用static_layer
    costmap_params_node = Node(
        package='rclcpp_components',
        executable='component_container_isolated',
        name='costmap_params_container',
        namespace='',
        parameters=[costmap_common_params],
        output='screen'
    )

    return LaunchDescription([
        costmap_params_node,  # 先加载覆盖参数
        nav2_bringup,
        pointcloud_to_laserscan,
        rviz2
    ])

