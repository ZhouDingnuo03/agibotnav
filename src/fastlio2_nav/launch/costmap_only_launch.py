import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

def generate_launch_description():
    # 配置文件路径
    config_dir = os.path.join(get_package_share_directory('fastlio2_nav'), 'config')
    costmap_params = os.path.join(config_dir, 'costmap_only_params.yaml')

    # 步骤1：创建组件容器（运行代价地图组件）
    costmap_container = ComposableNodeContainer(
        name='costmap_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_isolated',  # Nav2标准组件容器
        composable_node_descriptions=[
            # 步骤2：加载代价地图组件（核心）
            ComposableNode(
                package='nav2_costmap_2d',
                plugin='nav2_costmap_2d::Costmap2DROS',  # 代价地图核心插件
                name='global_costmap',
                namespace='global_costmap',
                parameters=[costmap_params],
                remappings=[
                    ('costmap', 'global_costmap/costmap'),
                    ('costmap_updates', 'global_costmap/costmap_updates')
                ]
            )
        ],
        output='screen'
    )

    # 可选：RViz2可视化
    rviz_config = os.path.join(get_package_share_directory('nav2_bringup'), 'rviz', 'nav2_default_view.rviz')
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        costmap_container,
        rviz2  # 不需要可视化可注释掉
    ])

