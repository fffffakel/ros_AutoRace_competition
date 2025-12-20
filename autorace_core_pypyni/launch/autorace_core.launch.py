import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='autorace_core_pypyni', # Имя твоего пакета
            executable='lane_follower',   # Имя точки входа (как в setup.py)
            name='lane_follower_node',
            output='screen',
            parameters=[
                # Здесь можно будет передавать параметры, если понадобится
            ]
        )
    ])