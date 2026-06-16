#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── TCP Server — receives bag from ELE ────────────────────────────
        Node(
            package='UGV_SvrSide',
            executable='ugv_svrside',
            name='ugv_server_side',
            output='screen',
        ),

        # ── Livox Bag Reader — plays bag and launches FAST-LIO2 ───────────
        Node(
            package='UGV_SvrSide',
            executable='livox_bag_reader',
            name='livox_bag_reader',
            output='screen',
        ),

        # ── Radio Node — telemetry link to ELE ───────────────────────────
        Node(
            package='UGV_SvrSide',
            executable='radio_node',
            name='radio_node',
            output='screen',
        ),

        # ── Costmap Generator — builds costmap from FAST-LIO2 PCD ────────
        Node(
            package='Cost_Map',
            executable='lidar_costmap_node',
            name='lidar_costmap_generator',
            output='screen',
        ),

        # ── Template Matcher — localises WALL-R in the costmap ───────────
        Node(
            package='Cost_Map',
            executable='lidar_template_matcher_node',
            name='lidar_template_matcher_node',
            output='screen',
        ),

        # ── Navigation Node — A* path planner ────────────────────────────
        Node(
            package='ugv_navigation',
            executable='navigation_node2',
            name='navigation_node',
            output='screen',
        ),

        # ── Mission Control — operator UI on laptop ───────────────────────
        # Note: this node should ideally run on the laptop, not the UGV.
        # Include here only if running everything on the same machine.
        # Node(
        #     package='mission_control',
        #     executable='mission_control',
        #     name='mission_control',
        #     output='screen',
        # ),

    ])