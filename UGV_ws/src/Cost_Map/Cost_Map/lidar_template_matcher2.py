#!/usr/bin/env python3

# ROS2 Core
from unittest import result

import rclpy
from rclpy.node import Node

# ROS2 Messages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

# Third-Party Libraries
import open3d as o3d
import numpy as np
import copy
import os

from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)

# -------------------------------------------------------
# ROS2 NODE
# -------------------------------------------------------

class LidarTemplateMatcherNode(Node):

    def __init__(self):
        super().__init__('lidar_template_matcher_node')

        self.get_logger().info("Lidar Template Matcher Node Started")

        # -------------------------------------------------------
        # PARAMETERS
        # -------------------------------------------------------

        try:
            package_share_directory = get_package_share_directory('Cost_Map')
        except PackageNotFoundError:
            package_share_directory = os.path.join(
                os.getcwd(), 'src', 'Cost_Map')

        default_scene = os.path.join(
            package_share_directory,
            'resource',
            'WALLR1.pcd'
        )
        default_template = os.path.join(
            package_share_directory,
            'resource',
            'rc_car_template.pcd'
        )

        self.declare_parameter('scene_file', default_scene)
        self.declare_parameter('template_file', default_template)
        self.declare_parameter('voxel_size', 0.05)

        self.scene_file = self.get_parameter(
            'scene_file').get_parameter_value().string_value

        self.template_file = self.get_parameter(
            'template_file').get_parameter_value().string_value

        self.voxel_size = self.get_parameter(
            'voxel_size').get_parameter_value().double_value

        # -------------------------------------------------------
        # PUBLISHERS
        # -------------------------------------------------------

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/detected_object_pose',
            10
        )

        # -------------------------------------------------------
        # SERVICES
        # -------------------------------------------------------

        self.service = self.create_service(
            Trigger,
            'run_template_matching',
            self.run_matching_callback
        )

        self.get_logger().info(
            "Service ready → ros2 service call "
            "/run_template_matching std_srvs/srv/Trigger '{}'"
        )

    # -------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------

    def preprocess_point_cloud(self, pcd, voxel_size):

        self.get_logger().info("Downsampling point cloud...")

        pcd_down = pcd.voxel_down_sample(voxel_size)

        self.get_logger().info("Estimating normals...")

        radius_normal = voxel_size * 2

        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal,
                max_nn=30
            )
        )

        self.get_logger().info("Computing FPFH features...")

        radius_feature = voxel_size * 5

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature,
                max_nn=100
            )
        )

        return pcd_down, fpfh
    
    # -------------------------------------------------------
    # GLOBAL REGISTRATION
    # -------------------------------------------------------

    '''def execute_global_registration(
        self,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
    ):

        distance_threshold = voxel_size * 1.0

        self.get_logger().info("Running RANSAC registration...")

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,

            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),

            ransac_n=4,

            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),

                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                )
            ],

            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                4000000,
                500
            )
        )

        return result'''

    def execute_global_registration(
        self,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
    ):
        """
        Coarse alignment using Fast Global Registration (FGR).
        More stable and faster than RANSAC.
        """

        distance_threshold = voxel_size * 1.0

        print("Running Fast Global Registration...")

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold,
                iteration_number=2000,
                tuple_scale=0.95,
                tuple_test=True,
                maximum_tuple_count=500
            )
        )   

        return result

    # -------------------------------------------------------
    # ICP REFINEMENT
    # -------------------------------------------------------

    def refine_registration(
        self,
        source,
        target,
        initial_transform,
        voxel_size
    ):

        distance_threshold = voxel_size * 0.4

        self.get_logger().info("Running ICP refinement...")

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        return result

    # -------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------

    def validate_match(self, result, template, voxel_size):

        fitness = result.fitness
        rmse = result.inlier_rmse

        num_template_points = len(template.points)
        estimated_inliers = fitness * num_template_points

        self.get_logger().info(
            f"Fitness: {fitness:.3f}"
        )

        self.get_logger().info(
            f"RMSE: {rmse:.4f}"
        )

        self.get_logger().info(
            f"Estimated inliers: {estimated_inliers:.0f}"
        )

        MIN_FITNESS = 0.3
        MAX_RMSE = voxel_size * 0.5
        MIN_INLIERS = 500

        if fitness < MIN_FITNESS:
            self.get_logger().warn("Rejected: Low fitness")
            return False

        if rmse > MAX_RMSE:
            self.get_logger().warn("Rejected: High RMSE")
            return False

        if estimated_inliers < MIN_INLIERS:
            self.get_logger().warn("Rejected: Too few inliers")
            return False

        self.get_logger().info("Match accepted")
        return True

    # -------------------------------------------------------
    # PUBLISH DETECTED POSE
    # -------------------------------------------------------

    def publish_pose(self, transformation):

        translation = transformation[:3, 3]

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.position.x = float(translation[0])
        msg.pose.position.y = float(translation[1])
        msg.pose.position.z = float(translation[2])

        # Identity quaternion
        msg.pose.orientation.w = 1.0

        self.pose_publisher.publish(msg)

        self.get_logger().info(
            f"Published pose: "
            f"x={translation[0]:.2f}, "
            f"y={translation[1]:.2f}, "
            f"z={translation[2]:.2f}"
        )

    # -------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------

    def visualize_match(self, template, scene, transformation):

        template_temp = copy.deepcopy(template)
        scene_temp = copy.deepcopy(scene)

        template_temp.paint_uniform_color([1, 0, 0])
        scene_temp.paint_uniform_color([0.7, 0.7, 0.7])

        template_temp.transform(transformation)

        o3d.visualization.draw_geometries([
            template_temp,
            scene_temp
        ])

    # -------------------------------------------------------
    # SERVICE CALLBACK
    # -------------------------------------------------------

    def run_matching_callback(self, request, response):

        try:

            self.get_logger().info("Loading point clouds...")

            scene = o3d.io.read_point_cloud(self.scene_file)
            template = o3d.io.read_point_cloud(self.template_file)

            if not scene.has_points():
                response.success = False
                response.message = "Scene point cloud empty"
                return response

            if not template.has_points():
                response.success = False
                response.message = "Template point cloud empty"
                return response

            self.get_logger().info(
                f"Scene points: {len(scene.points)}"
            )

            self.get_logger().info(
                f"Template points: {len(template.points)}"
            )

            # -------------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------------

            bbox = scene.get_axis_aligned_bounding_box()
            scene = scene.crop(bbox)

            scene_down, scene_fpfh = self.preprocess_point_cloud(
                scene,
                self.voxel_size
            )

            template_down, template_fpfh = self.preprocess_point_cloud(
                template,
                self.voxel_size
            )

            # -------------------------------------------------------
            # GLOBAL REGISTRATION
            # -------------------------------------------------------

            result_ransac = self.execute_global_registration(
                template_down,
                scene_down,
                template_fpfh,
                scene_fpfh,
                self.voxel_size
            )

            self.get_logger().info(
                f"RANSAC fitness: {result_ransac.fitness}"
            )

            self.get_logger().info(
                f"RANSAC RMSE: {result_ransac.inlier_rmse}"
            )

            # -------------------------------------------------------
            # ICP REFINEMENT
            # -------------------------------------------------------

            result_icp = self.refine_registration(
                template,
                scene,
                result_ransac.transformation,
                self.voxel_size
            )

            self.get_logger().info(
                f"ICP fitness: {result_icp.fitness}"
            )

            self.get_logger().info(
                f"ICP RMSE: {result_icp.inlier_rmse}"
            )

            # -------------------------------------------------------
            # VALIDATION
            # -------------------------------------------------------

            """valid = self.validate_match(
                result_icp,
                template,
                self.voxel_size
            )

            if not valid:

                response.success = False
                response.message = "Template match rejected"

                return response"""

            # -------------------------------------------------------
            # PUBLISH POSE
            # -------------------------------------------------------

            self.publish_pose(result_ransac.transformation)

            self.get_logger().info(
                f"RANSAC translation: "
                f"{result_ransac.transformation[:3, 3]}"
            )

            self.get_logger().info(
                f"ICP translation: "
                f"{result_icp.transformation[:3, 3]}"
            )

            # -------------------------------------------------------
            # VISUALIZATION
            # -------------------------------------------------------

            """self.visualize_match(
                template,
                scene,
                result_icp.transformation
            )"""

            response.success = True
            response.message = "Template matching completed"

            return response

        except Exception as e:

            self.get_logger().error(f"Matching failed: {e}")

            response.success = False
            response.message = str(e)

            return response


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = LidarTemplateMatcherNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()