#!/usr/bin/env python3

# ROS2 Core
from unittest import result

import rclpy
from rclpy.node import Node

# ROS2 Messages
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, DurabilityPolicy

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

        # QoS for RViz compatibility
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.publisher_pointcloud = self.create_publisher(
            PointCloud2,
            '/pointcloud',
            qos
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
                max_nn=50
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
    
    # PCD loading and processing
    def load_pcd_and_process(self):
        try:
            pcd = o3d.io.read_point_cloud(self.scene_file)
            if not pcd.has_points():
                self.get_logger().error(f"PCD file has no points: {o3d.io.read_point_cloud(self.scene_file)}")
                return None, None, None, None

            points = np.asarray(pcd.points)

            points, T, plane_model = self.align_floor_to_xy_plane(points)

            pcd.points = o3d.utility.Vector3dVector(points)

            """# Rotate 90° around Y to align sensor frame → map frame
            # You can change this rotation if your PCD is already in the correct orientation, or if it uses a different convention.
            theta = np.pi / 2
            R = np.array([
                [ np.cos(theta), 0, np.sin(theta)],
                [ 0,             1, 0            ],
                [-np.sin(theta), 0, np.cos(theta)],
            ])
            points = points @ R.T"""

            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            self.get_logger().info(f"Loaded {len(points)} points. min={min_bound[:2]}  max={max_bound[:2]}")

            pcd.points = o3d.utility.Vector3dVector(points)

            return pcd, points, min_bound, max_bound

        except Exception as e:
            self.get_logger().error(f"Failed to load PCD: {e}")
            return None, None, None, None

    def remove_floor(
            self,
            pcd,
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=2000):

        original_points = len(pcd.points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        self.get_logger().info(
            f"Floor plane inliers: {len(inliers)}"
        )

        no_floor = pcd.select_by_index(
            inliers,
            invert=True
        )

        self.get_logger().info(
            f"Remaining points after floor removal: "
            f"{len(no_floor.points)} / {original_points}"
        )

        return no_floor

    def align_floor_to_xy_plane(
            self,
            points,
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000):

        # Accept Open3D point clouds directly
        if isinstance(points, o3d.geometry.PointCloud):
            points = np.asarray(points.points)

        points = np.asarray(points, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64)
        )

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        a, b, c, d = plane_model

        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal)

        # Force consistent floor orientation
        if abs(normal[0]) > abs(normal[1]):
            if normal[0] < 0:
                normal *= -1
        else:
            if normal[1] < 0:
                normal *= -1

        if normal[2] < 0:
            normal *= -1
            d *= -1

        target = np.array([0.0, 0.0, 1.0])

        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c = np.dot(normal, target)

        if s < 1e-8:
            R = np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])

            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        rotated = (R @ points.T).T

        floor_z = np.mean(rotated[inliers, 2])
        rotated[:, 2] -= floor_z

        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3] = -floor_z

        return rotated, T, plane_model

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

        distance_threshold = voxel_size * 1.5

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
                100000,
                0.999
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

        distance_threshold = voxel_size * 0.5

        self.get_logger().info(
            "Running Fast Global Registration..."
        )

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold,
                iteration_number=128,
                maximum_tuple_count=1000,
                tuple_scale=0.95,
                tuple_test=True
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

    def publish_pointcloud(self, points, frame_id="map"):
        """
        Publish Nx3 numpy array as ROS2 PointCloud2
        """

        points = np.asarray(points, dtype=np.float32)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        cloud_msg = point_cloud2.create_cloud(
            header,
            fields,
            points
        )

        self.publisher_pointcloud.publish(cloud_msg)

        self.get_logger().info(
            f"Published point cloud with {len(points)} points"
        )

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

            scene, points, min_bound, max_bound = self.load_pcd_and_process()
            template = o3d.io.read_point_cloud(self.template_file)
            
            template = template.voxel_down_sample(
                self.voxel_size
            )

            # -------------------------------------------------------
            # REMOVE FLOOR
            # -------------------------------------------------------

            scene = self.remove_floor(
                scene,
                distance_threshold=0.015
            )

            # Convert back to numpy for RViz publishing
            scene_no_floor_points = np.asarray(scene.points)

            self.get_logger().info(
                f"Scene after floor removal: "
                f"{len(scene_no_floor_points)} points"
            )

            scene.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2,
                    max_nn=30
                )
            )

            template.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2,
                    max_nn=30
                )
            )


            # -------------------------------------------------------
            # Publish FLOOR-REMOVED cloud to RViz2
            # -------------------------------------------------------
            # 
            self.publish_pointcloud(
                scene_no_floor_points,
                frame_id="world"
            )

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