#!/usr/bin/env python3
"""
ROS2 Point Cloud Template Matcher Node
=======================================
Matches an rc_car template PCD against incoming scene point clouds using:
  1. FPFH feature extraction
  2. RANSAC global registration (coarse alignment)
  3. Point-to-Plane ICP (fine alignment)

Subscribed Topics:
  /scene_cloud   (sensor_msgs/PointCloud2)  — live scene point cloud (e.g. from WALLR1.pcd)

Published Topics:
  /template_match/pose          (geometry_msgs/PoseStamped)   — estimated 6-DOF pose of the car
  /template_match/aligned_cloud (sensor_msgs/PointCloud2)     — template transformed into scene frame
  /template_match/score         (std_msgs/Float32)            — fitness score (lower = better fit)

Parameters (set via ros2 param or launch file):
  template_path      (string)  path to rc_car_template.pcd
  voxel_size         (float)   downsampling leaf size in metres          [default: 0.05]
  normal_radius      (float)   normal estimation radius multiplier       [default: 2.0  × voxel_size]
  fpfh_radius        (float)   FPFH feature radius multiplier            [default: 5.0  × voxel_size]
  ransac_dist        (float)   RANSAC inlier distance multiplier         [default: 1.5  × voxel_size]
  icp_dist           (float)   ICP max correspondence distance multiplier[default: 0.4  × voxel_size]
  fitness_threshold  (float)   publish only when fitness <= this value   [default: 0.3]
  min_scene_points   (int)     skip frames with fewer scene points       [default: 500]
"""

import time
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header
import struct


# ---------------------------------------------------------------------------
# Helpers: ROS2 <-> Open3D conversion
# ---------------------------------------------------------------------------

def ros2_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """Convert sensor_msgs/PointCloud2 to Open3D PointCloud (xyz only)."""
    field_map = {f.name: f.offset for f in msg.fields}
    point_step = msg.point_step
    n_points = msg.width * msg.height

    # Copy raw bytes into a contiguous structured array
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)

    xyz = np.zeros((n_points, 3), dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        off = field_map[axis]
        # Extract 4 bytes per point for this axis, copy to ensure contiguity
        xyz[:, i] = np.frombuffer(
            raw[:, off:off+4].copy().tobytes(), dtype=np.float32
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


def o3d_to_ros2(pcd: o3d.geometry.PointCloud, frame_id: str,
                stamp) -> PointCloud2:
    """Convert Open3D PointCloud to sensor_msgs/PointCloud2."""
    points = np.asarray(pcd.points, dtype=np.float32)
    n = len(points)

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12  # 3 × float32

    data = points.tobytes()

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.data = data
    msg.is_dense = True
    return msg


def matrix_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    """Convert a 4×4 homogeneous transform to PoseStamped."""
    from scipy.spatial.transform import Rotation as R  # lazy import

    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = stamp

    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])

    rot = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = rot.as_quat()  # scipy returns [x,y,z,w]
    ps.pose.orientation.x = float(qx)
    ps.pose.orientation.y = float(qy)
    ps.pose.orientation.z = float(qz)
    ps.pose.orientation.w = float(qw)

    return ps


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(pcd: o3d.geometry.PointCloud,
               voxel_size: float,
               normal_radius: float,
               fpfh_radius: float):
    """Downsample → estimate normals → compute FPFH features."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(k=15)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100)
    )
    return down, fpfh


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def global_registration(src_down, src_fpfh,
                         tgt_down, tgt_fpfh,
                         voxel_size: float,
                         ransac_dist: float):
    """RANSAC-based global registration."""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    return result


def refine_icp(src_down, tgt_down, init_transform, icp_dist):
    """Multi-scale ICP: coarse → fine to avoid local minima."""
    T = init_transform
    for dist in [icp_dist * 8, icp_dist * 4, icp_dist]:
        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=dist,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100),
        )
        T = result.transformation
    return result

def crop_scene_around_template(scene_down, template_down, transform, margin=1.0):
    """Crop scene to a box around where RANSAC placed the template."""
    tmpl_transformed = o3d.geometry.PointCloud(template_down)
    tmpl_transformed.transform(transform)
    bbox = tmpl_transformed.get_axis_aligned_bounding_box()
    # Expand box by margin in all directions
    bbox.min_bound = bbox.min_bound - margin
    bbox.max_bound = bbox.max_bound + margin
    return scene_down.crop(bbox)

# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class TemplateMatcherNode(Node):

    def __init__(self):
        super().__init__('lidar_template_matcher_node')

        # ---- Declare & read parameters ----
        self.declare_parameter('template_path', '')
        self.declare_parameter('voxel_size',        0.05)
        self.declare_parameter('normal_radius_mult', 2.0)
        self.declare_parameter('fpfh_radius_mult',   5.0)
        self.declare_parameter('ransac_dist_mult',   1.5)
        self.declare_parameter('icp_dist_mult',      0.4)
        self.declare_parameter('fitness_threshold',  0.3)
        self.declare_parameter('min_scene_points',   500)

        template_path = self.get_parameter('template_path').value
        self.voxel_size   = self.get_parameter('voxel_size').value
        nm = self.get_parameter('normal_radius_mult').value
        fm = self.get_parameter('fpfh_radius_mult').value
        rm = self.get_parameter('ransac_dist_mult').value
        im = self.get_parameter('icp_dist_mult').value

        self.normal_radius      = nm * self.voxel_size
        self.fpfh_radius        = fm * self.voxel_size
        self.ransac_dist        = rm * self.voxel_size
        self.icp_dist           = im * self.voxel_size
        self.fitness_threshold  = self.get_parameter('fitness_threshold').value
        self.min_scene_points   = self.get_parameter('min_scene_points').value

        # ---- Load & preprocess template ----
        if not template_path:
            self.get_logger().fatal('template_path parameter is not set! Shutting down.')
            raise SystemExit(1)

        self.get_logger().info(f'Loading template from: {template_path}')
        raw_template = o3d.io.read_point_cloud(template_path)
        if len(raw_template.points) == 0:
            self.get_logger().fatal('Template PCD is empty or could not be read.')
            raise SystemExit(1)

        self.get_logger().info(
            f'Template loaded: {len(raw_template.points)} points. '
            f'Preprocessing (voxel={self.voxel_size:.3f}m)…'
        )
        self.tmpl_down, self.tmpl_fpfh = preprocess(
            raw_template,
            self.voxel_size,
            self.normal_radius,
            self.fpfh_radius,
        )
        self.get_logger().info(
            f'Template preprocessed → {len(self.tmpl_down.points)} points after downsampling.'
        )

        # ---- Publishers ----
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_pose    = self.create_publisher(PoseStamped,   '/template_match/pose',          qos)
        self.pub_cloud   = self.create_publisher(PointCloud2,   '/template_match/aligned_cloud', qos)
        self.pub_fitness = self.create_publisher(Float32,       '/template_match/score',         qos)

        # ---- Subscriber ----
        self.sub = self.create_subscription(
            PointCloud2,
            '/scene_cloud',
            self.scene_callback,
            qos,
        )

        self._last_T   = np.eye(4)   # warm-start ICP across frames
        self._frame_no = 0

        self.get_logger().info('TemplateMatcherNode ready. Waiting for /scene_cloud …')

    # ------------------------------------------------------------------ #

    def scene_callback(self, msg: PointCloud2):
        t0 = time.perf_counter()
        self._frame_no += 1

        # Convert to Open3D
        scene_pcd = ros2_to_o3d(msg)
        n_raw = len(scene_pcd.points)

        if n_raw < self.min_scene_points:
            self.get_logger().warn(
                f'Frame {self._frame_no}: only {n_raw} scene points — skipping.'
            )
            return

        # Preprocess scene
        scene_down, scene_fpfh = preprocess(
            scene_pcd,
            self.voxel_size,
            self.normal_radius,
            self.fpfh_radius,
        )

        # ---- Global registration (RANSAC) ----
        ransac_result = global_registration(
            self.tmpl_down, self.tmpl_fpfh,
            scene_down,     scene_fpfh,
            self.voxel_size, self.ransac_dist,
        )

        # ---- Crop scene to RANSAC bbox before ICP ----
        scene_cropped = crop_scene_around_template(
            scene_down, self.tmpl_down, ransac_result.transformation, margin=1.0)

        self.get_logger().info(
            f'Scene cropped to {len(scene_cropped.points)} pts around RANSAC result '
            f'(was {len(scene_down.points)})'
        )

        # ---- Fine alignment (ICP) ----
        icp_result = refine_icp(
            self.tmpl_down, scene_cropped,
            ransac_result.transformation,
            self.icp_dist,
        )

        T        = icp_result.transformation          # 4×4 transform
        fitness  = icp_result.fitness                 # inlier ratio (higher = better)
        rmse     = icp_result.inlier_rmse
        score    = 1.0 - fitness                      # lower = better match (for threshold)

        dt = time.perf_counter() - t0

        if score > self.fitness_threshold:
            self.get_logger().warn(
                f'Score {score:.4f} > threshold {self.fitness_threshold} — match not confident.'
            )
            # Still publish score so the user can monitor
            self.pub_fitness.publish(Float32(data=float(score)))
            return

        # Warm-start for next frame
        self._last_T = T

        # ---- Publish pose ----
        pose_msg = matrix_to_pose_stamped(T, msg.header.frame_id, msg.header.stamp)
        self.pub_pose.publish(pose_msg)

        self.get_logger().info(
            f'Pose {pose_msg}'
            f'Frame {self._frame_no}: fitness={fitness:.4f}  rmse={rmse:.4f}  '
            f'score={score:.4f}  time={dt*1e3:.1f}ms'
        )

        # ---- Publish aligned template cloud ----
        aligned = o3d.geometry.PointCloud(self.tmpl_down)
        aligned.transform(T)
        cloud_msg = o3d_to_ros2(aligned, msg.header.frame_id, msg.header.stamp)
        self.pub_cloud.publish(cloud_msg)

        # ---- Publish fitness score ----
        self.pub_fitness.publish(Float32(data=float(score)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TemplateMatcherNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()