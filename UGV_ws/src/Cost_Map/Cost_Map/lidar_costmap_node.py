#!/usr/bin/env python3
"""
lidar_costmap_generator.py
==========================
ROS2 node that builds occupancy-grid costmaps from a LiDAR point cloud
stored in a PCD file.

Three costmap layers are published:
  /static_costmap    – raw terrain classification (free / caution / obstacle)
  /inflation_costmap – static layer with an obstacle buffer zone
  /master_costmap    – element-wise maximum of the two layers above

The node exposes two services:
  /generate_costmap  – (re-)generate and publish all layers
  /set_thresholds    – update classification parameters at runtime
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import maximum_filter, minimum_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from threshold_interfaces.srv import SetThresholds


# - Cost values used throughout the node
COST_FREE     = 10   # Flat, traversable terrain
COST_CAUTION  = 50   # Mildly rough terrain – proceed carefully
COST_BUFFER   = 100   # Inflation buffer around obstacles
COST_OBSTACLE = 100  # Impassable terrain


class LidarCostmapGenerator(Node):
    """
    Generates multi-layer occupancy-grid costmaps from a static PCD file.

    Processing pipeline (triggered by /generate_costmap):
      1. Load and align the point cloud so the floor lies on the XY plane.
      2. Rasterise points into an elevation map (average Z per grid cell).
      3. Estimate terrain roughness via local max-minus-min filtering.
      4. Classify cells into free / caution / obstacle → static layer.
      5. Dilate obstacles by an inflation radius → inflation layer.
      6. Combine both layers element-wise → master layer.
      7. Publish all three layers as OccupancyGrid messages and save to disk.
    """

    def __init__(self):
        super().__init__('lidar_costmap_generator')

        self._declare_parameters()
        self._read_parameters()
        self._create_publishers()
        self._create_services()

        # Broadcast a static TF so RViz2 can resolve the 'map' frame.
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_map_frame()

        self.get_logger().info(
            "Service ready → ros2 service call /generate_costmap std_srvs/srv/Trigger '{}'"
        )
        self.get_logger().info(
            "Service ready → ros2 service call /set_thresholds "
            "threshold_interfaces/srv/SetThresholds '{}'"
        )

        # In __init__, after your existing publishers:
        self._latest_costmap: OccupancyGrid | None = None


    def _declare_parameters(self):
        """Register all node parameters with their default values."""
        self.declare_parameter('pcd_file_path',             'src/Cost_Map/resource/WALLR2.pcd')
        self.declare_parameter('map_frame_id',              'map')
        self.declare_parameter('costmap_resolution',        0.05)
        self.declare_parameter('inflation_radius_meters',   0.2) # 0.05
        self.declare_parameter('flat_caution_threshold',    0.01)
        self.declare_parameter('caution_obstacle_threshold', 0.08)
        # Neighbourhood size for the local-extrema filters used in roughness estimation.
        # Larger values smooth noise but may blur small obstacles.
        self.declare_parameter('max_filter_size', 2)
        self.declare_parameter('min_filter_size', 2)

    def _read_parameters(self):
        """Cache parameter values as instance attributes for convenient access."""
        def get_str(name):  return self.get_parameter(name).get_parameter_value().string_value
        def get_f64(name):  return self.get_parameter(name).get_parameter_value().double_value
        def get_int(name):  return self.get_parameter(name).get_parameter_value().integer_value

        self.pcd_file_path              = get_str('pcd_file_path')
        self.map_frame_id               = get_str('map_frame_id')
        self.costmap_resolution         = get_f64('costmap_resolution')
        self.inflation_radius_meters    = get_f64('inflation_radius_meters')
        self.flat_caution_threshold     = get_f64('flat_caution_threshold')
        self.caution_obstacle_threshold = get_f64('caution_obstacle_threshold')
        self.max_filter_size            = get_int('max_filter_size')
        self.min_filter_size            = get_int('min_filter_size')

    def _create_publishers(self):
        """Create all OccupancyGrid and PointCloud2 publishers.

        TRANSIENT_LOCAL durability ensures late-joining subscribers (e.g. RViz2)
        receive the last published message immediately upon connection.
        """
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_master    = self.create_publisher(OccupancyGrid, '/master_costmap',    qos)
        #self.pub_static    = self.create_publisher(OccupancyGrid, '/static_costmap',    qos)
        #self.pub_inflation = self.create_publisher(OccupancyGrid, '/inflation_costmap', qos)
        self.pub_cloud     = self.create_publisher(PointCloud2,   '/pointcloud',        qos)

    def _create_services(self):
        """Advertise the generate and threshold-update services."""
        self.create_service(Trigger,        'generate_costmap', self._on_generate_costmaps)
        self.create_service(SetThresholds,  'set_thresholds',   self._on_set_thresholds)

    # - Service handlers

    def _on_set_thresholds(self, request, response):
        """Update classification parameters at runtime without restarting the node."""
        self.flat_caution_threshold     = request.flat_caution_threshold
        self.caution_obstacle_threshold = request.caution_obstacle_threshold
        self.costmap_resolution         = request.map_resolution
        self.inflation_radius_meters    = request.inflation_radius_meters
        self.max_filter_size            = request.max_filter_size
        self.min_filter_size            = request.min_filter_size

        self.get_logger().info(
            f"Thresholds updated → "
            f"flat={self.flat_caution_threshold}, "
            f"caution={self.caution_obstacle_threshold}, "
            f"resolution={self.costmap_resolution}, "
            f"inflation={self.inflation_radius_meters}, "
            f"max_filter={self.max_filter_size}, "
            f"min_filter={self.min_filter_size}"
        )

        response.success = True
        response.message = "Thresholds updated"
        return response

    def _on_generate_costmaps(self, request, response):
        """Load the PCD, compute all costmap layers, publish, and save to disk."""
        pcd, points, min_bound, max_bound = self._load_and_align_pcd()
        if pcd is None:
            response.success = False
            response.message = "Failed to load PCD — check pcd_file_path."
            return response

        width_cells  = int((max_bound[0] - min_bound[0]) / self.costmap_resolution) + 1
        height_cells = int((max_bound[1] - min_bound[1]) / self.costmap_resolution) + 1
        self.get_logger().info(f"Grid: {width_cells} x {height_cells} cells")

        elevation_map, has_data = self._compute_elevation_map(
            points, min_bound, width_cells, height_cells
        )

        static    = self._build_static_layer(elevation_map, has_data)
        inflation = self._build_inflation_layer(static)
        master    = np.maximum(static, inflation)

        self._publish_costmap(master, width_cells, height_cells, min_bound, '/master_costmap', self.pub_master)
        #self._save_costmap_to_npy(master, min_bound)

        # Pass the grid directly — no subscription needed
        points = self._filter_by_costmap_grid(points, master, min_bound, threshold=99)

        self._publish_pointcloud(points)

        response.success = True
        response.message = f"Costmaps generated: {width_cells}x{height_cells} cells"
        self.get_logger().info(response.message)
        return response


    def _filter_by_costmap(self, points: np.ndarray,
                            threshold: int = 99) -> np.ndarray:
        """Keep only points whose (x, y) falls in a costmap cell >= threshold."""
        if self._latest_costmap is None:
            self.get_logger().warn('No costmap received — skipping costmap filter.')
            return points

        info = self._latest_costmap.info
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        w, h = info.width, info.height
        grid = np.array(self._latest_costmap.data, dtype=np.int16).reshape(h, w)

        col = np.floor((points[:, 0] - ox) / res).astype(np.int32)
        row = np.floor((points[:, 1] - oy) / res).astype(np.int32)

        in_bounds = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        cost_values = np.full(len(points), -1, dtype=np.int16)
        cost_values[in_bounds] = grid[row[in_bounds], col[in_bounds]]

        mask = cost_values >= threshold
        filtered = points[mask]  # plain numpy boolean index — no Open3D needed
        self.get_logger().info(
            f'Costmap filter: {len(points):,} → {len(filtered):,} pts '
            f'(threshold={threshold})'
        )
        return filtered

    def _filter_by_costmap_grid(self, points: np.ndarray, grid: np.ndarray,
                                min_bound: np.ndarray, threshold: int = 99) -> np.ndarray:
        """Filter points using the locally computed costmap grid (no subscription needed)."""
        col = np.floor((points[:, 0] - min_bound[0]) / self.costmap_resolution).astype(np.int32)
        row = np.floor((points[:, 1] - min_bound[1]) / self.costmap_resolution).astype(np.int32)

        h, w = grid.shape
        in_bounds = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        cost_values = np.full(len(points), -1, dtype=np.int16)
        cost_values[in_bounds] = grid[row[in_bounds], col[in_bounds]]

        mask = cost_values >= threshold
        filtered = points[mask]
        self.get_logger().info(
            f'Costmap filter: {len(points):,} → {len(filtered):,} pts (threshold={threshold})'
        )
        return filtered

    # - TF

    def _publish_map_frame(self):
        """Publish a zero-offset static transform from 'world' to the map frame."""
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id  = self.map_frame_id
        t.transform.rotation.w = 1.0  # identity rotation
        self._tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Static TF published: world → {self.map_frame_id}")

    # - Point-cloud processing

    def _load_and_align_pcd(self):
        """
        Load a PCD file from disk and align it so the floor lies on the XY plane.

        Returns (pcd, points, min_bound, max_bound), or (None, …) on failure.
        """
        try:
            pcd = o3d.io.read_point_cloud(self.pcd_file_path)
            if not pcd.has_points():
                self.get_logger().error(f"PCD file has no points: {self.pcd_file_path}")
                return None, None, None, None

            points = np.asarray(pcd.points)
            points, _transform, _plane = self._align_floor_to_xy_plane(points)
            pcd.points = o3d.utility.Vector3dVector(points)

            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            self.get_logger().info(
                f"Loaded {len(points)} points  |  "
                f"XY min={min_bound[:2]}  max={max_bound[:2]}"
            )
            return pcd, points, min_bound, max_bound

        except Exception as exc:
            self.get_logger().error(f"Failed to load PCD: {exc}")
            return None, None, None, None

    def _align_floor_to_xy_plane(
        self,
        points,
        distance_threshold: float = 0.02,
        ransac_n:           int   = 3,
        num_iterations:     int   = 1000,
    ):
        """
        Fit a ground plane with RANSAC, then rotate and translate the entire
        cloud so that plane coincides with Z = 0.

        Returns:
            rotated    – Nx3 array of transformed points
            T          – 4×4 homogeneous transform that was applied
            plane_model – (a, b, c, d) coefficients of the fitted plane
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal)

        # Ensure the normal points upward (+Z direction).
        if normal[2] < 0:
            normal *= -1
            d      *= -1

        # Build the rotation matrix that maps `normal` onto [0, 0, 1].
        target = np.array([0.0, 0.0, 1.0])
        v      = np.cross(normal, target)
        s      = np.linalg.norm(v)       # sine of the rotation angle
        cos_a  = np.dot(normal, target)  # cosine of the rotation angle

        if s < 1e-8:
            # Vectors are already (nearly) parallel — no rotation needed.
            R = np.eye(3)
        else:
            # Rodrigues' rotation formula via the skew-symmetric cross product matrix.
            vx = np.array([
                [ 0,    -v[2],  v[1]],
                [ v[2],  0,    -v[0]],
                [-v[1],  v[0],  0   ],
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - cos_a) / (s ** 2))

        rotated = (R @ points.T).T

        # Translate so the average Z of the inlier floor points is exactly 0.
        floor_z = np.mean(rotated[inliers, 2])
        rotated[:, 2] -= floor_z

        T = np.eye(4)
        T[:3, :3] = R
        T[2,  3 ] = -floor_z

        return rotated, T, plane_model

    def _compute_elevation_map(self, points, min_bound, width_cells, height_cells):
        """
        Rasterise the point cloud into a 2-D grid of average Z values.

        Each cell accumulates the Z values of all points that fall within it;
        the mean is taken as the representative elevation.

        Returns:
            elevation_map – (height_cells × width_cells) float32 array of mean Z
            has_data      – boolean mask; True where at least one point fell
        """
        z_sums   = np.zeros((height_cells, width_cells), dtype=np.float32)
        z_counts = np.zeros((height_cells, width_cells), dtype=np.int32)

        x_idx = ((points[:, 0] - min_bound[0]) / self.costmap_resolution).astype(int)
        y_idx = ((points[:, 1] - min_bound[1]) / self.costmap_resolution).astype(int)

        in_bounds = (
            (x_idx >= 0) & (x_idx < width_cells) &
            (y_idx >= 0) & (y_idx < height_cells)
        )

        np.add.at(z_sums,   (y_idx[in_bounds], x_idx[in_bounds]), points[in_bounds, 2])
        np.add.at(z_counts, (y_idx[in_bounds], x_idx[in_bounds]), 1)

        elevation_map = np.divide(
            z_sums, z_counts,
            out=np.zeros_like(z_sums),
            where=z_counts != 0,
        )
        has_data = z_counts > 0
        return elevation_map, has_data

    # - Costmap layer construction

    def _build_static_layer(self, elevation_map, has_data):
        """
        Classify each grid cell by terrain roughness.

        Roughness is approximated as the difference between the locally maximum
        and minimum elevation values (using a small neighbourhood filter).
        Cells with no LiDAR returns are left as -1 (unknown).

        Cost mapping:
          -1  unknown  (no data)
          10  free     (roughness ≤ flat_caution_threshold)
          50  caution  (flat_caution_threshold < roughness ≤ caution_obstacle_threshold)
         100  obstacle (roughness > caution_obstacle_threshold)
        """
        roughness = (
            maximum_filter(elevation_map, size=self.max_filter_size) -
            minimum_filter(elevation_map, size=self.min_filter_size)
        )

        static = np.full(elevation_map.shape, -1, dtype=np.int16)

        static[has_data & (roughness <= self.flat_caution_threshold)]       = COST_FREE
        static[has_data & (roughness >  self.flat_caution_threshold) &
                          (roughness <= self.caution_obstacle_threshold)]   = COST_CAUTION
        static[has_data & (roughness >  self.caution_obstacle_threshold)]   = COST_OBSTACLE

        self.get_logger().info(
            f"Static layer — "
            f"free:{np.sum(static == COST_FREE)}  "
            f"caution:{np.sum(static == COST_CAUTION)}  "
            f"obstacle:{np.sum(static == COST_OBSTACLE)}  "
            f"unknown:{np.sum(static == -1)}"
        )
        return static

    def _build_inflation_layer(self, static):
        """
        Dilate the obstacle footprint outward by inflation_radius_meters.

        Cells that fall inside the inflated zone but are not themselves obstacles
        are marked COST_BUFFER (90), warning the planner to steer clear.
        Unknown cells (-1) are never upgraded — inflation only applies where we
        have actual sensor data.
        """
        obstacle_mask = (static == COST_OBSTACLE).astype(np.uint8) * 255

        # Ensure at least one cell of inflation even for sub-resolution radii.
        inflation_cells = max(1, int(self.inflation_radius_meters / self.costmap_resolution))
        kernel_size     = 2 * inflation_cells + 1
        kernel          = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        inflated_mask   = cv2.dilate(obstacle_mask, kernel, iterations=1)

        inflation = static.copy()

        # Buffer zone: inflated area that isn't itself an obstacle and has sensor data.
        buffer_zone = (inflated_mask > 0) & (static < COST_OBSTACLE) & (static >= 0)
        inflation[buffer_zone] = np.maximum(inflation[buffer_zone], COST_BUFFER)

        return inflation

    # - Publishing helpers

    def _publish_pointcloud(self, points):
        """Publish an Nx3 numpy array as a ROS2 PointCloud2 message."""
        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame_id

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = point_cloud2.create_cloud(header, fields, points.astype(np.float32))
        self.pub_cloud.publish(cloud_msg)
        self.get_logger().info(f"Published point cloud with {len(points)} points")

    def _publish_costmap(self, data_int16, width, height, min_bound, topic_name, publisher):
        """
        Pack a 2-D int16 cost grid into an OccupancyGrid message and publish it.

        OccupancyGrid uses int8 in the range [-1, 100], so values are clamped
        before casting to avoid silent wrap-around.
        """
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.info.resolution = float(self.costmap_resolution)
        msg.info.width      = width
        msg.info.height     = height
        msg.info.origin.position.x    = float(min_bound[0])
        msg.info.origin.position.y    = float(min_bound[1])
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        clamped  = np.clip(data_int16, -1, 100)
        msg.data = clamped.flatten().astype(np.int8).tolist()

        arr = np.array(msg.data)
        self.get_logger().info(
            f"Publishing {topic_name}  |  cells={len(msg.data)}  "
            f"min={arr.min()}  max={arr.max()}  "
            f"with data: {np.sum(arr >= 0)}"
        )
        publisher.publish(msg)

    # - Save to disk

    def _find_save_folder(self):
        """
        Locate (or create) the output directory for .npy costmap files.

        Walks up the directory tree from this file looking for the workspace's
        src/Cost_Map/Cost_Map_Coordinates folder.  Falls back to a sibling
        directory of this script if the workspace layout is not found.
        """
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "src" / "Cost_Map" / "Cost_Map_Coordinates"
            if candidate.exists():
                return candidate

        self.get_logger().warn("Could not find workspace src — saving next to installed file.")
        return Path(__file__).parent / "Cost_Map_Coordinates"

    def _save_costmap_to_npy(self, data_int16, min_bound):
        """
        Persist the master costmap as a timestamped .npy file.

        Each row of the saved array is [world_x, world_y, cost] for every cell
        that has sensor data (cost ≥ 0).  Unknown cells are omitted.
        """
        try:
            save_dir = self._find_save_folder()
            save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Saving costmap to: {save_dir.resolve()}")

            rows, cols = np.where(data_int16 >= 0)
            coords_and_costs = np.column_stack([
                min_bound[0] + cols * self.costmap_resolution,   # world X
                min_bound[1] + rows * self.costmap_resolution,   # world Y
                data_int16[rows, cols],                          # cost value
            ])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath  = save_dir / f"costmap_{timestamp}.npy"
            np.save(filepath, coords_and_costs)
            self.get_logger().info(f"Saved {len(coords_and_costs)} cells → {filepath}")

        except Exception as exc:
            self.get_logger().error(f"Failed to save costmap: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarCostmapGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()