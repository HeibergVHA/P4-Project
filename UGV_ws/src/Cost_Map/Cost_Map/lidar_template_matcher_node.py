#!/usr/bin/env python3
"""
Lidar Template Matcher Node
============================
Triggered by a ROS2 service call. Loads a scene PCD and a template PCD,
then finds the 6-DOF pose of the template object (RC car) in the scene.

Pipeline:
  1. Load & align scene floor to the XY-plane
  2. Remove floor points
  3. Z-height filter  (keep only car-height band)
  4. FGR global registration  (coarse alignment)
  5. Scene crop around FGR result  (prevents ICP from drifting)
  6. Multi-scale ICP refinement  (fine alignment)
  7. Publish pose + aligned point cloud

Call the service:
  ros2 service call /run_template_matching std_srvs/srv/Trigger '{}'
"""

import os
import copy
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Pose2D
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from nav_msgs.msg import OccupancyGrid


class LidarTemplateMatcherNode(Node):

    def __init__(self):
        super().__init__('lidar_template_matcher_node')

        # - Parameters 

        try:
            pkg = get_package_share_directory('Cost_Map')
        except PackageNotFoundError:
            pkg = os.path.join(os.getcwd(), 'src', 'Cost_Map')

        self.declare_parameter('scene_file',
            os.path.join(pkg, 'resource', 'WALLR2.pcd'))
        self.declare_parameter('template_file',
            os.path.join(pkg, 'resource', 'rc_car_template.pcd'))

        # Voxel leaf size (metres).
        # 0.02 m works well for a 1.9 M-point scene — coarse enough for
        # distinctive FPFH features, fine enough to resolve the car shape.
        self.declare_parameter('voxel_size', 0.02)

        # Z band to keep after floor removal (metres, relative to floor = 0).
        # The RC-car template sits at z ≈ -0.50 → +0.18 before floor
        # alignment, so a [-0.10, 0.55] band captures the car while
        # discarding the ground and high clutter.
        self.declare_parameter('z_min',  -0.55)
        self.declare_parameter('z_max',   0.1)

        # Expand the crop box around the FGR result before running ICP.
        # Larger values are safer but slow ICP down.
        self.declare_parameter('icp_crop_margin', 0.2)

        self.scene_file    = self.get_parameter('scene_file').value
        self.template_file = self.get_parameter('template_file').value
        self.voxel_size    = self.get_parameter('voxel_size').value
        self.z_min         = self.get_parameter('z_min').value
        self.z_max         = self.get_parameter('z_max').value
        self.icp_margin    = self.get_parameter('icp_crop_margin').value

        # - Subscribers
        self._latest_costmap = None
        self._latest_costmap_npy_info = None
        self._latest_costmap_npy_path = None
        self._costmap_coords_dir = self._find_costmap_coordinates_folder()

        costmap_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/costmap/costmap',
            self._costmap_callback,
            costmap_qos,
        )

        # - Publishers

        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        #self.publisher_ = self.create_publisher(Pose2D, '/rover_pose_2D', 10)

        # TRANSIENT_LOCAL so RViz2 receives the cloud even if it subscribes late
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.cloud_pub = self.create_publisher(PointCloud2, '/pointcloud', qos)

        # - Service 

        self.srv = self.create_service(Trigger, 'run_template_matching',
                                       self._match_callback)

        self.get_logger().info(
            'Ready — call:  ros2 service call /run_template_matching std_srvs/srv/Trigger \'{}\''
        )

    # Service callback - top-level orchestration

    def _costmap_callback(self, msg: OccupancyGrid):
        self._latest_costmap = msg

        self.get_logger().info(
            f'Received costmap: '
            f'{msg.info.width}x{msg.info.height} '
            f'res={msg.info.resolution:.3f}'
        )

    def _match_callback(self, request, response):
        try:
            # 1. Load scene & rotate floor to XY plane
            scene = self._load_scene()

            # 2. Remove floor plane
            #scene = self._remove_floor(scene)

            # 3b. Costmap filter — keep only obstacle-region points
            scene = self._filter_by_costmap(scene, threshold=99)

            if len(scene.points) < 100:
                raise RuntimeError(
                    f'Costmap filter left only {len(scene.points)} points — '
                    'check coordinate alignment or lower the threshold.'
                )

            # 3. Z-height filter — discard ground noise and tall obstacles
            scene = self._filter_z(scene)

            # Publish the cleaned scene to RViz2
            self._publish_cloud(np.asarray(scene.points))

            # 4. Load & downsample template
            template = self._load_template()

            # 5. Compute FPFH features for both clouds
            scene_down,    scene_fpfh    = self._preprocess(scene)
            template_down, template_fpfh = self._preprocess(template)

            self.get_logger().info(
                f'Scene: {len(scene_down.points)} pts  |  '
                f'Template: {len(template_down.points)} pts  (after voxel={self.voxel_size} m)'
            )

            # 6. FGR global registration (coarse)
            #fgr = self._fgr(template_down, scene_down, template_fpfh, scene_fpfh)
            fgr = self._ransac(template_down, scene_down, template_fpfh, scene_fpfh)

            #self.get_logger().info(f'Scene FPFH shape: {np.asarray(scene_fpfh.data).shape}')
            #self.get_logger().info(f'Template FPFH shape: {np.asarray(template_fpfh.data).shape}')
            #self.get_logger().info(f'Scene points: {len(scene_down.points)}')
            #self.get_logger().info(f'Template points: {len(template_down.points)}')

            # If no correspondences found, FGR failed to find any matches at all
            if len(fgr.correspondence_set) == 0 or fgr.fitness < 0.05:
                raise RuntimeError(
                    f'RANSAC failed: fitness={fgr.fitness:.4f}, '
                    f'correspondences={len(fgr.correspondence_set)}'
                )

            self.get_logger().info(f'ransac   fitness={fgr.fitness:.4f}  rmse={fgr.inlier_rmse:.4f}')

            # 7. Crop scene to FGR bounding box — keeps ICP from drifting
            #scene_cropped = self._crop_around_result(scene_down, template_down, fgr.transformation)
            """scene_cropped = self._crop_around_result(
                scene,
                template_down,
                fgr.transformation
            )
            self.get_logger().info(
                f'Scene cropped to {len(scene_cropped.points)} pts around FGR result'
            )"""

            # 8. Multi-scale ICP refinement (fine)
            #icp = self._icp(template_down, scene_down, fgr.transformation)
            #icp = fgr.transformation
            #self.get_logger().info(f'ICP   fitness={icp.fitness:.4f}  rmse={icp.inlier_rmse:.4f}')

            # Use ICP result if it improved on FGR, otherwise keep FGR
            #final_T = icp.transformation if icp.fitness >= fgr.fitness else fgr.transformation
            final_T = fgr.transformation
            final_T = np.array(final_T)

            # Rotate result by 90 degree around z-axis to match the template's forward direction with the car's forward direction in the scene
            #R = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0], [np.sin(-np.pi/2), np.cos(-np.pi/2),0], [0,0,1]])
            #final_T[:3, :3] = final_T[:3, :3] @ R


            # Store final position as (x, y, yaw) for later use
            #final_pos = np.array([final_T[0, 3], final_T[1, 3], np.arctan2(final_T[2, 1], final_T[1, 1]) - np.pi/2])

            #final_R = np.array([[np.cos(final_pos[2]), -np.sin(final_pos[2]), 0], [np.sin(final_pos[2]), np.cos(final_pos[2]),0], [0,0,1]])
            #final_T[:3, :3] = final_R
            #final_T[:3, 2] = 0

            self.get_logger().info(
                f'Template centroid: {self.template_centroid}\n'
                f'Final T:\n{final_T}\n'
                f'Translation: [{final_T[0,3]:.3f}, {final_T[1,3]:.3f}, {final_T[2,3]:.3f}]\n'
                f'Scene pts after filter: {len(scene.points)}\n'
                f'RANSAC fitness: {fgr.fitness:.4f}  correspondences: {len(fgr.correspondence_set)}'
            )

            # Publish the 2D pose (x, y, theta) for use by other nodes
            #self._publish_pose_2D(final_T)

            # Save transform matrix as numpy array for later use 
            # THIS IS THE ONE TO UNCOMMENT
            self._save_transform(final_T)

            # 9. Publish pose
            self._publish_pose(final_T)

            response.success = True
            response.message = (
                f'Match complete. '
                f'FGR fitness={fgr.fitness:.3f}' #  ICP fitness={icp.fitness:.3f
            )

        except Exception as e:
            self.get_logger().error(f'Matching failed: {e}')
            response.success = False
            response.message = str(e)

        return response

    # Step 1 - Load scene and align floor to Z=0

    def _load_scene(self) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(self.scene_file)
        if not pcd.has_points():
            raise RuntimeError(f'Scene PCD has no points: {self.scene_file}')

        pts = np.asarray(pcd.points, dtype=np.float64)
        pts, self.floor_T = self._align_floor(pts)
        pcd.points = o3d.utility.Vector3dVector(pts)

        self.get_logger().info(
            f'Scene loaded: {len(pts):,} pts  '
            f'X=[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  '
            f'Z=[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]'
        )
        return pcd

    def _align_floor(self, points: np.ndarray):
        """
        Fit a plane to the dominant flat surface (floor), rotate so that
        plane normal points along +Z, then shift so floor sits at Z=0.
        Returns (rotated_points, 4×4_transform).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02, ransac_n=3, num_iterations=1000)

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal)

        # Make the normal point upward (toward +Z)
        if normal[2] < 0:
            normal *= -1

        # Rodrigues rotation: normal → [0, 0, 1]
        target = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, target)
        s = np.linalg.norm(v)

        if s < 1e-8:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - np.dot(normal, target)) / s ** 2)

        rotated = (R @ points.T).T

        # Translate so the floor plane sits exactly at Z=0
        floor_z = np.mean(rotated[inliers, 2])
        rotated[:, 2] -= floor_z

        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3] = -floor_z

        return rotated, T

    # Step 2 - Remove floor plane

    def _remove_floor(self, pcd: o3d.geometry.PointCloud,
                      dist_thresh=0.06) -> o3d.geometry.PointCloud:
        """
        Segment the dominant plane (floor) and return only the non-floor points.
        After _align_floor the floor is at Z≈0, so a tight threshold works well.
        """
        _, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=2000)

        no_floor = pcd.select_by_index(inliers, invert=True)

        self.get_logger().info(
            f'Floor removal: {len(inliers)} floor pts removed, '
            f'{len(no_floor.points)} remaining'
        )
        return no_floor
    
    # Step 3 - Z-height filter

    def _filter_z(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Keep only points within [z_min, z_max] above the floor.

        Why this matters for WALLR1.pcd:
          - 1.93 M points spread across Z = -7 → +4 m
          - The RC car sits in Z ≈ 0 → 0.7 m (after floor alignment)
          - Wall returns, ceiling, and ground noise outside this band
            pollute FPFH features and pull ICP to the wrong solution
        """
        pts = np.asarray(pcd.points)
        mask = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        filtered = pcd.select_by_index(np.where(mask)[0])

        self.get_logger().info(
            f'Z-filter [{self.z_min:.2f}, {self.z_max:.2f}] m: '
            f'{len(pcd.points):,} → {len(filtered.points):,} pts'
        )
        return filtered

    def _filter_by_costmap(self, pcd: o3d.geometry.PointCloud,
                            threshold: int = 253) -> o3d.geometry.PointCloud:
        """Keep only points whose (x, y) falls in a costmap cell >= threshold."""
        if self._latest_costmap_npy_info is None:
            data = self._wait_for_latest_costmap_npy()
            self._latest_costmap_npy_info = self._prepare_costmap_from_npy(data)

        if self._latest_costmap_npy_info is None:
            if self._latest_costmap is None:
                self.get_logger().warn('No costmap data available — skipping costmap filter.')
                return pcd
            info   = self._latest_costmap.info
            res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
            w, h   = info.width, info.height
            grid   = np.array(self._latest_costmap.data, dtype=np.int16).reshape(h, w)
        else:
            info = self._latest_costmap_npy_info
            res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
            w, h = info['width'], info['height']
            grid = info['grid']

        pts = np.asarray(pcd.points)
        col = np.floor((pts[:, 0] - ox) / res).astype(np.int32)
        row = np.floor((pts[:, 1] - oy) / res).astype(np.int32)

        in_bounds = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        cost_values = np.full(len(pts), -1, dtype=np.int16)
        cost_values[in_bounds] = grid[row[in_bounds], col[in_bounds]]

        mask = cost_values >= threshold
        filtered = pcd.select_by_index(np.where(mask)[0])
        self.get_logger().info(
            f'Costmap filter: {len(pts):,} → {len(filtered.points):,} pts '
            f'(threshold={threshold})'
        )
        return filtered

    def _find_costmap_coordinates_folder(self):
        for parent in Path(__file__).resolve().parents:
            candidate = parent / 'src' / 'Cost_Map' / 'Cost_Map_Coordinates'
            if candidate.exists():
                return candidate

        self.get_logger().warn(
            'Could not locate Cost_Map_Coordinates folder; .npy costmap loading may not be available.'
        )
        return None

    def _wait_for_latest_costmap_npy(self) -> np.ndarray:
        if self._costmap_coords_dir is None:
            raise RuntimeError('Cost_Map_Coordinates folder not found')

        while True:
            npy_files = sorted(
                self._costmap_coords_dir.glob('*.npy'),
                key=lambda path: path.stat().st_mtime
            )
            if npy_files:
                latest = npy_files[-1]
                data = np.load(latest)
                if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
                    self._latest_costmap_npy_path = latest
                    self.get_logger().info(
                        f'Loaded latest costmap file: {latest.name} '
                        f'(shape={data.shape})'
                    )
                    return data
                self.get_logger().warn(
                    f'Skipping invalid .npy file: {latest.name} shape={getattr(data, "shape", None)}'
                )
            self.get_logger().info(
                f'Waiting for a .npy file in: {self._costmap_coords_dir}'
            )
            time.sleep(1.0)

    def _prepare_costmap_from_npy(self, raw: np.ndarray):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise RuntimeError(f'Invalid costmap .npy shape: {raw.shape}')

        xs = raw[:, 0]
        ys = raw[:, 1]
        costs = raw[:, 2].astype(np.int16)

        resolution = self._infer_costmap_resolution(xs, ys)
        min_x, min_y = float(xs.min()), float(ys.min())
        max_x, max_y = float(xs.max()), float(ys.max())

        width = int(round((max_x - min_x) / resolution)) + 1
        height = int(round((max_y - min_y) / resolution)) + 1
        grid = np.full((height, width), -1, dtype=np.int16)

        cols = np.rint((xs - min_x) / resolution).astype(np.int32)
        rows = np.rint((ys - min_y) / resolution).astype(np.int32)
        valid = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
        )
        grid[rows[valid], cols[valid]] = costs[valid]

        return {
            'grid': grid,
            'origin_x': min_x,
            'origin_y': min_y,
            'resolution': resolution,
            'width': width,
            'height': height,
        }

    def _infer_costmap_resolution(self, xs: np.ndarray, ys: np.ndarray) -> float:
        def infer(values: np.ndarray) -> float:
            uniq = np.unique(np.round(values, 6))
            if len(uniq) < 2:
                return 0.05
            diffs = np.diff(np.sort(uniq))
            diffs = diffs[diffs > 1e-6]
            return float(diffs.min()) if diffs.size else 0.05

        res_x = infer(xs)
        res_y = infer(ys)
        return min(res_x, res_y) if res_x and res_y else max(res_x, res_y, 0.05)

    # Step 4 - Load & downsample template

    def _load_template(self) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(self.template_file)
        if not pcd.has_points():
            raise RuntimeError(f'Template PCD has no points: {self.template_file}')

        centroid = pcd.get_center()
        pcd.translate(-centroid)
        self.template_centroid = centroid  # save to add back when publishing pose

        # The template already has pre-computed normals (x y z nx ny nz),
        # but we re-estimate them after downsampling for consistency.
        #pcd = pcd.voxel_down_sample(self.voxel_size)

        return pcd

    # Step 5 - Preprocessing: normals + FPFH features

    def _preprocess(self, pcd: o3d.geometry.PointCloud):
        """
        Downsample → estimate normals → compute FPFH features.
        Returns (downsampled_pcd, fpfh_feature).
        """
        down = pcd.voxel_down_sample(self.voxel_size)

        # Normal search radius: 2× voxel gives stable normals without being
        # too local (noisy) or too global (smoothed over edges)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size, max_nn=50))

        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=50))

        # Consistent orientation — required for Point-to-Plane ICP
        #down.orient_normals_consistent_tangent_plane(k=15)

        # FPFH radius: 5× voxel captures enough neighbourhood context for
        # distinctive descriptors; 100 neighbours caps memory usage
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))

        return down, fpfh

    # Step 6 - FGR global registration (coarse)

    def _fgr(self, src_down, tgt_down, src_fpfh, tgt_fpfh):
        """
        Fast Global Registration — faster and often more robust than RANSAC
        for partial-overlap scenes like a single object in a large map.

        distance_threshold = 0.5 × voxel_size is intentionally tight here
        because FGR uses tuple constraints internally; too loose causes false
        correspondences on a dense scene.
        """
        best = None
        for _ in range(10):
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=self.voxel_size,
                    iteration_number=256,        # more iterations → more reliable
                    maximum_tuple_count=2000,    # more tuples → better constraints
                    tuple_scale=0.95,
                    tuple_test=True,
                )
            )
            if best is None or result.fitness > best.fitness:
                best = result
            
            # Good enogh result, no need to keep trying random restarts
            if best.fitness >= 0.7:
                break
            
        return best

    def _ransac(self, src_down, tgt_down, src_fpfh, tgt_fpfh):
        """
        RANSAC Global Registration.

        More robust than FGR in highly noisy or ambiguous feature matches,
        but typically slower. Good for partial overlap and sparse correspondences.
        """
        best = None

        for _ in range(10):
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down,
                tgt_down,
                src_fpfh,
                tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=self.voxel_size * 5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 5),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100000, 0.999
                )
            )

            if best is None or result.fitness > best.fitness:
                best = result

            # Early stop if good enough alignment is found
            #if best.fitness >= 0.7:
            #    break

        return best

    # Step 7 - Crop scene around FGR result

    def _crop_around_result(self, scene_down, template_down,
                             transform: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Transform the template into the scene frame using the FGR result,
        compute its axis-aligned bounding box, expand it by icp_margin, and
        return only the scene points inside that box.

        Without this step ICP sees the entire 1.9 M-point scene and drifts
        toward the nearest dense cluster instead of refining the car match.
        """
        tmpl_in_scene = copy.deepcopy(template_down)
        tmpl_in_scene.transform(transform)

        bbox = tmpl_in_scene.get_axis_aligned_bounding_box()
        bbox.min_bound = bbox.min_bound - self.icp_margin
        bbox.max_bound = bbox.max_bound + self.icp_margin

        return scene_down.crop(bbox)

    # Step 8 - Multi-scale ICP refinement (fine)

    def _icp(self, src, tgt, init_T: np.ndarray):
        """
        Run ICP three times at decreasing correspondence distances
        (coarse → medium → fine).  Starting loose avoids the local-minimum
        trap that caused ICP to degrade the FGR result in earlier testing.
        """
        T = init_T
        for scale in [1.0, 0.5, 0.25]:
            dist = self.voxel_size * scale
            result = o3d.pipelines.registration.registration_icp(
                src, tgt,
                max_correspondence_distance=dist,
                init=T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100),
            )
            T = result.transformation

        return result

    # Publishers

    def _publish_cloud(self, points: np.ndarray, frame_id='map'):
        """Publish an Nx3 float32 array as sensor_msgs/PointCloud2."""
        points = points.astype(np.float32)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg = point_cloud2.create_cloud(header, fields, points)
        self.cloud_pub.publish(msg)
        self.get_logger().info(f'Published cloud: {len(points)} pts → /pointcloud')

    def _publish_pose(self, T: np.ndarray, frame_id='map'):
        """Publish the 4×4 transform as geometry_msgs/PoseStamped (identity quaternion)."""
        msg = PoseStamped()

        #T = np.linalg.inv(self.floor_T) @ T

        #T = np.linalg.inv(self.floor_T) @ T
        R = T[:3, :3]
        trace = R[0,0] + R[1,1] + R[2,2]
        w = np.sqrt(max(0, 1 + trace)) / 2
        x = (R[2,1] - R[1,2]) / (4*w)
        y = (R[0,2] - R[2,0]) / (4*w)
        z = (R[1,0] - R[0,1]) / (4*w)

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(T[0, 3]) # + self.template_centroid[0]
        msg.pose.position.y = float(T[1, 3]) # + self.template_centroid[1]
        msg.pose.position.z = float(T[2, 3]) # + self.template_centroid[2]
        msg.pose.orientation.w = float(w)
        msg.pose.orientation.x = float(x)
        msg.pose.orientation.y = float(y)
        msg.pose.orientation.z = float(z)
        self.pose_pub.publish(msg)
        self.get_logger().info(
            f'Published pose: x={T[0,3]:.3f}  y={T[1,3]:.3f}  z={T[2,3]:.3f}'
        )

    def _publish_pose_2D(self, pose):
        msg = Pose2D()

        msg.x = pose[0]
        msg.y = pose[1]
        msg.theta = pose[2]

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'Publishing: x={msg.x}, y={msg.y}, theta={msg.theta}'
        )

    def _save_transform(self, T: np.ndarray):
        """
        Save final 4x4 transform matrix as .npy file.
        """

        save_dir = os.path.join(
            os.getcwd(),
            'src',
            'Cost_Map',
            'TemplateMatching'
        )

        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"T_rover_{timestamp}.npy")

        np.save(save_path, T)

        self.get_logger().info(
            f'Saved transform to: {save_path}'
        )

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