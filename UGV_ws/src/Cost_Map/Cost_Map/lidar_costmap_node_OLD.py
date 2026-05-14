#!/usr/bin/env python3

# ROS2 Core 
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

# ROS2 Messages & Services 
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

# Third-Party Libraries
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import maximum_filter, minimum_filter


# From numpy library 
from datetime import datetime
from pathlib import Path

# Custom Interfaces 
from threshold_interfaces.srv import SetThresholds

# ROS2 Node that generates costmaps from a LiDAR point cloud stored in a PCD file.
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2


class LidarCostmapGenerator(Node):
    

    def __init__(self):
        super().__init__('lidar_costmap_generator')
        self.get_logger().info("NODE STARTED")

        # ── Parameters ────────────────────────────────────────────────────
        #self.declare_parameter('pcd_file_path',
        #    '/home/jesper-kwame-jensen/Desktop/P4-Project/UGV_ws/src/Cost_Map/PCD_File/scans.pcd')
        #ros2_ws/PCD/name.pcd
        self.declare_parameter('pcd_file_path', 'src/Cost_Map/resource/WALLR2.pcd')  # <-- Update this path to your PCD file
        self.declare_parameter('map_frame_id',            'map')
        self.declare_parameter('costmap_resolution',      0.05)
        self.declare_parameter('inflation_radius_meters', 0.05)
        self.declare_parameter('flat_caution_threshold', 0.03)
        self.declare_parameter('caution_obstacle_threshold', 0.08)
        # These filter sizes control how we compute terrain roughness. 
        # Larger sizes will consider a bigger neighborhood, which can help smooth out noise but may also blur small obstacles.
        self.declare_parameter('max_filter_size', 2) 
        self.declare_parameter('min_filter_size', 2)

        # Read parameters once at startup; they can be updated via the /set_thresholds service.
        self.pcd_file_path           = self.get_parameter('pcd_file_path').get_parameter_value().string_value
        self.map_frame_id            = self.get_parameter('map_frame_id').get_parameter_value().string_value
        self.costmap_resolution      = self.get_parameter('costmap_resolution').get_parameter_value().double_value
        self.inflation_radius_meters = self.get_parameter('inflation_radius_meters').get_parameter_value().double_value
        self.flat_caution_threshold = self.get_parameter('flat_caution_threshold').get_parameter_value().double_value
        self.caution_obstacle_threshold = self.get_parameter('caution_obstacle_threshold').get_parameter_value().double_value
        self.min_filter_size = self.get_parameter('min_filter_size').get_parameter_value().integer_value
        self.max_filter_size = self.get_parameter('max_filter_size').get_parameter_value().integer_value

        # Standard QoS — make sure RViz2 is open and subscribed BEFORE calling
        # the service so it catches the publish.
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher_master_costmap    = self.create_publisher(OccupancyGrid, '/master_costmap', qos)
        self.publisher_static_costmap    = self.create_publisher(OccupancyGrid, '/static_costmap', qos)
        self.publisher_inflation_costmap = self.create_publisher(OccupancyGrid, '/inflation_costmap', qos)
        self.publisher_pointcloud = self.create_publisher(PointCloud2, '/pointcloud', qos)

        # Static TF so RViz2 can resolve the 'map' frame
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_map_frame()

        # Services
        self.srv = self.create_service(Trigger, 'generate_costmap', self.generate_costmaps_service)
        self.threshold_srv = self.create_service(SetThresholds,'set_thresholds',self.set_thresholds_callback)  

        #Log info to guide the user on how to call the services
        self.get_logger().info("Service ready → ros2 service call /generate_costmap std_srvs/srv/Trigger '{}'")
        self.get_logger().info("Service ready → ros2 service call /set_thresholds threshold_interfaces/srv/SetThresholds '{}'")

    # Service handler for updating thresholds at runtime    
    def set_thresholds_callback(self, request, response):
        self.flat_caution_threshold = request.flat_caution_threshold
        self.caution_obstacle_threshold = request.caution_obstacle_threshold
        self.costmap_resolution = request.map_resolution
        self.inflation_radius_meters = request.inflation_radius_meters
        self.max_filter_size = request.max_filter_size
        self.min_filter_size = request.min_filter_size

        self.get_logger().info(
            f"Updated thresholds → flat={self.flat_caution_threshold}, "
            f"caution={self.caution_obstacle_threshold}, "
            f"res={self.costmap_resolution}, "
            f"inflation={self.inflation_radius_meters}"
            f"max_filter={self.max_filter_size}"  
            f"min_filter={self.min_filter_size}"
            
        )

        response.success = True
        response.message = "Thresholds updated"
        return response

    # TF broadcaster for the static map frame. 
    # This allows RViz2 to visualize the costmaps in the correct coordinate frame.
    def _publish_map_frame(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id  = self.map_frame_id
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Static TF published: world → {self.map_frame_id}")

    # PCD loading and processing
    def load_pcd_and_process(self):
        try:
            pcd = o3d.io.read_point_cloud(self.pcd_file_path)
            if not pcd.has_points():
                self.get_logger().error(f"PCD file has no points: {self.pcd_file_path}")
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

    # Elevation map computation: average Z per cell, with a count of points to know where we have data.
    def compute_elevation_map(self, points, min_bound, width_cells, height_cells):
        grid_z_sums   = np.zeros((height_cells, width_cells), dtype=np.float32)
        grid_z_counts = np.zeros((height_cells, width_cells), dtype=np.int32)

        x_idx = ((points[:, 0] - min_bound[0]) / self.costmap_resolution).astype(int)
        y_idx = ((points[:, 1] - min_bound[1]) / self.costmap_resolution).astype(int)

        valid = (x_idx >= 0) & (x_idx < width_cells) & \
                (y_idx >= 0) & (y_idx < height_cells)

        np.add.at(grid_z_sums,   (y_idx[valid], x_idx[valid]), points[valid, 2])
        np.add.at(grid_z_counts, (y_idx[valid], x_idx[valid]), 1)

        elevation_map = np.divide(
            grid_z_sums, grid_z_counts,
            out=np.zeros_like(grid_z_sums),
            where=grid_z_counts != 0,
        )
        # Boolean mask: True where we actually have LiDAR data
        has_data = grid_z_counts > 0
        return elevation_map, has_data

    def align_floor_to_xy_plane(
            self,
            points,
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000):

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

    def publish_pointcloud(self, points):
        """
        Publish Nx3 numpy array as ROS2 PointCloud2
        """

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame_id

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
            points.astype(np.float32)
        )

        self.publisher_pointcloud.publish(cloud_msg)

        self.get_logger().info(
            f"Published point cloud with {len(points)} points"
        )

    #  Service handler
    def generate_costmaps_service(self, request, response):

        # Load PCD and compute Costmap layers
        pcd, points, min_bound, max_bound = self.load_pcd_and_process()
        if pcd is None:
            response.success = False
            response.message = "Failed to load PCD — check pcd_file_path."
            return response
        
        # Publish the point cloud for visualization (optional, but helpful for debugging and RViz2 visualization)
        self.publish_pointcloud(points)

        # Grid size 
        width_cells  = int((max_bound[0] - min_bound[0]) / self.costmap_resolution) + 1
        height_cells = int((max_bound[1] - min_bound[1]) / self.costmap_resolution) + 1
        self.get_logger().info(f"Grid: {width_cells} x {height_cells} cells")

        # Elevation map and data mask — we need the elevation map to compute roughness, and the mask to know where we have data
        elevation_map, has_data = self.compute_elevation_map(points, min_bound, width_cells, height_cells)

        # Compute terrain roughness as the difference between max and min filtered elevation values
        # in this case the filters are used to find local max and min elevations in a neighborhood, and their difference gives a measure of roughness.
        terrain_roughness = (maximum_filter(elevation_map, size=self.get_parameter('max_filter_size').value) 
        - minimum_filter(elevation_map, size=self.get_parameter('min_filter_size').value))

        #Static layer — classify each cell based on height jump thresholds 
        static = np.full((height_cells, width_cells), -1, dtype=np.int16)

        #Flat terrain marked as free with a cost of 10.
        static[has_data & (terrain_roughness <= self.flat_caution_threshold)] = 10

        #Passable terrain marked as caution with a cost of 50.
        static[has_data & (terrain_roughness > self.flat_caution_threshold) & 
               (terrain_roughness <= self.caution_obstacle_threshold)] = 50

        #Unpassable terrain marked as an obstacle with a cost of 100.
        static[has_data & (terrain_roughness > self.caution_obstacle_threshold)] = 100

        # Debug: log the distribution of cell types in the static layer
        self.get_logger().info(
            f"Static layer  — free:{np.sum(static==10)}  "
            f"caution:{np.sum(static==50)}  obstacle:{np.sum(static==100)}  "
            f"unknown:{np.sum(static==-1)}"
        )

        # Inflation layer — dilate obstacles, but only where we have data (don't inflate unknown)
        obstacle_mask = (static == 100).astype(np.uint8) * 255
        # Inflation radius in cells, ensuring at least 1 cell of inflation for small radii.
        inflation_cells = max(1, int(self.inflation_radius_meters / self.costmap_resolution))
        kernel          = np.ones((2 * inflation_cells + 1, 2 * inflation_cells + 1), np.uint8)
        inflated_mask   = cv2.dilate(obstacle_mask, kernel, iterations=1)
        
        inflation = static.copy()   # int16 copy

        # Buffer zone around obstacles marked as caution with a cost of 90, 
        # but only where we have data and it's not already an obstacle.
        buffer_zone = (inflated_mask > 0) & (static < 100) & (static >= 0)
        inflation[buffer_zone] = np.maximum(inflation[buffer_zone], 90)

        # Master layer — element-wise max (int16, so -1 behaves correctly) 
        # np.maximum(-1, 10) = 10  ✓   np.maximum(-1, -1) = -1  ✓
        master = np.maximum(static, inflation)

        # Publish Master, Static, and Inflation costmaps 
        self.publish_costmap(static,    width_cells, height_cells, min_bound, '/static_costmap',    self.publisher_static_costmap)
        self.publish_costmap(inflation, width_cells, height_cells, min_bound, '/inflation_costmap', self.publisher_inflation_costmap)
        self.publish_costmap(master,    width_cells, height_cells, min_bound, '/master_costmap',    self.publisher_master_costmap)
        self.save_costmap_to_npy(master, min_bound)

        response.success = True
        response.message = f"Costmaps generated: {width_cells}x{height_cells} cells"
        self.get_logger().info(response.message)
        return response

    # Publish a costmap layer as an OccupancyGrid message so RViz2 can visualize that directly.
    def publish_costmap(self, data_int16, width, height, min_bound, topic_name, publisher):
        
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = self.map_frame_id
        msg.info.resolution    = float(self.costmap_resolution)
        msg.info.width         = width
        msg.info.height        = height
        msg.info.origin.position.x    = float(min_bound[0])
        msg.info.origin.position.y    = float(min_bound[1])
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        # Clamp to [-1, 100] before casting so no value wraps in int8
        clamped = np.clip(data_int16, -1, 100)
        msg.data = clamped.flatten().astype(np.int8).tolist()

        # Debug: confirm the message has real content
        arr = np.array(msg.data)
        self.get_logger().info(
            f"Publishing {topic_name}  size={len(msg.data)}  "
            f"min={arr.min()}  max={arr.max()}  "
            f"non-unknown cells: {np.sum(arr >= 0)}"
        )

        publisher.publish(msg)

    # Helper to find a save folder for the .npy files. 
    # It looks for the workspace src folder and creates a Cost_Map_Coordinates folder there. 
    # If it can't find the workspace, it saves next to the script.
    def find_save_folder(self):
        # Walk up from the installed file location until we find the workspace src folder
        current = Path(__file__).resolve()
        for parent in current.parents:
            src = parent / "src" / "Cost_Map" / "Cost_Map_Coordinates"
            if src.exists():
                return src
        # Fallback: create it next to the script if workspace not found
        self.get_logger().warn("could not find workspace src — saving next to installed file")
        return Path(__file__).parent / "Cost_Map_Coordinates"

    # Save the costmap to a .npy file with (x, y, cost) for each cell that has data (cost >= 0).
    def save_costmap_to_npy(self, data_int16, min_bound):
        try:
            Save_File = self.find_save_folder()
            self.get_logger().info(f"Saving to: {Save_File.resolve()}")
            Save_File.mkdir(parents=True, exist_ok=True)
            rows, cols = np.where(data_int16 >= 0)
            coords_and_values = np.column_stack([
                min_bound[0] + cols * self.costmap_resolution,
                min_bound[1] + rows * self.costmap_resolution,
                data_int16[rows, cols],
            ])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Save_File / f"costmap_{timestamp}.npy"
            np.save(filename, coords_and_values)
            self.get_logger().info(f"Saved {len(coords_and_values)} cells → {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save costmap: {e}")



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

# If this script is run directly, execute the main function to start the node.
if __name__ == '__main__':
    main()