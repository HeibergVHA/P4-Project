#!/usr/bin/env python3
from control_interfaces.srv import SetGoal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq
import time
import json

# ─────────────────────────────────────────────────────────────────────────────

class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')
        self.get_logger().info('Navigation node started.')

        # ── Add a second service to trigger the test suite ────────────────
        # self.run_tests_srv = self.create_service(
        #     Trigger, '/run_astar_tests', self.run_tests_callback
        # )

        self.create_service(SetGoal, '/set_goal', self._cb_set_goal)
        self.open_list = []
        self.closed    = set()
        self.g_score   = {}
        self.came_from = {}
        self.resolution = 0.05

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.visited_pub = self.create_publisher(OccupancyGrid, '/astar_visited', qos)
        self.path_pub    = self.create_publisher(OccupancyGrid, '/astar_path',    qos)

        self.path_pub_ugv = self.create_publisher(String, '/ugv/astar_points', 10)
        self.ugv_pos_pub = self.create_publisher(PoseStamped, '/ugv/position', 10)
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_map_tf()


    # Load costmap

    def _load_costmap(self, path: str):
        raw = np.load(path)
        self.min_x = raw[:, 0].min()
        self.min_y = raw[:, 1].min()
        self.min_bound = (self.min_x, self.min_y)

        cols = int(round((raw[:, 0].max() - self.min_x) / self.resolution)) + 1
        rows = int(round((raw[:, 1].max() - self.min_y) / self.resolution)) + 1

        self.grid = np.full((rows, cols), -1, dtype=np.int16)
        for x_world, y_world, cost in raw:
            col = int(round((x_world - self.min_x) / self.resolution))
            row = int(round((y_world - self.min_y) / self.resolution))
            self.grid[row, col] = int(cost)

        self.map_rows, self.map_cols = self.grid.shape
        self.get_logger().info(f'Grid built: {self.map_rows} rows x {self.map_cols} cols')
    # ── TF ────────────────────────────────────────────────────────────────

    def _publish_map_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id  = 'map'
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)

    # ── A* ────────────────────────────────────────────────────────────────

    def _reset_astar(self):
        """Clear all A* state so the algorithm can run fresh."""
        self.open_list.clear()
        self.closed.clear()
        self.g_score.clear()
        self.came_from.clear()

    def a_star(self):
        if not self._cell_passable(*self.start):
            self.get_logger().error(f'Start {self.start} is not passable.')
            return None
        if not self._cell_passable(*self.goal):
            self.get_logger().error(f'Goal {self.goal} is not passable.')
            return None

        start_cost = float(self.grid[self.start[0], self.start[1]])
        self.g_score[self.start]   = start_cost
        self.came_from[self.start] = None
        heapq.heappush(self.open_list, (start_cost, start_cost, *self.start))

        iteration = 0

        while self.open_list:
            f, g, row, col = heapq.heappop(self.open_list)
            current = (row, col)

            if g > self.g_score.get(current, float('inf')):
                continue

            if current == self.goal:
                path = self.reconstruct_path()
                self._publish_visited()
                self._publish_path(path)
                return path

            self.closed.add(current)
            self.generate_neighbours(current, g)

            iteration += 1
            if iteration % 500 == 0:
                self._publish_visited()

        return None

    def generate_neighbours(self, current, current_g):
        row, col = current
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nb_row = row + dr
                nb_col = col + dc
                nb     = (nb_row, nb_col)
                if nb in self.closed:
                    continue
                if not (0 <= nb_row < self.map_rows and 0 <= nb_col < self.map_cols):
                    continue
                if not self._cell_passable(nb_row, nb_col):
                    continue
                nb_cost     = float(self.grid[nb_row, nb_col])
                dist_mult   = np.hypot(dr, dc)
                tentative_g = current_g + dist_mult * nb_cost
                if tentative_g < self.g_score.get(nb, float('inf')):
                    self.g_score[nb]   = tentative_g
                    self.came_from[nb] = current
                    h = np.hypot(nb_row - self.goal[0], nb_col - self.goal[1])
                    heapq.heappush(self.open_list, (tentative_g + h, tentative_g, nb_row, nb_col))

    def reconstruct_path(self):
        path, node = [], self.goal
        while node is not None:
            path.append(node)
            node = self.came_from[node]
        path.reverse()
        return path

    def _cell_passable(self, row, col):
        cost = int(self.grid[row, col])
        return cost >= 0 and cost < 100

    # ── RViz2 ─────────────────────────────────────────────────────────────

    def _publish_visited(self):
        overlay = np.full((self.map_rows, self.map_cols), -1, dtype=np.int16)
        for (r, c) in self.closed:
            overlay[r, c] = 50
        self._publish_grid(overlay, '/astar_visited', self.visited_pub)

    def _publish_path(self, path):
        overlay = np.full((self.map_rows, self.map_cols), -1, dtype=np.int16)
        for (r, c) in path:
            overlay[r, c] = 100
        self._publish_grid(overlay, '/astar_path', self.path_pub)

    def _publish_grid(self, overlay, topic_name, publisher):
        msg = OccupancyGrid()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.header.frame_id           = 'map'
        msg.info.resolution           = float(self.resolution)
        msg.info.width                = self.map_cols
        msg.info.height               = self.map_rows
        msg.info.origin.position.x    = float(self.min_bound[0])
        msg.info.origin.position.y    = float(self.min_bound[1])
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0
        clamped  = np.clip(overlay, -1, 100)
        msg.data = clamped.flatten().astype(np.int8).tolist()
        publisher.publish(msg)

    # ── Test runner ───────────────────────────────────────────────────────

    # def run_tests_callback(self, request, response):
    #     """
    #     Runs all 100 test cases sequentially, publishes each path to RViz2,
    #     then logs a full summary.

    #     Trigger with:
    #         ros2 service call /run_astar_tests std_srvs/srv/Trigger '{}'
    #     """
    #     passed     = 0
    #     failed     = 0
    #     total_time = 0.0
    #     failures   = []

    #     self.get_logger().info('=' * 60)
    #     self.get_logger().info(f'Starting A* test suite — {len(TEST_CASES)} cases')
    #     self.get_logger().info('=' * 60)

    #     for n, case in TEST_CASES.items():
    #         self.start = case['start']
    #         self.goal  = case['goal']

    #         # Reset all A* state before each run
    #         self._reset_astar()

    #         t0   = time.time()
    #         path = self.a_star()
    #         dt   = time.time() - t0
    #         total_time += dt

    #         if path:
    #             passed += 1
    #             self.get_logger().info(
    #                 f'  [{n:>3}/100]  ✓  {self.start} → {self.goal}'
    #                 f'  |  {len(path)} cells  |  {dt:.2f}s'
    #             )
    #         else:
    #             failed += 1
    #             failures.append(n)
    #             self.get_logger().warn(
    #                 f'  [{n:>3}/100]  ✗  {self.start} → {self.goal}'
    #                 f'  |  no path  |  {dt:.2f}s'
    #             )

    #     # ── Summary ───────────────────────────────────────────────────────
    #     self.get_logger().info('=' * 60)
    #     self.get_logger().info(f'  Passed   : {passed} / 100')
    #     self.get_logger().info(f'  Failed   : {failed} / 100')
    #     self.get_logger().info(f'  Total    : {total_time:.1f}s  (avg {total_time/100:.2f}s/case)')
    #     if failures:
    #         self.get_logger().warn(f'  Failed cases: {failures}')
    #     self.get_logger().info('=' * 60)

    #     response.success = True
    #     response.message = f'{passed}/100 passed in {total_time:.1f}s'
    #     return response

    # ── Single navigation service ─────────────────────────────────────────

    def _clear_start_radius(self, radius_meters: float = 0.5):
        """
        Clear obstacle cells within radius_meters of the start position.
        This prevents the rover's own footprint in the costmap from
        blocking the path from the very first cell.
        """
        radius_cells = int(np.ceil(radius_meters / self.resolution))
        row, col = self.start

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if np.hypot(dr, dc) <= radius_cells:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.map_rows and 0 <= c < self.map_cols:
                        if self.grid[r, c] == 100:
                            self.grid[r, c] = 10  # set to free

    def start_navigation_callback(self, request, response):
        self.get_logger().info('Navigation requested.')
        self._reset_astar()
        path = self.a_star()
        self.get_logger().info(str(path))
        if path:
            response.success = True
            response.message = f'Path found: {len(path)} waypoints.'
        else:
            response.success = False
            response.message = 'No path found.'
        return response

    def _cb_set_goal(self, request, response):
        try:
            self._load_costmap(request.costmap_path)
        except Exception as e:
            response.success = False
            response.message = f'Failed to load costmap: {e}'
            return response
        # Convert world coordinates to grid
        start_col = int(round((request.start_x - self.min_x) / self.resolution))
        start_row = int(round((request.start_y - self.min_y) / self.resolution))
        goal_col  = int(round((request.goal_x  - self.min_x) / self.resolution))
        goal_row  = int(round((request.goal_y  - self.min_y) / self.resolution))

        self.start = (start_row, start_col)
        self.goal  = (goal_row,  goal_col)

        self.get_logger().info(
            f'Goal set: start=({request.start_x:.2f}, {request.start_y:.2f}) '
            f' -> cell{self.start}'
            f' goal=({request.goal_x:.2f}, {request.goal_y:.2f}) '
            f' -> cell{self.goal}'
        )

        self._clear_start_radius(radius_meters=0.5)
        self._reset_astar()
        path = self.a_star()

        if path:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = path[0][1] * self.resolution
            pose_msg.pose.position.y = path[0][0] * self.resolution
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0

            self.ugv_pos_pub.publish(pose_msg)

            # Publish path in grid relative coordinates
            world_points = [
                [col*self.resolution, row*self.resolution]
                for row, col in path
            ]

            msg = String()
            msg.data = json.dumps(world_points)
            self.path_pub_ugv.publish(msg)

            response.success = True
            response.message = f'Path found: {len(path)} waypoints.'
        else:
            response.success = False
            response.message = 'No path found.'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()