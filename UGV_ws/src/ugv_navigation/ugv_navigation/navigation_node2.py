#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import numpy as np
import heapq
import time


# ── Paste your 100 TEST_CASES dict here, at module level ─────────────────────
TEST_CASES = {
      1: {"start": (590, 69), "goal": ( 81,689), "obs_pct": 0.21, "straight_dist_m": 40.1},
      2: {"start": ( 17,727), "goal": (686,382), "obs_pct": 0.22, "straight_dist_m": 37.6},
      3: {"start": (682, 15), "goal": ( 75,230), "obs_pct": 0.23, "straight_dist_m": 32.2},
      4: {"start": (625, 17), "goal": (685,572), "obs_pct": 0.32, "straight_dist_m": 27.9},
      5: {"start": (818,555), "goal": (492,296), "obs_pct": 0.28, "straight_dist_m": 20.8},
      6: {"start": ( 63,518), "goal": (542,462), "obs_pct": 0.25, "straight_dist_m": 24.1},
      7: {"start": (390,491), "goal": ( 22,633), "obs_pct": 0.39, "straight_dist_m": 19.7},
      8: {"start": ( 19, 69), "goal": (514,683), "obs_pct": 0.33, "straight_dist_m": 39.4},
      9: {"start": (155,438), "goal": (664, 58), "obs_pct": 0.44, "straight_dist_m": 31.8},
     10: {"start": (600,208), "goal": (650,542), "obs_pct": 0.29, "straight_dist_m": 16.9},
     11: {"start": (174,491), "goal": (413,167), "obs_pct": 0.48, "straight_dist_m": 20.1},
     12: {"start": (541,320), "goal": (234,309), "obs_pct": 0.31, "straight_dist_m": 15.4},
     13: {"start": (  4,499), "goal": (701,237), "obs_pct": 0.16, "straight_dist_m": 37.2},
     14: {"start": (739,358), "goal": (118,340), "obs_pct": 0.28, "straight_dist_m": 31.1},
     15: {"start": (647,326), "goal": (391,247), "obs_pct": 0.20, "straight_dist_m": 13.4},
     16: {"start": (301,485), "goal": (234, 96), "obs_pct": 0.52, "straight_dist_m": 19.7},
     17: {"start": (297,625), "goal": ( 75, 81), "obs_pct": 0.23, "straight_dist_m": 29.4},
     18: {"start": ( 68, 24), "goal": (341,616), "obs_pct": 0.44, "straight_dist_m": 32.6},
     19: {"start": (777,238), "goal": (305,290), "obs_pct": 0.25, "straight_dist_m": 23.7},
     20: {"start": (556,107), "goal": (219,309), "obs_pct": 0.36, "straight_dist_m": 19.6},
     21: {"start": (740,475), "goal": ( 31,189), "obs_pct": 0.18, "straight_dist_m": 38.2},
     22: {"start": (676, 99), "goal": (421,171), "obs_pct": 0.40, "straight_dist_m": 13.2},
     23: {"start": (483,455), "goal": ( 92, 15), "obs_pct": 0.31, "straight_dist_m": 29.4},
     24: {"start": (843,309), "goal": (340,203), "obs_pct": 0.30, "straight_dist_m": 25.7},
     25: {"start": ( 57,475), "goal": (501,463), "obs_pct": 0.28, "straight_dist_m": 22.2},
     26: {"start": (248,539), "goal": (761,206), "obs_pct": 0.41, "straight_dist_m": 30.6},
     27: {"start": (812,681), "goal": (791, 40), "obs_pct": 0.22, "straight_dist_m": 32.1},
     28: {"start": (149,698), "goal": (653,572), "obs_pct": 0.20, "straight_dist_m": 26.0},
     29: {"start": ( 50,542), "goal": ( 33,  9), "obs_pct": 0.05, "straight_dist_m": 26.7},
     30: {"start": (610,729), "goal": (181,225), "obs_pct": 0.20, "straight_dist_m": 33.1},
     31: {"start": (713,156), "goal": (245, 27), "obs_pct": 0.35, "straight_dist_m": 24.3},
     32: {"start": ( 58,266), "goal": (784,541), "obs_pct": 0.26, "straight_dist_m": 38.8},
     33: {"start": (185,455), "goal": (795,556), "obs_pct": 0.31, "straight_dist_m": 30.9},
     34: {"start": ( 74,147), "goal": (341,720), "obs_pct": 0.21, "straight_dist_m": 31.6},
     35: {"start": (234,195), "goal": (416,367), "obs_pct": 0.52, "straight_dist_m": 12.5},
     36: {"start": (586,199), "goal": (765,404), "obs_pct": 0.44, "straight_dist_m": 13.6},
     37: {"start": (331,625), "goal": (316,153), "obs_pct": 0.47, "straight_dist_m": 23.6},
     38: {"start": (165,362), "goal": (619, 20), "obs_pct": 0.28, "straight_dist_m": 28.4},
     39: {"start": (222,238), "goal": (651, 72), "obs_pct": 0.35, "straight_dist_m": 23.0},
     40: {"start": (854,685), "goal": (632, 15), "obs_pct": 0.07, "straight_dist_m": 35.3},
     41: {"start": (598,352), "goal": ( 52, 81), "obs_pct": 0.35, "straight_dist_m": 30.5},
     42: {"start": (561,208), "goal": (585,497), "obs_pct": 0.44, "straight_dist_m": 14.5},
     43: {"start": (128,490), "goal": (481,631), "obs_pct": 0.58, "straight_dist_m": 19.0},
     44: {"start": (675,464), "goal": (197,105), "obs_pct": 0.19, "straight_dist_m": 29.9},
     45: {"start": (121,576), "goal": (423,318), "obs_pct": 0.26, "straight_dist_m": 19.9},
     46: {"start": (845,637), "goal": (590,533), "obs_pct": 0.10, "straight_dist_m": 13.8},
     47: {"start": (637, 26), "goal": (508,489), "obs_pct": 0.39, "straight_dist_m": 24.0},
     48: {"start": (173,709), "goal": (633, 80), "obs_pct": 0.33, "straight_dist_m": 39.0},
     49: {"start": (284,233), "goal": (773, 60), "obs_pct": 0.34, "straight_dist_m": 25.9},
     50: {"start": ( 40,412), "goal": (182,213), "obs_pct": 0.17, "straight_dist_m": 12.2},
     51: {"start": (754,124), "goal": ( 23, 70), "obs_pct": 0.16, "straight_dist_m": 36.6},
     52: {"start": (738,681), "goal": (274,410), "obs_pct": 0.14, "straight_dist_m": 26.9},
     53: {"start": ( 48,150), "goal": (166,514), "obs_pct": 0.41, "straight_dist_m": 19.1},
     54: {"start": (836, 55), "goal": (859,691), "obs_pct": 0.12, "straight_dist_m": 31.8},
     55: {"start": (520,572), "goal": (804,703), "obs_pct": 0.22, "straight_dist_m": 15.6},
     56: {"start": (665,625), "goal": (273,404), "obs_pct": 0.27, "straight_dist_m": 22.5},
     57: {"start": (168, 58), "goal": (605,498), "obs_pct": 0.25, "straight_dist_m": 31.0},
     58: {"start": (593,331), "goal": (420,541), "obs_pct": 0.24, "straight_dist_m": 13.6},
     59: {"start": (105,397), "goal": (219,717), "obs_pct": 0.50, "straight_dist_m": 17.0},
     60: {"start": (689,419), "goal": (515, 18), "obs_pct": 0.11, "straight_dist_m": 21.9},
     61: {"start": (485,627), "goal": (217,223), "obs_pct": 0.29, "straight_dist_m": 24.2},
     62: {"start": (396,292), "goal": (823,456), "obs_pct": 0.30, "straight_dist_m": 22.9},
     63: {"start": (536,248), "goal": (359,626), "obs_pct": 0.37, "straight_dist_m": 20.9},
     64: {"start": (322,700), "goal": (173,528), "obs_pct": 0.24, "straight_dist_m": 11.4},
     65: {"start": (102, 92), "goal": (461,532), "obs_pct": 0.24, "straight_dist_m": 28.4},
     66: {"start": (448,633), "goal": ( 66,492), "obs_pct": 0.37, "straight_dist_m": 20.4},
     67: {"start": (698,559), "goal": ( 33,709), "obs_pct": 0.12, "straight_dist_m": 34.1},
     68: {"start": (790,381), "goal": ( 80,508), "obs_pct": 0.30, "straight_dist_m": 36.1},
     69: {"start": (112,729), "goal": (578,357), "obs_pct": 0.16, "straight_dist_m": 29.8},
     70: {"start": (118,599), "goal": (728,452), "obs_pct": 0.36, "straight_dist_m": 31.4},
     71: {"start": (628,656), "goal": (391,  9), "obs_pct": 0.11, "straight_dist_m": 34.5},
     72: {"start": (548,287), "goal": ( 46,155), "obs_pct": 0.28, "straight_dist_m": 26.0},
     73: {"start": (477,536), "goal": (204,167), "obs_pct": 0.31, "straight_dist_m": 23.0},
     74: {"start": (503,288), "goal": (790,119), "obs_pct": 0.25, "straight_dist_m": 16.7},
     75: {"start": (859,414), "goal": (  8,195), "obs_pct": 0.14, "straight_dist_m": 43.9},
     76: {"start": (628,472), "goal": (668,210), "obs_pct": 0.35, "straight_dist_m": 13.3},
     77: {"start": ( 84,245), "goal": (630, 57), "obs_pct": 0.28, "straight_dist_m": 28.9},
     78: {"start": (812,306), "goal": (484,116), "obs_pct": 0.19, "straight_dist_m": 19.0},
     79: {"start": (694,418), "goal": (222,126), "obs_pct": 0.37, "straight_dist_m": 27.8},
     80: {"start": (301,445), "goal": ( 82, 76), "obs_pct": 0.29, "straight_dist_m": 21.5},
     81: {"start": (249, 11), "goal": (401,501), "obs_pct": 0.45, "straight_dist_m": 25.7},
     82: {"start": (117,159), "goal": (416,531), "obs_pct": 0.34, "straight_dist_m": 23.9},
     83: {"start": (  2,251), "goal": (868,386), "obs_pct": 0.17, "straight_dist_m": 43.8},
     84: {"start": (667,260), "goal": (218, 99), "obs_pct": 0.24, "straight_dist_m": 23.8},
     85: {"start": (454,475), "goal": (703,696), "obs_pct": 0.20, "straight_dist_m": 16.6},
     86: {"start": (136, 62), "goal": (460,192), "obs_pct": 0.18, "straight_dist_m": 17.5},
     87: {"start": (835,517), "goal": ( 78,183), "obs_pct": 0.20, "straight_dist_m": 41.4},
     88: {"start": (254,104), "goal": (772, 32), "obs_pct": 0.10, "straight_dist_m": 26.1},
     89: {"start": (589,513), "goal": (460,182), "obs_pct": 0.29, "straight_dist_m": 17.8},
     90: {"start": (561,123), "goal": (155,599), "obs_pct": 0.26, "straight_dist_m": 31.3},
     91: {"start": (112,728), "goal": (335,487), "obs_pct": 0.33, "straight_dist_m": 16.4},
     92: {"start": (704,286), "goal": (120, 84), "obs_pct": 0.27, "straight_dist_m": 30.9},
     93: {"start": (486,108), "goal": (868,220), "obs_pct": 0.35, "straight_dist_m": 19.9},
     94: {"start": (717,576), "goal": (843,382), "obs_pct": 0.24, "straight_dist_m": 11.6},
     95: {"start": (478,551), "goal": (840,284), "obs_pct": 0.31, "straight_dist_m": 22.5},
     96: {"start": (  0,307), "goal": (551, 76), "obs_pct": 0.19, "straight_dist_m": 29.9},
     97: {"start": (284,167), "goal": (444,442), "obs_pct": 0.57, "straight_dist_m": 15.9},
     98: {"start": (848,622), "goal": (324, 13), "obs_pct": 0.32, "straight_dist_m": 40.2},
     99: {"start": (740,240), "goal": (264,514), "obs_pct": 0.40, "straight_dist_m": 27.5},
    100: {"start": (191,607), "goal": ( 42, 23), "obs_pct": 0.22, "straight_dist_m": 30.1},
}
# ─────────────────────────────────────────────────────────────────────────────


class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')
        self.get_logger().info('Navigation node started.')

        self.start_navigation_srv = self.create_service(
            Trigger, '/start_navigation', self.start_navigation_callback
        )

        # ── Add a second service to trigger the test suite ────────────────
        self.run_tests_srv = self.create_service(
            Trigger, '/run_astar_tests', self.run_tests_callback
        )

        raw = np.load('/ros2_ws/src/Cost_Map/Cost_Map_Coordinates/costmap_20260524_170731.npy')

        self.resolution = 0.05
        self.min_x      = raw[:, 0].min()
        self.min_y      = raw[:, 1].min()
        self.min_bound  = (self.min_x, self.min_y)

        cols = int(round((raw[:, 0].max() - self.min_x) / self.resolution)) + 1
        rows = int(round((raw[:, 1].max() - self.min_y) / self.resolution)) + 1

        self.grid = np.full((rows, cols), -1, dtype=np.int16)
        for x_world, y_world, cost in raw:
            col = int(round((x_world - self.min_x) / self.resolution))
            row = int(round((y_world - self.min_y) / self.resolution))
            self.grid[row, col] = int(cost)

        self.map_rows, self.map_cols = self.grid.shape
        self.get_logger().info(f'Grid built: {self.map_rows} rows x {self.map_cols} cols')

        self.open_list = []
        self.closed    = set()
        self.g_score   = {}
        self.came_from = {}

        self.start = (0, 0)
        self.goal  = (10, 10)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.visited_pub = self.create_publisher(OccupancyGrid, '/astar_visited', qos)
        self.path_pub    = self.create_publisher(OccupancyGrid, '/astar_path',    qos)

        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_map_tf()

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

    def run_tests_callback(self, request, response):
        """
        Runs all 100 test cases sequentially, publishes each path to RViz2,
        then logs a full summary.

        Trigger with:
            ros2 service call /run_astar_tests std_srvs/srv/Trigger '{}'
        """
        passed     = 0
        failed     = 0
        total_time = 0.0
        failures   = []

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Starting A* test suite — {len(TEST_CASES)} cases')
        self.get_logger().info('=' * 60)

        for n, case in TEST_CASES.items():
            self.start = case['start']
            self.goal  = case['goal']

            # Reset all A* state before each run
            self._reset_astar()

            t0   = time.time()
            path = self.a_star()
            dt   = time.time() - t0
            total_time += dt

            if path:
                passed += 1
                self.get_logger().info(
                    f'  [{n:>3}/100]  ✓  {self.start} → {self.goal}'
                    f'  |  {len(path)} cells  |  {dt:.2f}s'
                )
            else:
                failed += 1
                failures.append(n)
                self.get_logger().warn(
                    f'  [{n:>3}/100]  ✗  {self.start} → {self.goal}'
                    f'  |  no path  |  {dt:.2f}s'
                )

        # ── Summary ───────────────────────────────────────────────────────
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Passed   : {passed} / 100')
        self.get_logger().info(f'  Failed   : {failed} / 100')
        self.get_logger().info(f'  Total    : {total_time:.1f}s  (avg {total_time/100:.2f}s/case)')
        if failures:
            self.get_logger().warn(f'  Failed cases: {failures}')
        self.get_logger().info('=' * 60)

        response.success = True
        response.message = f'{passed}/100 passed in {total_time:.1f}s'
        return response

    # ── Single navigation service ─────────────────────────────────────────

    def start_navigation_callback(self, request, response):
        self.get_logger().info('Navigation requested.')
        self._reset_astar()
        path = self.a_star()
        if path:
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