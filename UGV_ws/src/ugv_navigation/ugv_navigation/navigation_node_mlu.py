import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import numpy as np
import heapq
import glob, os

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy


class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')
        self.get_logger().info('Navigation node has been started.')

        self.start_navigation_srv = self.create_service(
            Trigger,
            '/start_navigation',
            self.start_navigation_callback
        )

        # ── Load costmap ──────────────────────────────────────────────────
        raw = self._load_latest_costmap("src/Cost_Map/Cost_Map_Coordinates")
        raw_pos = self._load_latest_templateMatch("src/Cost_Map/TemplateMatching")

        self._res   = 0.05
        self._min_x = raw[:, 0].min()
        self._min_y = raw[:, 1].min()

        cols = int(round((raw[:, 0].max() - self._min_x) / self._res)) + 1
        rows = int(round((raw[:, 1].max() - self._min_y) / self._res)) + 1

        # Row 0 = lowest y, row N-1 = highest y  (no flip — matches RViz2 frame)
        self.grid = np.full((rows, cols), -1, dtype=np.int16)
        for x_world, y_world, cost in raw:
            col = int(round((x_world - self._min_x) / self._res))
            row = int(round((y_world - self._min_y) / self._res))
            self.grid[row, col] = int(cost)

        self.get_logger().info(
            f"Grid shape: {self.grid.shape}  "
            f"x=[{self._min_x:.2f}, {raw[:,0].max():.2f}]  "
            f"y=[{self._min_y:.2f}, {raw[:,1].max():.2f}]"
        )

        # ── Start / goal (world coords → grid cells) ──────────────────────
        # Change these to your desired world-frame waypoints.
        #start_world = (0.0,  0.0)
        #raw_pos = np.array(raw_pos)
        start_world = (raw_pos[3,0], raw_pos[3,1])
        goal_world  = (10.0, 3.0)

        self.start = self._world_to_grid(*start_world)
        self.goal  = self._world_to_grid(*goal_world)

        self.get_logger().info(f"Start: world={start_world}  grid={self.start}  cost={self.grid[self.start]}")
        self.get_logger().info(f"Goal:  world={goal_world}   grid={self.goal}   cost={self.grid[self.goal]}")

        # Warn early if start/goal are not navigable
        for label, cell in [("Start", self.start), ("Goal", self.goal)]:
            r, c = cell
            rows_, cols_ = self.grid.shape
            if not (0 <= r < rows_ and 0 <= c < cols_):
                self.get_logger().error(f"{label} cell {cell} is OUT OF BOUNDS!")
            elif self.grid[r, c] < 0:
                self.get_logger().error(f"{label} cell {cell} is UNKNOWN (cost=-1)!")
            elif self.grid[r, c] >= 100:
                self.get_logger().error(f"{label} cell {cell} is an OBSTACLE (cost={self.grid[r,c]})!")

        # ── Latched publisher for RViz2 ───────────────────────────────────
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/a_star_path', latched_qos)

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) → (row, col).  Row increases with y — no flip."""
        col = int(round((x - self._min_x) / self._res))
        row = int(round((y - self._min_y) / self._res))
        return (row, col)

    def _grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) → world (x, y) at cell centre.  Inverse of _world_to_grid."""
        x = self._min_x + col * self._res
        y = self._min_y + row * self._res
        return (x, y)

    # ── A* ────────────────────────────────────────────────────────────────

    def a_star(self):
        open_list = []
        closed    = set()
        g_score   = {}
        came_from = {}

        g_score[self.start]   = 0.0
        came_from[self.start] = None
        h0 = float(np.hypot(self.start[0] - self.goal[0],
                             self.start[1] - self.goal[1]))
        heapq.heappush(open_list, (h0, 0.0, *self.start))

        while open_list:
            f, g, row, col = heapq.heappop(open_list)
            current = (row, col)

            if g > g_score.get(current, float('inf')):
                continue                        # stale entry

            if current == self.goal:
                self.get_logger().info("A*: path found!")
                return self._reconstruct(came_from)

            closed.add(current)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nb = (row + dr, col + dc)
                    if nb in closed:
                        continue

                    nr, nc = nb
                    rows_, cols_ = self.grid.shape
                    if not (0 <= nr < rows_ and 0 <= nc < cols_):
                        continue

                    nb_cost = int(self.grid[nr, nc])
                    #if nb_cost < 0 or nb_cost >= 100:   # unknown or lethal
                    #    continue
                    ignore_cost = g < (0.5 / self._res)
                    if not ignore_cost and (nb_cost < 0 or nb_cost >= 100):
                        continue

                    # Use cost+1 so zero-cost free cells still advance g
                    ignore_cost = g < (0.5 / self._res)  # g is in step-units, threshold converted to match
                    step = np.hypot(dr, dc) * (1 if ignore_cost else nb_cost + 1)
                    new_g    = g + step

                    if new_g < g_score.get(nb, float('inf')):
                        g_score[nb]   = new_g
                        came_from[nb] = current
                        h = float(np.hypot(nr - self.goal[0], nc - self.goal[1]))
                        heapq.heappush(open_list, (new_g + h, new_g, nr, nc))

        self.get_logger().warn("A*: no path found.")
        return None

    def _reconstruct(self, came_from: dict) -> list:
        path, node = [], self.goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    # ── RViz2 publisher ───────────────────────────────────────────────────

    def _publish_path(self, path: list):
        msg        = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for row, col in path:
            pose              = PoseStamped()
            pose.header       = msg.header
            x, y              = self._grid_to_world(row, col)
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0          # flat on the ground plane
            pose.pose.orientation.w = 1.0       # identity rotation
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.get_logger().info(f"Published path with {len(msg.poses)} poses on /a_star_path")

    # ── Service callback ──────────────────────────────────────────────────

    def start_navigation_callback(self, request, response):
        self.get_logger().info('Received /start_navigation request.')
        path = self.a_star()
        if path:
            self._publish_path(path)
            response.success = True
            response.message = f'Path found with {len(path)} steps.'
        else:
            response.success = False
            response.message = 'No path could be found.'
        return response

    def _load_latest_costmap(self, folder: str) -> np.ndarray:
        files = glob.glob(os.path.join(folder, "costmap_*.npy"))
        if not files:
            raise FileNotFoundError(f"No costmap_*.npy files found in '{folder}'")
        latest = max(files, key=os.path.getmtime)
        self.get_logger().info(f"Loading costmap: {latest}")
        return np.load(latest)
    
    def _load_latest_templateMatch(self, folder: str) -> np.ndarray:
        files = glob.glob(os.path.join(folder, "T_rover*.npy"))
        if not files:
            raise FileNotFoundError(f"No T_rover*.npy files found in '{folder}'")
        latest = max(files, key=os.path.getmtime)
        self.get_logger().info(f"Loading template match: {latest}")
        return np.load(latest)
    
    def _cells_from_start(self, row: int, col: int) -> float:
        """Euclidean distance in metres from the start cell."""
        dr = row - self.start[0]
        dc = col - self.start[1]
        return np.hypot(dr, dc) * self._res


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()