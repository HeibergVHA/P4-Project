import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import numpy as np
import heapq


class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')
        self.get_logger().info('Navigation node has been started.')

        self.start_navigation_srv = self.create_service(
            Trigger,
            '/start_navigation',
            self.start_navigation_callback
        )
        # The .npy file is shape (N, 3): each row is [x_world, y_world, cost]
        # We need to convert this into a 2D grid indexed by cell (row, col).
        raw = np.load("path/to/costmap_<timestamp>.npy")  # shape (N, 3)

        # Figure out grid dimensions from the world coordinates
        # raw[:, 0] = x values, raw[:, 1] = y values, raw[:, 2] = costs
        resolution = 0.05  # must match what lidar_costmap_node used
        min_x = raw[:, 0].min()
        min_y = raw[:, 1].min()

        cols = int(round((raw[:, 0].max() - min_x) / resolution)) + 1
        rows = int(round((raw[:, 1].max() - min_y) / resolution)) + 1

        # Build the 2D grid — unknown cells default to -1
        self.grid = np.full((rows, cols), -1, dtype=np.int16)
        for x_world, y_world, cost in raw:
            col = int(round((x_world - min_x) / resolution))
            row = int(round((y_world - min_y) / resolution))
            self.grid[row, col] = int(cost)

        # ── A* state ──────────────────────────────────────────────────────
        self.open_list  = []           # min-heap: (f_cost, g_cost, row, col)
        self.closed     = set()        # set of (row, col) already expanded
        self.g_score    = {}           # (row, col) → best cost found so far
        self.came_from  = {}           # (row, col) → (row, col) | None

        # Start and goal in grid cells (not world coords yet)
        self.start  = (0, 0)
        self.goal   = (10, 10)

    def a_star(self):
        start_cost = float(self.grid[self.start[0], self.start[1]])

        # Seed the three structures
        self.g_score[self.start]   = start_cost
        self.came_from[self.start] = None
        heapq.heappush(self.open_list, (start_cost, start_cost, *self.start))
        #                               ^f=g+h=g    ^g           ^row, col

        while self.open_list:
            f, g, row, col = heapq.heappop(self.open_list)
            current = (row, col)

            # Skip if we already found a cheaper route to this cell
            if g > self.g_score.get(current, float('inf')):
                continue

            if current == self.goal:
                print("Path found!")
                path = self.reconstruct_path()
                return path

            self.closed.add(current)
            self.generate_neighbours(current, g)

        print("No path found.")
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

                # Skip if already expanded
                if nb in self.closed:
                    continue

                # Skip if out of bounds
                rows, cols = self.grid.shape
                if not (0 <= nb_row < rows and 0 <= nb_col < cols):
                    continue

                nb_cost = int(self.grid[nb_row, nb_col])

                # Skip unknown (-1) and obstacles (100)
                if nb_cost < 0 or nb_cost >= 100:
                    continue

                # Diagonal moves cost more (√2 × cell cost)
                dist_mult   = np.hypot(dr, dc)          # 1.0 cardinal, 1.414 diagonal
                tentative_g = current_g + dist_mult * nb_cost

                # Only update if this route is cheaper than any we've seen
                if tentative_g < self.g_score.get(nb, float('inf')):
                    self.g_score[nb]   = tentative_g
                    self.came_from[nb] = current

                    h = np.hypot(nb_row - self.goal[0], nb_col - self.goal[1])
                    heapq.heappush(self.open_list, (tentative_g + h, tentative_g, nb_row, nb_col))

    def reconstruct_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = self.came_from[node]
        path.reverse()
        return path

    def start_navigation_callback(self, request, response):
        self.get_logger().info('Received request to start navigation.')
        response.success = True
        response.message = 'Navigation started successfully.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()