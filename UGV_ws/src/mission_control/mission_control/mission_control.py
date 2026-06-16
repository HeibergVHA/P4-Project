import threading
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Cursor
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from control_interfaces.srv import SetMission, SetGoal

class MissionControl(Node):
    """
    Actions in steps
    
    1. Receives costmap path from template matching node aswell as the 
    starting position of the robot via /set_mission service.
    2. Displays the costmap and allows the user to hover over cell coordinates
    3. Accepts terminal input for the target coordinate.
    4. Sends start + goal goal to the A-star node via /set_goal service
    """

    def __init__(self):
        super().__init__('mission_control')

        self._costmap = None # Nx3 array of x,y,cost
        self._start = None # (x,y)
        self._goal = None # also (x,y)
        self._mission_ready = False

        # Service to receive map path and start position
        self.create_service(
            SetMission,
            '/set_mission',
            self._cb_set_mission
        )

        # Service client

        self._goal_client = self.create_client(SetGoal, '/set_goal')

        # Terminal input

        self._input_thread = threading.Thread(
            target = self._input_loop, daemon = True
        )
        self._input_thread.start()

        self.get_logger().info('Mission Control Node ready...')

    # ---- Service callback ----

    def _cb_set_mission(self, request, response):
        self.get_logger().info(
            f'Received mission: costmap_path={request.costmap_path} '
            f'start = ({request.start_x:.2f}, {request.start_y:.2f})'   
        )

        try:
            data = np.load(request.costmap_path)
            self._costmap_path = request.costmap_path
            if data.shape[1] != 3:
                raise ValueError(f'Expected costmap nx3 array')
            self._costmap = data
            self._start = (request.start_x, request.start_y)
            self._mission_ready = True
        except Exception as e:
            self.get_logger().error(f'Failed to load costmap: {e}')
            response.success = False
            response.message = str(e)
            return response

        response.success = True
        response.message = 'Mission set successfully.'
        return response

    # ---- Display costmap ----


    def _display_costmap(self):
        data = self._costmap
        xs, ys, costs = data[:, 0], data[:, 1], data[:, 2]

        # Reconstruct the 2D grid for imshow
        x_unique = np.sort(np.unique(xs))
        y_unique = np.sort(np.unique(ys))
        res = round(x_unique[1] - x_unique[0], 4)

        width  = len(x_unique)
        height = len(y_unique)

        grid = np.full((height, width), -1, dtype=np.float32)

        x_idx = np.round((xs - x_unique[0]) / res).astype(int)
        y_idx = np.round((ys - y_unique[0]) / res).astype(int)
        grid[y_idx, x_idx] = costs

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'costmap', [(0, 'grey'), (0.1, 'green'), (0.5, 'yellow'), (1.0, 'red')]
        )
        cmap.set_under('grey')  # unknown cells

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(
            grid,
            origin='lower',
            extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
            cmap=cmap,
            vmin=0, vmax=100,
            interpolation='nearest'
        )
        plt.colorbar(im, ax=ax, label='Cost (0=free, 100=obstacle)')

        if self._start:
            ax.plot(*self._start, 'b*', markersize=15, label='Start (WALL-R)')
            ax.legend()

        ax.set_title('Costmap - Hover to view coordinates, enter target in terminal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # Hover annotation
        annot = ax.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8),
            fontsize=8
        )
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes != ax:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            # Snap to nearest grid cell directly from coordinates
            xi = int(round((x - x_unique[0]) / res))
            yi = int(round((y - y_unique[0]) / res))
            if 0 <= xi < width and 0 <= yi < height:
                cost = grid[yi, xi]
                annot.xy = (x, y)
                annot.set_text(
                    f'x={x_unique[xi]:.2f}m  y={y_unique[yi]:.2f}m\n'
                    f'cost={int(cost) if cost >= 0 else "unknown"}'
                )
                annot.set_visible(True)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        plt.tight_layout()
        plt.show()

    # def _display_costmap(self):
    #     data = self._costmap
    #     xs, ys, costs = data[:,0], data[:,1], data[:,2]

    #     # Map costs to colours: free=green, caution=yellow, obstacle = red

    #     cmap = mcolors.LinearSegmentedColormap.from_list(
    #         'costmap', [(0, 'green'), (0.5, 'yellow'), (1, 'red')]
    #     )
    #     norm = mcolors.Normalize(vmin=0, vmax=100)

    #     # Figsize is in inches, so 10x10 is a large square
    #     fig, ax = plt.subplots(figsize=(10,10))

    #     sc = ax.scatter(
    #         xs, ys, c=costs, cmap=cmap, norm=norm,
    #         s=1, marker='s'
    #     )
    #     plt.colorbar(sc, label='Cost (0=free, 100=obstacle)')

    #     #Plot start position

    #     if self._start:
    #         ax.plot(
    #             #* unpacks the tuple into x and y b* does blue star marker
    #             *self._start, 'b*', markersize = 15, label='Start (WALL-R)'
    #         )
    #         ax.legend()


    #     ax.set_title(
    #         'Costmap - Hover to view coordinates, enter target position in terminal'
    #     )
    #     ax.set_xlabel('X(m)')
    #     ax.set_ylabel('Y(m)')
    #     ax.set_aspect('equal')

    #     # Hover annotation for showing coordinates

    #     annot = ax.annotate(
    #         '', xy=(0,0), xytext=(10,10),
    #         textcoords='offset points',
    #         bbox=dict(boxstyle='round', fc='white', alpha = 0.8),
    #         fontsize=8
    #     )
    #     annot.set_visible(False)

    #     def on_hover(event):
    #         if event.inaxes != ax:
    #             return
    #         x, y = event.xdata, event.ydata
    #         if x is None or y is None:
    #             return
            
    #         # Find neares cell
    #         dists = np.hypot(xs-x, ys-y)
    #         # argmin returns the index of the minimum value in the array, which corresponds to the nearest cell
    #         idx = np.argmin(dists)

    #         annot.xy = (xs[idx], ys[idx])
    #         annot.set_text(
    #             f'x={xs[idx]:.2f}m y={ys[idx]:.2f}m\ncost={int(costs[idx])}'
    #         )

    #         annot.set_visible(True)
    #         fig.canvas.draw_idle()

    #     fig.canvas.mpl_connect('motion_notify_event', on_hover)
    #     plt.tight_layout()
    #     plt.show()

    # ---- Terminal input ----

    def _input_loop(self):
        while rclpy.ok():
            if not self._mission_ready:
                continue

            print('\nCostmap loaded. Enter target coordinates')

            try:
                goal_x = float(input('  Goal X (m):  ').strip())
                goal_y = float(input('  Goal Y (m):  ').strip())
            except ValueError:
                print('Invalid input')
                continue

            self._goal = (goal_x, goal_y)

            print(f' Start : ({self._start[0]:.2f}, {self._start[1]:.2f})\n'
                  f' Goal  : ({self._goal[0]:.2f}, {self._goal[1]:.2f})\n'
                  f' Sending goal to A-star node...'
            )
            self._send_goal()
            self._mission_ready = False # Prevent multiple inputs until next mission is set
            
    # A-star service call

    def _send_goal(self):
        if not self._goal_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('A-star service not available')
            return
        request = SetGoal.Request()
        request.costmap_path = self._costmap_path
        request.start_x = float(self._start[0])
        request.start_y = float(self._start[1])
        request.goal_x = float(self._goal[0])
        request.goal_y = float(self._goal[1])

        future = self._goal_client.call_async(request)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info(f'A* accepted goal: {result.message}')
        else:
            self.get_logger().error(f'A* rejected goal: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    node = MissionControl()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if node._mission_ready and node._costmap is not None:
                node._display_costmap()  # runs on main thread, blocks until window closed
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

