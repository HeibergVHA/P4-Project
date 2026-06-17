import threading
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Cursor
import json
from std_msgs.msg import String
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from control_interfaces.srv import SetMission, SetGoal
import time

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
        self._waypoints = []

        # Service to receive map path and start position
        self.create_service(
            SetMission,
            '/set_mission',
            self._cb_set_mission
        )

        # Service client

        self._goal_client = self.create_client(SetGoal, '/set_goal')
        
        self._waypoint_pub = self.create_publisher(String, 'ugv/radio_out/target_waypoint', 10)
        # Terminal input
        self.create_service(Trigger, '/send_waypoints', self._cb_send_waypoints)
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

        x_unique = np.sort(np.unique(xs))
        y_unique = np.sort(np.unique(ys))
        res = round(x_unique[1] - x_unique[0], 4)

        width  = len(x_unique)
        height = len(y_unique)

        grid = np.full((height, width), -1, dtype=np.float32)

        x_idx = np.round((xs - x_unique[0]) / res).astype(int)
        y_idx = np.round((ys - y_unique[0]) / res).astype(int)

        x_idx = np.clip(x_idx, 0, width - 1)
        y_idx = np.clip(y_idx, 0, height - 1)

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

        # ---- Terminal input ----

    def _input_loop(self):
        while rclpy.ok():
            if not self._mission_ready:
                continue

            print(f'\nDetected start position: ({self._start[0]:.2f}, {self._start[1]:.2f})')
            
            while True:
                choice = input('  Accept start position? (1=Accept, 0=Override): ').strip()
                if choice == '1':
                    break
                elif choice == '0':
                    try:
                        self._start = (
                            float(input('  Start X (m): ').strip()),
                            float(input('  Start Y (m): ').strip())
                        )
                        break
                    except ValueError:
                        print('  Invalid coordinates, try again.')
                else:
                    print('  Please type 1 to accept or 0 to override.')

            try:
                goal_x = float(input('  Goal X (m): ').strip())
                goal_y = float(input('  Goal Y (m): ').strip())
            except (ValueError, EOFError):
                print('Invalid input')
                continue

            self._goal = (goal_x, goal_y)
            print(
                f'  Start : ({self._start[0]:.2f}, {self._start[1]:.2f})\n'
                f'  Goal  : ({goal_x:.2f}, {goal_y:.2f})\n'
                f'  Sending to A*...'
            )
            self._send_goal()
            self._mission_ready = False
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

    def _cb_send_waypoints(self, request, response):
        if not self._waypoints:
            response.success = False
            response.message = 'No waypoints defined.'
            return response

        threading.Thread(target=self._send_waypoints_loop, daemon=True).start()
        response.success = True
        response.message = f'Sending {len(self._waypoints)} waypoints'
        return response

    def _send_waypoints_loop(self):
        for i, (x, y, z, yaw) in enumerate(self._waypoints):
            packet = {
                'type': 'waypoint',
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'yaw': float(yaw),
            }
            msg = String()
            msg.data = json.dumps(packet) + '\n'
            self._waypoint_pub.publish(msg)
            self.get_logger().info(f'Sent waypoint {i+1}/{len(self._waypoints)}: {packet}')
            time.sleep(0.5)  # small delay between waypoints
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

