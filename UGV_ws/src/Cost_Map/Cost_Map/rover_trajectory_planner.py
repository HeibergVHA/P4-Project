#!/usr/bin/env python3
"""
Trajectory Planner Node for ROS2  –  steering-constrained A*
-------------------------------------------------------------
Subscribes to:
  /master_costmap   (nav_msgs/OccupancyGrid)
  /rover_pose_2D    (geometry_msgs/Pose2D)   – start (x, y, theta)

Publishes:
  /waypoints        (nav_msgs/Path)
  /goal_pose        (geometry_msgs/PoseStamped)
  /goal_marker      (visualization_msgs/Marker)       – green arrow
  /waypoint_markers (visualization_msgs/MarkerArray)  – blue spheres/arrows

Planning algorithm
------------------
A* whose state is (col, row, heading_index).

The rover's heading is discretised into N_HEADINGS equal bins.  A move from
one cell to a neighbour is only allowed when the required heading change
does not exceed `max_steering_angle_deg` (a ROS parameter).  Larger-but-
within-limit turns add a configurable `steering_cost_weight` penalty so the
planner prefers gentle curves over sharp ones.

Parameters
----------
frame_id               (str,   default 'map')
cost_weight            (float, default 2.0)   – costmap value → travel cost
max_cost_threshold     (int,   default 90)    – cells >= this are obstacles
waypoint_spacing       (int,   default 5)     – keep every N-th A* cell
max_steering_angle_deg (float, default 30.0)  – maximum turn per grid step [deg]
steering_cost_weight   (float, default 1.5)   – extra cost per radian turned
n_headings             (int,   default 16)    – heading discretisation bins
"""

import math
import heapq

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Pose2D, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from std_srvs.srv import Trigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


def angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a - b, result in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def heuristic(col_a: int, row_a: int, col_b: int, row_b: int) -> float:
    return math.hypot(col_a - col_b, row_a - row_b)


# ---------------------------------------------------------------------------
# Steering-constrained A*
# ---------------------------------------------------------------------------

# 8-connected move table: (dcol, drow)
_MOVES = [
    (1,  0), (-1,  0), (0,  1), (0, -1),
    (1,  1), (1,  -1), (-1, 1), (-1, -1),
]
_DIAG = {(1, 1), (1, -1), (-1, 1), (-1, -1)}


def astar_steered(
    grid: dict,
    start_col: int,
    start_row: int,
    start_yaw: float,
    goal_col: int,
    goal_row: int,
    max_steer_rad: float,
    steering_cost_weight: float,
    cost_weight: float,
    n_headings: int,
) -> list:
    """
    Steering-constrained A* search.

    State: (col, row, heading_index)

    A move to a neighbour is rejected when the required heading change exceeds
    max_steer_rad.  Within-limit turns add a steering penalty proportional to
    |delta_heading| * steering_cost_weight, so the planner prefers gentle arcs.

    Returns
    -------
    List of (col, row, yaw_rad) from start to goal, or [] if no path found.
    """
    heading_step = 2.0 * math.pi / n_headings

    def yaw_to_idx(yaw: float) -> int:
        return int(round(yaw % (2.0 * math.pi) / heading_step)) % n_headings

    def idx_to_yaw(idx: int) -> float:
        return idx * heading_step

    # Precompute move headings and step distances once
    move_info = []
    for dcol, drow in _MOVES:
        move_yaw = math.atan2(drow, dcol)
        step     = math.sqrt(2.0) if (dcol, drow) in _DIAG else 1.0
        move_info.append((move_yaw, step))

    start_hidx  = yaw_to_idx(start_yaw)
    start_state = (start_col, start_row, start_hidx)
    goal_xy     = (goal_col, goal_row)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_state))

    came_from: dict = {start_state: None}
    g_score:   dict = {start_state: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        col, row, hidx = current

        if (col, row) == goal_xy:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                c, r, hi = node
                path.append((c, r, idx_to_yaw(hi)))
                node = came_from[node]
            path.reverse()
            return path

        current_yaw = idx_to_yaw(hidx)

        for (dcol, drow), (move_yaw, step_dist) in zip(_MOVES, move_info):
            nb_col = col + dcol
            nb_row = row + drow

            if (nb_col, nb_row) not in grid:
                continue  # obstacle or outside map

            # --- Steering constraint -----------------------------------------
            turn = abs(angle_diff(move_yaw, current_yaw))
            if turn > max_steer_rad:
                continue  # exceeds maximum steering angle — reject move

            # --- Travel cost -------------------------------------------------
            cell_cost   = grid[(nb_col, nb_row)]
            steer_cost  = steering_cost_weight * turn
            travel_cost = step_dist + cost_weight * cell_cost + steer_cost

            nb_hidx   = yaw_to_idx(move_yaw)
            nb_state  = (nb_col, nb_row, nb_hidx)
            tentative_g = g_score[current] + travel_cost

            if tentative_g < g_score.get(nb_state, math.inf):
                came_from[nb_state] = current
                g_score[nb_state]   = tentative_g
                f = tentative_g + heuristic(nb_col, nb_row, goal_col, goal_row)
                heapq.heappush(open_heap, (f, nb_state))

    return []  # no path found


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TrajectoryPlannerNode(Node):

    def __init__(self):
        super().__init__('trajectory_planner')

        # --- parameters ---
        self.declare_parameter('frame_id',               'map')
        self.declare_parameter('cost_weight',             2.0)
        self.declare_parameter('max_cost_threshold',      90)
        self.declare_parameter('waypoint_spacing',        5)
        self.declare_parameter('max_steering_angle_deg',  30.0)
        self.declare_parameter('steering_cost_weight',    1.5)
        self.declare_parameter('n_headings',              16)

        self.frame_id             = self.get_parameter('frame_id').value
        self.cost_weight          = self.get_parameter('cost_weight').value
        self.max_cost             = self.get_parameter('max_cost_threshold').value
        self.wp_spacing           = self.get_parameter('waypoint_spacing').value
        self.max_steer_rad        = math.radians(
                                        self.get_parameter('max_steering_angle_deg').value)
        self.steering_cost_weight = self.get_parameter('steering_cost_weight').value
        self.n_headings           = self.get_parameter('n_headings').value

        # --- state ---
        self.costmap_data:     np.ndarray | None = None
        self.costmap_res:      float = 0.1
        self.costmap_origin_x: float = 0.0
        self.costmap_origin_y: float = 0.0
        self.start: Pose2D | None = None
        self.goal:  Pose2D | None = None

        self._latest_path:      Path        | None = None
        self._latest_goal_mk:   Marker      | None = None
        self._latest_wp_mk:     MarkerArray | None = None
        self._latest_goal_pose: PoseStamped | None = None

        # --- subscribers ---
        self.create_subscription(OccupancyGrid, '/master_costmap', self._cb_costmap, 10)
        self.create_subscription(Pose2D,        '/rover_pose_2D',  self._cb_start,   10)

        # --- publishers ---
        self.pub_path      = self.create_publisher(Path,        '/waypoints',        10)
        self.pub_goal_mk   = self.create_publisher(Marker,      '/goal_marker',      10)
        self.pub_wp_mk     = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, '/goal_pose',        10)

        self.plan_service = self.create_service(Trigger,'/plan_trajectory',self._srv_plan_trajectory)

        self.create_timer(1.0, self._publish_latest)

        self.get_logger().info(
            f'TrajectoryPlannerNode started  '
            f'(max_steer={math.degrees(self.max_steer_rad):.1f} deg, '
            f'n_headings={self.n_headings})'
        )
        self.get_logger().info('Ready — call:  ros2 service call /plan_trajectory std_srvs/srv/Trigger "{}"')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _srv_plan_trajectory(self, request, response):

        if self.start is None:
            response.success = False
            response.message = 'No start pose received.'
            return response

        if self.costmap_data is None:
            response.success = False
            response.message = 'No costmap received.'
            return response

        self.get_logger().info('Trajectory planning service called.')

        self._plan()

        response.success = True
        response.message = 'Trajectory planned successfully.'

        return response

    def _cb_costmap(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            self.get_logger().warn('Received empty OccupancyGrid. Ignored.')
            return

        self.costmap_res      = msg.info.resolution
        self.costmap_origin_x = msg.info.origin.position.x
        self.costmap_origin_y = msg.info.origin.position.y
        self.costmap_data     = np.array(msg.data, dtype=np.int8).reshape(h, w)

        self.get_logger().info(
            f'OccupancyGrid received: {w}x{h} cells, '
            f'res={self.costmap_res:.3f} m, '
            f'origin=({self.costmap_origin_x:.2f}, {self.costmap_origin_y:.2f})'
        )
        #self._try_plan()

    def _cb_start(self, msg: Pose2D):
        self.start = msg
        # Temporary fixed goal: 1 m straight ahead in the rover's heading direction
        self.goal = Pose2D(
            x=msg.x + math.cos(msg.theta),
            y=msg.y + math.sin(msg.theta),
            theta=msg.theta,
        )
        self.get_logger().info(
            f'Start updated: ({msg.x:.2f}, {msg.y:.2f}, '
            f'{math.degrees(msg.theta):.1f} deg)'
        )
        #self._try_plan()

    # -----------------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------------

    def _try_plan(self):
        if self.start is None or self.costmap_data is None:
            return
        self._plan()

    def _plan(self):
        arr = self.costmap_data
        res = self.costmap_res
        ox  = self.costmap_origin_x
        oy  = self.costmap_origin_y

        # ------------------------------------------------------------------
        # Build passable-cell dict  {(col, row): float_cost}
        # Obstacles: value == -1 (unknown) or value >= max_cost_threshold
        # ------------------------------------------------------------------
        passable_mask      = (arr >= 0) & (arr < self.max_cost)
        rows_idx, cols_idx = np.where(passable_mask)
        costs              = arr[rows_idx, cols_idx].astype(float)
        grid: dict = {
            (int(c), int(r)): float(cost)
            for r, c, cost in zip(rows_idx, cols_idx, costs)
        }

        # ------------------------------------------------------------------
        # Coordinate helpers
        # ------------------------------------------------------------------
        def world_to_cell(wx: float, wy: float):
            return (int(round((wx - ox) / res)),   # col
                    int(round((wy - oy) / res)))    # row

        def cell_to_world(col: int, row: int):
            return ox + col * res, oy + row * res

        # ------------------------------------------------------------------
        # Discretise start / goal; force them passable
        # ------------------------------------------------------------------
        s_col, s_row = world_to_cell(self.start.x, self.start.y)
        g_col, g_row = world_to_cell(self.goal.x,  self.goal.y)
        grid.setdefault((s_col, s_row), 0.0)
        grid.setdefault((g_col, g_row), 0.0)

        # ------------------------------------------------------------------
        # Steering-constrained A*
        # ------------------------------------------------------------------
        raw_path = astar_steered(
            grid                 = grid,
            start_col            = s_col,
            start_row            = s_row,
            start_yaw            = self.start.theta,
            goal_col             = g_col,
            goal_row             = g_row,
            max_steer_rad        = self.max_steer_rad,
            steering_cost_weight = self.steering_cost_weight,
            cost_weight          = self.cost_weight,
            n_headings           = self.n_headings,
        )

        if not raw_path:
            self.get_logger().warn(
                f'No path found (max_steer={math.degrees(self.max_steer_rad):.1f} deg). '
                'Consider increasing max_steering_angle_deg.'
            )
            return

        self.get_logger().info(f'A* path found: {len(raw_path)} cells.')

        # ------------------------------------------------------------------
        # Subsample -> waypoints  (col, row, yaw)
        # ------------------------------------------------------------------
        waypoints_cells = raw_path[::self.wp_spacing]
        if waypoints_cells[-1] != raw_path[-1]:
            waypoints_cells.append(raw_path[-1])

        # Convert to world coordinates; heading is already stored in raw_path
        waypoints = []
        for col, row, yaw in waypoints_cells:
            wx, wy = cell_to_world(col, row)
            waypoints.append((wx, wy, yaw))

        # Override final waypoint heading with exact goal theta
        if waypoints:
            wx, wy, _ = waypoints[-1]
            waypoints[-1] = (wx, wy, self.goal.theta)

        self.get_logger().info(f'Waypoints: {len(waypoints)}')

        # ------------------------------------------------------------------
        # Build ROS messages
        # ------------------------------------------------------------------
        now    = self.get_clock().now().to_msg()
        header = Header(frame_id=self.frame_id, stamp=now)

        self._latest_path      = self._build_path(header, waypoints)
        self._latest_goal_mk   = self._build_goal_marker(header)
        self._latest_wp_mk     = self._build_waypoint_markers(header, waypoints)
        self._latest_goal_pose = self._build_goal_pose(header)

        self._publish_latest()

    # -----------------------------------------------------------------------
    # Message builders
    # -----------------------------------------------------------------------

    def _build_path(self, header: Header, waypoints: list) -> Path:
        path = Path(header=header)
        for wx, wy, yaw in waypoints:
            ps = PoseStamped(header=header)
            ps.pose.position.x  = wx
            ps.pose.position.y  = wy
            ps.pose.position.z  = 0.0
            ps.pose.orientation = yaw_to_quaternion(yaw)
            path.poses.append(ps)
        return path

    def _build_goal_marker(self, header: Header) -> Marker:
        """Green arrow at the goal pose."""
        mk          = Marker(header=header)
        mk.ns       = 'goal'
        mk.id       = 0
        mk.type     = Marker.ARROW
        mk.action   = Marker.ADD
        mk.pose.position.x  = self.goal.x
        mk.pose.position.y  = self.goal.y
        mk.pose.position.z  = 0.05
        mk.pose.orientation = yaw_to_quaternion(self.goal.theta)
        mk.scale.x  = 0.4
        mk.scale.y  = 0.08
        mk.scale.z  = 0.08
        mk.color    = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)   # green
        return mk

    def _build_waypoint_markers(self, header: Header, waypoints: list) -> MarkerArray:
        """Blue spheres + heading arrows for each waypoint."""
        array = MarkerArray()

        delete_all = Marker(header=header)

        delete_all.header = header
        delete_all.action = Marker.DELETEALL

        # IMPORTANT:
        # Do NOT set namespace/id for DELETEALL markers

        array.markers.append(delete_all)

        for i, (wx, wy, yaw) in enumerate(waypoints):
            sphere             = Marker(header=header)
            sphere.ns          = 'waypoints'
            sphere.id          = i * 2
            sphere.type        = Marker.SPHERE
            sphere.action      = Marker.ADD
            sphere.pose.position.x  = wx
            sphere.pose.position.y  = wy
            sphere.pose.position.z  = 0.05
            sphere.pose.orientation = yaw_to_quaternion(0.0)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
            sphere.color   = ColorRGBA(r=0.0, g=0.4, b=1.0, a=0.9)   # blue
            array.markers.append(sphere)

            arrow             = Marker(header=header)
            arrow.ns          = 'waypoints'
            arrow.id          = i * 2 + 1
            arrow.type        = Marker.ARROW
            arrow.action      = Marker.ADD
            arrow.pose.position.x  = wx
            arrow.pose.position.y  = wy
            arrow.pose.position.z  = 0.05
            arrow.pose.orientation = yaw_to_quaternion(yaw)
            arrow.scale.x = 0.25
            arrow.scale.y = 0.05
            arrow.scale.z = 0.05
            arrow.color   = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.8)    # light blue
            array.markers.append(arrow)

        return array

    def _build_goal_pose(self, header: Header) -> PoseStamped:
        gp                  = PoseStamped(header=header)
        gp.pose.position.x  = self.goal.x
        gp.pose.position.y  = self.goal.y
        gp.pose.position.z  = 0.0
        gp.pose.orientation = yaw_to_quaternion(self.goal.theta)
        return gp

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_latest(self):
        if self.start is None or self.costmap_data is None:
            return
        if self._latest_path      is not None: self.pub_path.publish(self._latest_path)
        if self._latest_goal_mk   is not None: self.pub_goal_mk.publish(self._latest_goal_mk)
        if self._latest_wp_mk     is not None: self.pub_wp_mk.publish(self._latest_wp_mk)
        if self._latest_goal_pose is not None: self.pub_goal_pose.publish(self._latest_goal_pose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()