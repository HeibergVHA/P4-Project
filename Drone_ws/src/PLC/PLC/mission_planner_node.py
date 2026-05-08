import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# try:
#     from px4_msgs.msg import VehicleOdometry, VehicleAttitudeSetpoint, \
#     OffboardControlMode, VehicleCommand
#     PX4_AVAILABLE = True
# except ImportError:
#     PX4_AVAILABLE = False


# ros2 run <package> mission_planner_node --ros-args -p input_source:=local lookahead_dist:=2.0 waypoint_radius:=1.0 publish_rate:=50.0


# Waypoint list: (x [m], y [m], z [m], yaw [deg])

DEFAULT_WAYPOINTS = [
    (  0.0,   0.0,  0.0,   0.0),
    (  0.0,   0.0,  2.0,   0.0),
    ( 10.0,   0.0,  2.0,   0.0),
    ( 10.0,  10.0,  2.0,  90.0),
    (  0.0,  10.0,  2.0, 180.0),
    (  0.0,  20.0,  2.0,  90.0),
    ( 10.0,  20.0,  2.0,   0.0),
    ( 10.0,  30.0,  2.0,  90.0),
    (  0.0,  30.0,  2.0, 180.0),
    (  0.0,   0.0,  2.0, 180.0)
]

# DEFAULT_WAYPOINTS = [
#     (  0.0,   0.0,  -2.0,   0.0),
#     ( 10.0,   0.0,  -5.0,   0.0),
#     ( 10.0,   10.0,  -5.0,   0.0),
#     ( 0.0,   10.0,  -5.0,   0.0),
#     (  0.0,   0.0,  -5.0,  0.0)
# ]
# DEFAULT_WAYPOINTS = [
#     # wheel
#     (  3.0,   0.0,  -5.0,  0.0),
#     (  1.6,   2.6,  -5.0,  0.0),
#     ( -0.8,   2.9,  -5.0,  0.0),
#     ( -2.8,   0.4,  -5.0,  0.0),
#     ( -2.8,  -0.4,  -5.0,  0.0),
#     ( -0.8,  -2.9,  -5.0,  0.0),

#     # barrel out
#     (  3.0,   0.0,  -5.0,  0.0),
#     (  5.0,   0.0,  -5.0,  0.0),
#     (  7.0,   0.0,  -5.0,  0.0),
#     (  9.0,   0.0,  -5.0,  0.0),
#     ( 11.0,   0.0,  -5.0,  0.0),

#     # cannonball returning
#     (  9.0,   1.5,  -5.0,  0.0),
#     (  7.0,   3.0,  -5.0,  0.0),
#     (  5.0,   2.0,  -5.0,  0.0),
#     (  0.0,   0.0,  -5.0,  0.0),
# ]
# Helpers
def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9: # Float check to make sure v is not devided by 0.
        return np.zeros_like(v)
    return v / n

def project_point_to_segment(point, seg_point_a, seg_point_b):
    seg_ab = seg_point_b - seg_point_a
    ab_len2 = np.dot(seg_ab, seg_ab)
    if ab_len2 < 1e-9:
        return seg_point_a.copy(), 0.0
    distance_along_seg = np.dot(point - seg_point_a, seg_ab) / ab_len2
    distance_along_seg = np.clip(distance_along_seg, 0.0, 1.0)
    closest_point = seg_point_a + distance_along_seg * seg_ab
    return closest_point, distance_along_seg

def yaw_to_quaternion_wxyz(yaw_deg): # Level-flight quaternion for a given yaw angle degrees.
    r = R.from_euler('z', yaw_deg, degrees=True)
    x, y, z, w = r.as_quat()   # scipy gives [x, y, z, w]
    return np.array([w, x, y, z])

def slerp_yaw(yaw_a_deg, yaw_b_deg, t): # Interpolate between two yaw angles (degrees) by fraction t in [0, 1]. Handles wraparound correctly
    t = float(np.clip(t, 0.0, 1.0))
    delta = (yaw_b_deg - yaw_a_deg + 180.0) % 360.0 - 180.0   # shortest arc
    return yaw_a_deg + t * delta


class PurePursuitMission(Node):
    def __init__(self):
        super().__init__('mission_planner')

        self.declare_parameter('input_source',   'local') # Input/startup parameters
        self.declare_parameter('lookahead_dist',  2.0)
        self.declare_parameter('capture_radius',  1.0)
        self.declare_parameter('publish_rate',   50.0)
        source                  = self.get_parameter('input_source').value
        self.max_lookahead      = float(self.get_parameter('lookahead_dist').value)
        self.waypoint_radius    = float(self.get_parameter('capture_radius').value)
        publish_rate            = float(self.get_parameter('publish_rate').value)
        self.L_current          = 0.0
        self.L_ramp_rate        = 1.0
        self.hold_pos           = np.zeros(3)
        self.hold_yaw           = 0.0
        self.mode               = "TRACK"
        self.current_lookahead  = 0.0
        self.t_transition_start = 0.0

        # QoS 
        px4_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 5
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.clock = self.get_clock()

        self.get_logger().info(f"MissionPlanner | source={source} | L={self.max_lookahead} m | waypoint_radius={self.waypoint_radius} m | rate={publish_rate} Hz")

        # Subscribers
        if source == 'local':
            self.create_subscription(
                PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos)
        elif source == 'vicon':
            self.create_subscription(
                PoseStamped, 'vicon_pose', self.vicon_pos_callback, 10)
        # elif source == 'px4':
        #     self.create_subscription(
        #         VehicleOdometry, '/fmu/out/vehicle_odometry', self.px4_odometry_callback, px4_qos)

        self.create_subscription( # Radio: append waypoint "x,y,z,yaw"
            String, 'uav/radio_in/target_waypoint', self.radio_waypoint_callback, 10)

        self.create_subscription( # Radio: mission commands  "start" | "pause" | "resume" | "abort"
            String, 'uav/radio_in/mission_command', self.radio_command_callback, 10)

        # Publisher
        self.target_pub = self.create_publisher(
            PoseStamped, 'uav/target_pos', 10)

        # Waypoint list
        self.waypoints= [np.array([x, y, z, yaw], dtype=float) for x, y, z, yaw in DEFAULT_WAYPOINTS]
        self.seg_idx = 0        # Index of the of the active segment. segment i goes from waypoints[i] to waypoints[i+1]

        # State
        self.pos = np.zeros(3)  # Current position [x, y, z]

        if source == 'px4':
            self.mission_active = True
        else:
            self.mission_active = True

        # Loop timer
        dt = 1.0 / publish_rate
        self.create_timer(dt, self.control_loop)

        self.get_logger().info('MissionPlanner node started — send "start", "pause", "resume", or "abort"')

    # Callbacks
    def local_pos_callback(self, msg):
        self.pos[:] = [msg.pose.position.x,
                       msg.pose.position.y,
                       msg.pose.position.z]

    def vicon_pos_callback(self, msg):
        self.pos[:] = [msg.pose.position.x,
                       msg.pose.position.y,
                       msg.pose.position.z]
        
    def px4_odometry_callback(self, msg):
        self.pos[:] = [msg.position[0],
                       msg.position[1],
                       msg.position[2]]

    def radio_waypoint_callback(self, msg): # Append a waypoint received over radio as 'x,y,z,yaw'.
        try:
            parts = [float(v) for v in msg.data.split(',')]
            if len(parts) != 4:
                raise ValueError("Need exactly 4 values: x,y,z,yaw")
            self.waypoints.append(np.array(parts))
            self.get_logger().info(f'Waypoint added: {parts}')
        except Exception as e:
            self.get_logger().warn(f'Bad waypoint message "{msg.data}": {e}')

    def radio_command_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            self.mission_active = True
            self.seg_idx = 0
            self.get_logger().info('Mission STARTED')
        elif cmd == 'pause':
            self.mission_active = False
            self.get_logger().info('Mission PAUSED — holding current reference')
        elif cmd == 'resume':
            self.mission_active = True
            self.get_logger().info('Mission RESUMED')
        elif cmd == 'abort':
            self.mission_active = False
            self.seg_idx = 0
            self.get_logger().warn('Mission ABORTED')
        else:
            self.get_logger().warn(f'Unknown command: "{cmd}"')

    # Pure-pursuit
    def pure_pursuit(self):
        t = self.clock.now().nanoseconds * 1e-9

        # Waypoints
        if self.seg_idx >= len(self.waypoints) - 1:
            wp = self.waypoints[-1]
            return wp[:3], wp[3]

        A = self.waypoints[self.seg_idx][:3]
        B = self.waypoints[self.seg_idx + 1][:3]

        has_next = (self.seg_idx + 2 < len(self.waypoints))
        if has_next:
            C = self.waypoints[self.seg_idx + 2][:3]

        p = self.pos

        dist_to_B = np.linalg.norm(p - B)

        closest_point, distance_along_seg = project_point_to_segment(p, A, B)

        if has_next:
            f_BC, distance_along_seg_BC = project_point_to_segment(p, B, C)
            d_AB = np.linalg.norm(p - closest_point)
            d_BC = np.linalg.norm(p - f_BC)
        else:
            d_AB = np.linalg.norm(p - closest_point)
            d_BC = np.inf

        # Lookahead computation
        dist_CP_B = np.linalg.norm(closest_point - B)
        dt = t - self.t_transition_start
        self.current_lookahead = min(self.max_lookahead, self.L_ramp_rate * dt) # Lookahead distance

        # Mode
        if self.mode == "TRACK":
            ref_out = closest_point + unit(B - A) * self.current_lookahead # Lookahead point
            if dist_CP_B <= self.max_lookahead:
                self.mode = "APPROACH"
                self.get_logger().info('Mode: APPROACH')
        elif self.mode == "APPROACH":
            ref_out = B.copy()
            if dist_to_B <= self.waypoint_radius:
                self.mode = "TRANSITION"
                self.get_logger().info('Mode: TRANSITION')
                self.t_transition_start = t
        elif self.mode == "TRANSITION":
            if has_next:
                ref_out = B.copy() + unit(C - B) * self.current_lookahead
                _, lookahead_distance_along_seg_BC = project_point_to_segment(ref_out, B, C)
                if (d_BC < d_AB) and distance_along_seg_BC < lookahead_distance_along_seg_BC or (d_BC < d_AB) and distance_along_seg_BC > self.max_lookahead:
                    self.t_transition_start = t
                    self.seg_idx += 1
                    self.mode = "TRACK"
                    self.get_logger().info('Mode: TRACK')
            else:
                ref_out = B.copy()

        # Yaw interpolation 
        yaw_A = self.waypoints[self.seg_idx][3] # Yaw at last waypoint
        yaw_B = self.waypoints[self.seg_idx + 1][3] # Target yaw at next waypoint

        delta_yaw = (yaw_B - yaw_A + 180.0) % 360.0 - 180.0 # shortest arc (wraparound handeling)
        yaw = yaw_A + float(distance_along_seg) * delta_yaw # Yaw based on progress along segment

        return ref_out, yaw

    def publish(self, target_xyz: np.ndarray, target_yaw: float):
        q_wxyz = yaw_to_quaternion_wxyz(target_yaw)
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(target_xyz[0])
        msg.pose.position.y = float(target_xyz[1])
        msg.pose.position.z = float(target_xyz[2])
        msg.pose.orientation.w = float(q_wxyz[0])
        msg.pose.orientation.x = float(q_wxyz[1])
        msg.pose.orientation.y = float(q_wxyz[2])
        msg.pose.orientation.z = float(q_wxyz[3])

        self.target_pub.publish(msg)

    # Control loop
    def control_loop(self):
        if not self.mission_active:
            self.publish(self.hold_pos, self.hold_yaw)
            return

        # Compute lookahead reference
        target_xyz, target_yaw = self.pure_pursuit()

        # Publish
        self.publish(target_xyz, target_yaw)


        self.hold_pos = target_xyz.copy()
        self.hold_yaw = target_yaw



def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitMission()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()