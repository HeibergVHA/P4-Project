import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import time
import threading
import serial
import json


def BSpline(points, smoothing, k):
    points = np.array(points)
    if len(points) < k + 1:
        raise ValueError(f"Need at least {k+1} points for a degree-{k} spline, got {len(points)}")

    # Chord-length parameterisation
    d = 0
    t = [0]
    for i in range(len(points) - 1):
        d += np.sqrt(np.sum((points[i+1] - points[i])**2))
        t.append(d)
    t = np.array(t)
    t = t / t[-1]   # normalise 0→1

    tck, _ = splprep([points[:, 0], points[:, 1]], u=t, s=smoothing, k=k)
    total_length = t[-1]
    n_points = int(np.clip(total_length * 10, 50, 5000))
    u_fine  = np.linspace(0, 1, n_points)

    x_u,  y_u  = splev(u_fine, tck)         # positions
    dx_u, dy_u = splev(u_fine, tck, der=1)  # first derivative
    ddx_u, ddy_u = splev(u_fine, tck, der=2)  # second derivative

    speed = np.sqrt(dx_u**2 + dy_u**2)
    kp = (dx_u * ddy_u - dy_u * ddx_u) / (speed**3 + 1e-9)  # Curvature for General Parametrizations.
    v_max = np.sqrt(1.0 / (np.abs(kp) + 1e-6))              # Max speed based on curvature. 
    v_max = np.clip(v_max, 0, 80)

    return v_max, x_u, y_u


class UGVController(Node):
    def __init__(self):
        super().__init__('ugv_controller')

        self.declare_parameter('input_source', 'serial')
        self.source = self.get_parameter('input_source').value
        self.get_logger().info(f"Using input source: {self.source}")

        # Serial
        self.ser = serial.Serial('/dev/ttyUSB1', baudrate=115200, timeout=1.0)
        self.zero_found = False       # FIX: declare before serial thread starts

        # Subscribers
        self.create_subscription(String, '/ugv/state', self.state_callback, 10)

        if self.source == 'ros':
            self.create_subscription(
                PoseStamped, 'ugv/position', self.position_callback, 10)

        self.create_subscription(String, 'ugv/astar_points',   self.astar_points_callback, 10)
        self.create_subscription(String, 'uav/radio_in/mission_command', self.mission_command_callback, 10)

        # Publishers
        self.steering_angle_pub = self.create_publisher(Float32, '/ugv/steering_angle', 10)

        # State
        self.current_x     = 0.0   # metres, world frame
        self.current_y     = 0.0
        self.current_z     = 0.0
        self.current_angle = 0.0   # degrees

        self.target_x = 0.0
        self.target_y = 0.0

        self.astar_points = []
        self.spline_ready = False
        self.s       = None
        self.x_smooth = None
        self.y_smooth = None

        self.L = 0.480          # wheelbase
        self.lookahead = 0.5    # lookahead distance
        self.metres_per_count = 0.57203 / 100 # Wheel circumference / counts per revolution

        self.RATE_HZ = 20
        self.dt = 1.0 / self.RATE_HZ
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Serial thread
        self.last_steer_deg = 90   # 90 = straight ahead after the +90 offset
        self.last_sent = 0
        self.serial_thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.serial_thread.start()

        self.get_logger().info('ugv_controller node started')

    # Callbacks
    def state_callback(self, msg):
        self.get_logger().info(f"UGV state: {msg.data}")

    def mission_command_callback(self, msg):
        self.get_logger().info(f"Mission command received: {msg.data}")

    def position_callback(self, msg): # Used when input_source == 'ros'.
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

        # Convert quaternion to yaw
        r = R.from_quat([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        yaw_deg = r.as_euler('zxy', degrees=True)[0]
        self.current_angle = yaw_deg

        if self.spline_ready:
            self.target_x, self.target_y = self.target_point(self.x_smooth, self.y_smooth)

    def astar_points_callback(self, msg): # expected format: "[[x0,y0],[x1,y1],...]"
        try:
            parsed = json.loads(msg.data)
            self.astar_points = [list(p) for p in parsed]
        except (json.JSONDecodeError, ValueError) as e:
            self.get_logger().error(f"Bad astar_points message: {e}")
            return

        if len(self.astar_points) < 6:
            self.get_logger().warn("Too few waypoints for degree-5 spline; need ≥ 6.")
            return

        try:
            points_np = np.array(self.astar_points, dtype=float)
            self.s, self.x_smooth, self.y_smooth = BSpline(points_np, 20000, 5)
            self.spline_ready = True
            self.target_x, self.target_y = self.target_point(self.x_smooth, self.y_smooth)
            self.get_logger().info(f"Spline fitted to {len(self.astar_points)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"BSpline fitting failed: {e}")


    def target_point(self, x_smooth, y_smooth): # Walk lookahead distance along the spline from the closest point.
        # Find the index of the spline point closest to the car
        dists = np.sqrt((x_smooth - self.current_x)**2 + (y_smooth - self.current_y)**2) # list
        u0 = int(np.argmin(dists))

        # Walk forward along the spline until we've covered tDistance
        s = 0.0
        uL = u0
        for i in range(u0, len(x_smooth) - 1):
            ds = np.sqrt((x_smooth[i+1] - x_smooth[i])**2 + (y_smooth[i+1] - y_smooth[i])**2)
            s += ds
            if s >= self.lookahead:
                uL = i
                break
        else:
            uL = len(x_smooth) - 1   # reached end of spline

        return float(x_smooth[uL]), float(y_smooth[uL])

    def steering_angle(self, target_x, target_y):
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        # Target in world frame (radians)
        angle_world = np.arctan2(dy, dx)

        # Alpha: angular error between heading and bearing to target
        # Normalise to [-pi, pi]
        alpha = angle_world - np.radians(self.current_angle)
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        Ld = np.hypot(dx, dy)
        if Ld < 1e-6:
            return 0.0   # already at target — go straight

        # Bicycle model
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), Ld)
        return delta

    # Serial thread
    def serial_loop(self): #Arduino sends lines like 'angle_deg,encoder_counts\n'
        while True:
            try:

                # Write
                if self.last_sent < time.time() - self.dt:
                    steer_byte = bytes([int(np.clip(self.last_steer_deg, 0, 180))])
                    self.ser.write(steer_byte)

                if self.source == 'ros':
                    continue

                # Read
                raw = self.ser.readline().decode(errors='ignore').strip()
                if raw and ',' in raw:
                    try:
                        angle_str, count_str = raw.split(',', 1)
                        angle_deg = float(angle_str)
                        count     = float(count_str)

                        if not self.zero_found: # Sanity check because the arduino sometimes sends wrong encoder counts before/during reset.
                            if count < 1:
                                self.zero_found = True
                        else:
                            self.current_angle = angle_deg
                            distance = count * self.metres_per_count
                            self.current_x += distance * np.cos(self.current_angle)
                            self.current_y += distance * np.sin(self.current_angle)

                            if self.spline_ready:
                                self.target_x, self.target_y = self.target_point(
                                    self.x_smooth, self.y_smooth)

                    except ValueError:
                        self.get_logger().warn(f"Malformed serial line: {raw}")

            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}")
                time.sleep(1.0)

            time.sleep(self.dt/2) # 40 Hz

    # Control loop (20 Hz timer)
    def control_loop(self):
        if not self.spline_ready:
            return

        delta = self.steering_angle(self.target_x, self.target_y)

        # Convert to degrees and map to between 0 and 180 instead of -90 to 90.
        steer_deg = np.degrees(-delta)
        steer_deg = np.clip(steer_deg, -90, 90)
        steer_deg = steer_deg + 90

        self.last_steer_deg = int(steer_deg)
        msg = Float32()
        msg.data = float(steer_deg)
        self.steering_angle_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UGVController()
    rclpy.spin(node)
    rclpy.shutdown()