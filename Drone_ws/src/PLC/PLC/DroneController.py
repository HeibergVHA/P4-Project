import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # 'Pose' data type.
from mavros_msgs.msg import State, Thrust
from mavros_msgs.srv import CommandBool, SetMode
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Maths helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])


def dcm_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    Shepperd's method — numerically stable.
    This is exactly what PX4's Quaternion(Dcm) constructor does.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Canonical form: w >= 0  (same as PX4's Quaternion::canonical())
    return q if q[0] >= 0.0 else -q


def body_z_to_attitude(body_z_sp: np.ndarray, yaw_sp: float):
    """
    Exact Python port of PX4 ControlMath::bodyzToAttitude().

    Given a desired body-z axis direction (unit vector, NED world frame)
    and a yaw setpoint, construct the full attitude DCM and return the
    quaternion [w, x, y, z] and the 3x3 rotation matrix.

    In NED: body-z points DOWN when level, so for upward thrust
            thr_sp = [tx, ty, -tz_magnitude]  and  body_z = -thr_sp normalised.

    Steps (mirroring ControlMath.cpp):
      1. body_z = normalize(-thr_sp)          [done by caller: thrustToAttitude]
      2. desired_body_x direction from yaw_sp: x_C = [cos(yaw), sin(yaw), 0]
      3. body_y = normalize(body_z × x_C)
      4. body_x = body_y × body_z
      5. R = [body_x | body_y | body_z]       (columns)
      6. q = quaternion(R)
    """
    # Zero vector guard (PX4 sets body_z = world_z = [0,0,1] NED)
    if np.linalg.norm(body_z_sp) < 1e-8:
        body_z_sp = np.array([0.0, 0.0, 1.0])

    body_z = normalize(body_z_sp)

    # Desired heading direction projected onto horizontal plane
    x_c = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0.0])

    # body_y = body_z × x_C  (right-hand rule)
    body_y_raw = np.cross(body_z, x_c)

    if np.linalg.norm(body_y_raw) < 1e-8:
        # Degenerate: body_z parallel to x_C → use world x as fallback
        body_y_raw = np.cross(body_z, np.array([1.0, 0.0, 0.0]))

    body_y = normalize(body_y_raw)

    # body_x = body_y × body_z
    body_x = np.cross(body_y, body_z)

    # Rotation matrix: columns are body axes expressed in world frame
    R = np.column_stack([body_x, body_y, body_z])   # shape (3,3)

    q = dcm_to_quaternion_wxyz(R)
    return q, R


def thrust_to_attitude(thr_sp: np.ndarray, yaw_sp: float):
    """
    Exact Python port of PX4 ControlMath::thrustToAttitude().

      body_z = normalize(-thr_sp)
      att quaternion via bodyzToAttitude
      thrust_body[2] = -‖thr_sp‖
    """
    body_z_sp = normalize(-thr_sp)
    q, R = body_z_to_attitude(body_z_sp, yaw_sp)
    thrust_z = -float(np.linalg.norm(thr_sp))
    return q, thrust_z


# ══════════════════════════════════════════════════════════════════════════════
# XY Position → Thrust vector controller
# Replicates: PositionControl::_positionController() (P on position)
#             PositionControl::_velocityController()  (PID on velocity, XY part)
# ══════════════════════════════════════════════════════════════════════════════

class XYController:
    """
    Cascaded P(pos) → PID(vel) → thrust vector [tx, ty] (NED, normalized).

    Gains mirror PX4 defaults:
        MPC_XY_P       = 0.95   (position P)
        MPC_XY_VEL_P_ACC = 1.8  (velocity P, in acc units)
        MPC_XY_VEL_I_ACC = 0.4
        MPC_XY_VEL_D_ACC = 0.2
        MPC_THR_HOVER  = 0.73   (normalises acc → thrust)
    """

    def __init__(
        self,
        Kp_pos: float = 0.95,
        Kp_vel: float = 1.2,
        Ki_vel: float = 0.4,
        Kd_vel: float = 0.2,
        hover_thrust: float = 0.73,
        max_vel_xy: float = 5.0,       # m/s – MPC_XY_VEL_MAX
        max_tilt: float = np.radians(10.0),  # rad – MPC_TILTMAX_AIR
        max_thr_xy: float = 0.9,
        tau_d: float = 0.1,            # derivative LP filter time constant
    ):
        self.Kp_pos     = Kp_pos
        self.Kp_vel     = Kp_vel
        self.Ki_vel     = Ki_vel
        self.Kd_vel     = Kd_vel
        self.hover_thrust = hover_thrust
        self.max_vel_xy = max_vel_xy
        self.max_tilt   = max_tilt
        self.max_thr_xy = max_thr_xy
        self.tau_d      = tau_d

        self._vel_int   = np.zeros(2)   # integrator [x, y]
        self._vel_filt  = np.zeros(2)   # filtered velocity for D term

    def reset(self):
        self._vel_int  = np.zeros(2)
        self._vel_filt = np.zeros(2)

    def update(
        self,
        pos_sp: np.ndarray,   # [x_ref, y_ref]  NED [m]
        pos:    np.ndarray,   # [x,     y    ]  NED [m]
        vel:    np.ndarray,   # [vx,    vy   ]  NED [m/s]
        dt:     float,
    ) -> np.ndarray:
        """
        Returns thr_xy: the [tx, ty] portion of the NED thrust vector
                        (normalised, hover_thrust = 1 g unit).
        """
        if dt <= 0.0:
            return np.zeros(2)

        # ── 1. Position loop (P) → velocity setpoint ──────────────────
        vel_sp = self.Kp_pos * (pos_sp - pos)

        # Saturate horizontal velocity (MPC_XY_VEL_MAX)
        vel_sp_norm = np.linalg.norm(vel_sp)
        if vel_sp_norm > self.max_vel_xy:
            vel_sp = vel_sp / vel_sp_norm * self.max_vel_xy

        # ── 2. Velocity loop (PID) → acceleration setpoint ────────────
        vel_err = vel_sp - vel

        # Derivative: low-pass filtered measured velocity (D on measurement)
        alpha = self.tau_d / (self.tau_d + dt)
        self._vel_filt = alpha * self._vel_filt + (1.0 - alpha) * vel
        vel_dot_filt   = (self._vel_filt - vel) / dt   # ~0 but filtered

        acc_sp = (
            self.Kp_vel * vel_err
            + self.Ki_vel * self._vel_int
            - self.Kd_vel * self._vel_filt   # D on measurement (PX4 style)
        )

        # ── 3. Anti-windup (clamp integrator if saturated) ─────────────
        thr_xy_raw = acc_sp / 9.81 * self.hover_thrust
        saturated  = np.linalg.norm(thr_xy_raw) >= self.max_thr_xy

        # Only integrate when not saturated AND error pushes away from limit
        if not saturated:
            self._vel_int += vel_err * dt
        else:
            # Anti-reset windup: only integrate the component that reduces error
            for i in range(2):
                stop = (thr_xy_raw[i] >= self.max_thr_xy and vel_err[i] >= 0.0) or \
                       (thr_xy_raw[i] <= -self.max_thr_xy and vel_err[i] <= 0.0)
                if not stop:
                    self._vel_int[i] += vel_err[i] * dt

        # ── 4. acc → normalised thrust (horizontal part) ───────────────
        # PX4: thr_sp_xy = acc_sp_xy / g * hover_thrust
        thr_xy = acc_sp / 9.81 * self.hover_thrust

        # ── 5. Tilt limiting: project onto max tilt cone ───────────────
        # max horizontal thrust given the Z thrust must stay above min
        # simplified: limit by tan(max_tilt)
        max_thr_h = np.tan(self.max_tilt)   # relative to unit vertical
        thr_h_norm = np.linalg.norm(thr_xy)
        if thr_h_norm > max_thr_h:
            thr_xy = thr_xy / thr_h_norm * max_thr_h

        return thr_xy


# ══════════════════════════════════════════════════════════════════════════════
# Z Position → thrust_body[2] controller
# Replicates: PositionControl::_positionController() (P on z)
#             PositionControl::_velocityController()  (PID on vz)
# ══════════════════════════════════════════════════════════════════════════════

class ZController:
    """
    Cascaded P(z) → PID(vz) → collective thrust (NED, D-axis).

    PX4 defaults:
        MPC_Z_P          = 1.0
        MPC_Z_VEL_P_ACC  = 4.0
        MPC_Z_VEL_I_ACC  = 2.0
        MPC_Z_VEL_D_ACC  = 0.0  (usually 0 in z)
        MPC_THR_HOVER    = 0.73
        MPC_THR_MIN      = 0.12
        MPC_THR_MAX      = 0.9
        MPC_Z_VEL_MAX_UP = 3.0 m/s
        MPC_Z_VEL_MAX_DN = 1.0 m/s
    """

    def __init__(
        self,
        Kp_pos: float  = 1.0,
        Kp_vel: float  = 4.0,
        Ki_vel: float  = 2.0,
        Kd_vel: float  = 0.0,
        hover_thrust: float = 0.73,
        thr_min: float = 0.12,
        thr_max: float = 0.90,
        vel_max_up: float = 3.0,
        vel_max_dn: float = 1.0,
        tau_d: float = 0.1,
    ):
        self.Kp_pos      = Kp_pos
        self.Kp_vel      = Kp_vel
        self.Ki_vel      = Ki_vel
        self.Kd_vel      = Kd_vel
        self.hover_thrust = hover_thrust
        self.thr_min     = thr_min
        self.thr_max     = thr_max
        self.vel_max_up  = vel_max_up   # max upward speed   (vz < 0 = up in NED)
        self.vel_max_dn  = vel_max_dn   # max downward speed (vz > 0 = down in NED)
        self.tau_d       = tau_d

        self._vel_int  = 0.0
        self._vz_filt  = 0.0

    def reset(self):
        self._vel_int = 0.0
        self._vz_filt = 0.0

    def update(self, z_sp: float, z: float, vz: float, dt: float) -> float:
        """
        Returns thrust_body[2]: always negative (upward), clamped to [−thr_max, −thr_min].

        z_sp, z  in NED metres  (negative = above ground, e.g. z_sp = −2.0)
        vz       in NED m/s     (negative = climbing)
        """
        if dt <= 0.0:
            return -self.hover_thrust

        # ── 1. Position loop (P) → vz setpoint ────────────────────────
        # NED: z_sp < z means we are too low → need negative vz (climb)
        vz_sp = self.Kp_pos * (z_sp - z)
        vz_sp = float(np.clip(vz_sp, -self.vel_max_up, self.vel_max_dn))

        # ── 2. Velocity loop (PID) ─────────────────────────────────────
        vz_err = vz_sp - vz

        # D term: low-pass filter vz, differentiate on measurement
        alpha = self.tau_d / (self.tau_d + dt)
        self._vz_filt = alpha * self._vz_filt + (1.0 - alpha) * vz

        # PX4 _velocityController D-axis:
        #   thrust_D = Kp*err + Ki*integral + Kd*vel_dot - hover_thrust
        #   (equilibrium at hover_thrust → subtract to center around 0)
        thrust_d = (
            self.Kp_vel * vz_err
            + self.Ki_vel * self._vel_int
            - self.Kd_vel * self._vz_filt
            - self.hover_thrust          # subtract hover so 0 error → hover thrust
        )

        # NED: thrust_d is negative for upward. Saturate in NED sense:
        #   uMax = -thr_min  (least upward)
        #   uMin = -thr_max  (most upward)
        u_max = -self.thr_min
        u_min = -self.thr_max
        thrust_d = float(np.clip(thrust_d, u_min, u_max))

        # ── 3. Anti-windup ─────────────────────────────────────────────
        stop = (thrust_d >= u_max and vz_err >= 0.0) or \
               (thrust_d <= u_min and vz_err <= 0.0)
        if not stop:
            self._vel_int += vz_err * dt
            # Clamp integral magnitude
            self._vel_int = float(np.clip(self._vel_int,
                                          -self.thr_max, self.thr_max))

        return thrust_d   # already negative (NED upward)


################ "ros2 run pakage node --ros-args -p input_source:=vicon" #############

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        self.declare_parameter('input_source', 'vicon')         # Input parameter when node is started. "local" to get current position from the pixhawk
        source = self.get_parameter('input_source').value       # "vicon" to get current position from "vicon_posee" topic.
        self.get_logger().info(f"Using input source: {source}") # I think like this: "ros2 run pakage node --ros-args -p input_source:=vicon"

        # Subscribers
        self.state_sub = self.create_subscription(              # Info about if the drone is connected, armed, and state.
            State, '/mavros/state', self.state_callback, 10)
        
        if source == 'local':
            self.local_pos_sub = self.create_subscription(      # Local position. ArduPilot EKF position.
                PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, 10)
            
        if source == 'vicon':
            self.vicon_pos_sub = self.create_subscription(      # Vicon topic.
                PoseStamped, 'vicon_pose', self.vicon_pos_callback, 10)
            
        self.target_pos_sub = self.create_subscription(         # Target position topic.
            PoseStamped, 'target_pos', self.target_pos_callback, 10)

        # Publishers 
        self.att_pub = self.create_publisher(                   # Control drone orientation (attitude) mavros topic.
            PoseStamped, '/mavros/setpoint_attitude/attitude', 10)

        self.thr_pub = self.create_publisher(                   # Control drone thrust mavros topic. This and attitude have to be published with identical time stamp for some reason... idk why... Claude said so...
            Thrust, '/mavros/setpoint_attitude/thrust', 10)

        # Service clients
        self.arming_client = self.create_client(                # To arm the drone...
            CommandBool, '/mavros/cmd/arming')

        self.set_mode_client = self.create_client(              # To set mode...
            SetMode, '/mavros/set_mode')

        # Controllers
        self.xy_controller = XYController(hover_thrust=0.73)
        self.z_controller  = ZController (hover_thrust=0.73)

        # State
        self.current_state = State()        # Stores if the drone is connected, armed, and state.

        # Current postition
        self.current_x = 0.0            # Meters
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_P = np.array([self.current_x, self.current_y, self.current_z])

        # Target position
        self.target_x = 0.0             # Meters
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_P = np.array([self.target_x, self.target_y, self.target_z])

        ############## EDIT THIS ONE ######################
        self.current_vel = 0.0

        # Orientation (attitude) setpoints
        self.target_roll  = 0.0         # Degrees
        self.target_pitch = 0.0
        self.target_yaw   = 0.0
        self.target_thrust = 0.0        # Thrust value 0.0-1.0

        self.RATE_HZ = 50
        self.dt = 1/self.RATE_HZ

        self.timer = self.create_timer(self.dt, self.control_loop) # 0.02 sek = 50 Hz

        self.get_logger().info('DroneController node started')

    # Callbacks
    def state_callback(self, msg):
        self.current_state = msg

    def local_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
        self.pos_0 = self.current_P  # This is used explicitly for velocity calculation
        self.current_P = np.array([self.current_x, self.current_y, self.current_z])

    def vicon_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
        self.pos_0 = self.current_P # This is used explicitly for velocity calculation
        self.current_P = np.array([self.current_x, self.current_y, self.current_z])

    def target_pos_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.target_P = np.array([self.target_x, self.target_y, self.target_z])

    
    def control_loop(self): # Control loop (50 Hz)
        
        if not self.current_state.guided or not self.current_state.armed: # Check if drone is ARMED and in correct flight mode

            return

        # Setting velocity using past position
        self.vel = (self.pos - self.pos_0) * self.RATE_HZ
        

        # ── XY: position + velocity → horizontal thrust vector ─────────
        thr_xy = self.xy_control.update(
            pos_sp = self.target_P[:2],
            pos    = self.current_P[:2],
            vel    = self.current_vel[:2],
            dt     = self.dt,
        )

        # ── Z: position + velocity → collective (vertical) thrust ───────
        thr_z = self.z_control.update(
            z_sp = self.target_P[2],
            z    = self.current_P[2],
            vz   = self.current_vel[2],
            dt   = self.dt,
        )

        thr_sp = np.array([thr_xy[0], thr_xy[1], thr_z])

        q_wxyz, thrust_body_z = thrust_to_attitude(thr_sp, self.yaw_sp)

        ########### Reguleringssystem ############
        #self.target_roll  = 0.0 ### In degrees ### PID based on position feedback from VICON or ...
        #self.target_pitch = 0.0 #### PID based on position feedback from VICON or ...
        #self.target_yaw   = 0.0 #### PID based on position feedback from VICON or ...
        self.target_thrust = thrust_body_z #### PID based on height over ground (and maby angle because the force vector no longer vertical). 0.0 to 1.0
        ### Lige her #########

        
        stamp = self.get_clock().now().to_msg()
        
        #r = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=True) # Euler degrees to quatanions -->
        #q = r.as_quat()  # [x, y, z, w]

        att_msg = PoseStamped()             # Create and send the orientation 
        att_msg.header.stamp = stamp
        att_msg.header.frame_id = 'map'
        att_msg.pose.orientation.w = q_wxyz[0]
        att_msg.pose.orientation.x = q_wxyz[1]
        att_msg.pose.orientation.y = q_wxyz[2]
        att_msg.pose.orientation.z = q_wxyz[3]

        thr_msg = Thrust()                  # Create and send the thrust
        thr_msg.header.stamp = stamp
        thr_msg.thrust = self.target_thrust

        self.att_pub.publish(att_msg)       # Send the orientation
        self.thr_pub.publish(thr_msg)       # Send the thrust

    # Service helpers
    def arm(self): # yes
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        self.get_logger().info('Arming requested')
        return future

    def set_mode(self, mode: str): # Not used, and should probably not be used. Changing state should probably exclusively be changed on the RC.
        req = SetMode.Request()
        req.custom_mode = mode  
        future = self.set_mode_client.call_async(req)
        self.get_logger().info(f'Mode change to {mode} requested')
        return future


def main(args=None):
    rclpy.init(args=args)
    node = DroneController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()