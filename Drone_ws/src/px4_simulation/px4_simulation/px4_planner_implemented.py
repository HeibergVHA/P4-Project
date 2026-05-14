"""
px4_position_controller.py
==========================
Full 3-axis position controller for PX4 X500 in ROS2 offboard mode.

XY controller  → attitude quaternion  (replicates PX4 ControlMath::thrustToAttitude /
                                        bodyzToAttitude exactly)
Z  controller  → thrust_body[2]        (replicates PX4 _velocityController D-axis logic)

PX4 source references
---------------------
  mc_pos_control/PositionControl/PositionControl.cpp  – _positionController(), _velocityController()
  mc_pos_control/PositionControl/ControlMath.cpp      – thrustToAttitude(), bodyzToAttitude()

NED convention throughout:
  +x  = North,  +y = East,  +z = Down
  Positive roll  = right side down  (+y acceleration)
  Positive pitch = nose down        (−x acceleration ... wait, NED: nose-down = +x)
  thrust_body[2] < 0                (upward in body frame)
"""

import rclpy
from rclpy.node import Node
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleCommand,
    VehicleAttitudeSetpoint,
    VehicleOdometry,
)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation


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
        Kp_vel: float = 1.8, # 1.2
        Ki_vel: float = 0.4,
        Kd_vel: float = 0.2,
        hover_thrust: float = 0.73,
        max_vel_xy: float = 5.0,       # m/s – MPC_XY_VEL_MAX
        max_tilt: float = np.radians(45.0),  # rad – MPC_TILTMAX_AIR
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
        thr_xy_raw = acc_sp / 9.81 # * self.hover_thrust
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
        thr_xy = acc_sp / 9.81 # * self.hover_thrust

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
        Kp_vel: float  = 1.5, # 4.0,
        Ki_vel: float  = 1.0, # 2.0,
        Kd_vel: float  = 0.0, # 0.0,
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
        self.past_vel = 0.0

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

'''class ZController:
    """
    Non-cascaded PID controller: z_error → collective thrust (NED, D-axis).
    Uses only measured z position — no velocity measurement required.
    D term is computed as a filtered finite difference of position error.
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 3.0,
        hover_thrust: float = 0.73,
        thr_min: float = 0.0, # 0.12,
        thr_max: float = 1.0, # 0.90,
        tau_d: float = 0.10,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.hover_thrust = hover_thrust
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.tau_d = tau_d

        self._int = 0.0
        self._deriv_filt = 0.0
        self._err_prev = None       # None until first call

    def reset(self):
        self._int = 0.0
        self._deriv_filt = 0.0
        self._err_prev = None

    def update(self, z_sp: float, z: float, dt: float) -> float:
        """
        Returns thrust_body[2]: always negative (upward),
        clamped to [-thr_max, -thr_min].

        z_sp, z  in NED metres  (negative = above ground)
        """
        if dt <= 0.0:
            return -self.hover_thrust

        # ── 1. Position error ──────────────────────────────────────────
        error = z_sp - z

        # ── 2. Filtered finite-difference derivative ───────────────────
        if self._err_prev is None:
            raw_deriv = 0.0
        else:
            raw_deriv = (error - self._err_prev) / dt

        alpha = self.tau_d / (self.tau_d + dt)
        self._deriv_filt = alpha * self._deriv_filt + (1.0 - alpha) * raw_deriv
        self._err_prev = error

        # ── 3. PID law ─────────────────────────────────────────────────
        thrust = (
            self.Kp * error
            + self.Ki * self._int
            + self.Kd * self._deriv_filt
            - self.hover_thrust
        )

        # ── 4. Saturate ────────────────────────────────────────────────
        u_min = -self.thr_max
        u_max = -self.thr_min
        thrust = float(np.clip(thrust, u_min, u_max))

        # ── 5. Anti-windup ─────────────────────────────────────────────
        stop = (thrust >= u_max and error >= 0.0) or \
               (thrust <= u_min and error <= 0.0)
        if not stop:
            self._int += error * dt
            self._int = float(np.clip(self._int, -self.thr_max, self.thr_max))

        return thrust'''

'''class ZController:
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

        return thrust_d'''

class PositionOnlyPDController:
    """
    Single-loop PD controller: x_ref → phi_cmd
    using only measured position (no velocity measurement, no IMU).

    Transfer function:
        phi_cmd(s)     -1     Kp·(tau·s + 1) + Kd·s
        ────────── = ─────  · ─────────────────────────
        X_ref(s)      g            s·(tau·s + 1)

    The derivative term is a filtered finite difference of position error,
    equivalent to Kd·s / (tau·s + 1) acting on the error signal.
    """

    def __init__(
        self,
        Kp: float = 0.6, # 0.6
        Kd: float = 1.2,
        tau: float = 0.15,      # LP filter on derivative — tune this up if noisy
        g: float = 9.81,
        phi_max: float = np.radians(45.0),
    ):
        self.Kp = Kp
        self.Kd = Kd
        self.tau = tau
        self.g = g
        self.phi_max = phi_max

        self._err_prev = 0.0        # previous position error
        self._deriv_filt = 0.0      # filtered derivative state

    def reset(self):
        self._err_prev = 0.0
        self._deriv_filt = 0.0

    def update(self, x_ref: float, x_meas: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # ── 1. Position error ──────────────────────────────────────────
        error = x_ref - x_meas

        # ── 2. Filtered derivative of error (= filtered -velocity) ─────
        # Discrete first-order LP applied to finite difference:
        #   D(s) = Kd · s / (tau·s + 1)
        raw_deriv = (error - self._err_prev) / dt
        alpha = self.tau / (self.tau + dt)
        self._deriv_filt = alpha * self._deriv_filt + (1.0 - alpha) * raw_deriv
        self._err_prev = error

        # ── 3. PD law → phi_cmd ────────────────────────────────────────
        phi_cmd = -(1.0 / self.g) * (self.Kp * error + self.Kd * self._deriv_filt)

        return float(np.clip(phi_cmd, -self.phi_max, self.phi_max))

# ══════════════════════════════════════════════════════════════════════════════
# ROS 2 Node
# ══════════════════════════════════════════════════════════════════════════════

class PositionControllerNode(Node):

    def __init__(self):
        super().__init__('px4_position_controller')

        # ── Setpoints ──────────────────────────────────────────────────
        self.pos_sp = np.array([1.0, 0.0, -2.5])   # NED [m]: x=2, y=1, alt=2m
        self.yaw_sp = 0.0                            # rad

        # ── Control rate ───────────────────────────────────────────────
        self.RATE_HZ = 100
        self.dt      = 1.0 / self.RATE_HZ

        # ── Controllers ────────────────────────────────────────────────
        #self.xy_ctrl = XYController(hover_thrust=0.73)
        self.z_ctrl  = ZController (hover_thrust=0.73)
        self.x_ctrl = PositionOnlyPDController()
        self.y_ctrl = PositionOnlyPDController()

        # ── Odometry state ─────────────────────────────────────────────
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

        # ── QoS ───────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_cb,
            qos,
        )

        # ── Publishers ─────────────────────────────────────────────────
        self.att_pub      = self.create_publisher(VehicleAttitudeSetpoint,
                                '/fmu/in/vehicle_attitude_setpoint_v1', qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode,
                                '/fmu/in/offboard_control_mode', qos)
        self.cmd_pub      = self.create_publisher(VehicleCommand,
                                '/fmu/in/vehicle_command', qos)

        self.counter = 0
        self.create_timer(self.dt, self.timer_cb)
        self.get_logger().info(
            f'Position controller started. Target: {self.pos_sp}  yaw: {np.degrees(self.yaw_sp):.1f}°'
        )

    # ── Callbacks ──────────────────────────────────────────────────────

    def odometry_cb(self, msg: VehicleOdometry):
        self.pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.past_vel = self.vel
        self.vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

    def timer_cb(self):
        self._publish_offboard_mode()

        if self.counter == 10:
            self._switch_offboard()
            self._arm()

        if self.counter >= 10:
            self._publish_attitude_setpoint()

        self.counter += 1

    # ── Core publish ───────────────────────────────────────────────────

    def _publish_attitude_setpoint(self):
        #self.pos_sp = np.array([float(self.counter/self.RATE_HZ), 0.0, -2.5])
        if self.counter % 2000 < 1000:
            self.pos_sp = np.array([1.0, 0.0, -2.5])
        else:
            self.pos_sp = np.array([0.0, 0.0, -2.5])

        # ── XY: position + velocity → horizontal thrust vector ─────────
        """thr_xy = self.xy_ctrl.update(
            pos_sp = self.pos_sp[:2],
            pos    = self.pos[:2],
            vel    = self.vel[:2],
            dt     = self.dt,
        )"""

        # After (PD on position only):
        phi_x = self.x_ctrl.update(self.pos_sp[0], self.pos[0], self.dt)
        phi_y = self.y_ctrl.update(self.pos_sp[1], self.pos[1], self.dt)

        # ── Z: position + velocity → collective (vertical) thrust ───────
        thr_z = self.z_ctrl.update(
            z_sp = self.pos_sp[2],
            z    = self.pos[2],
            vz   = self.vel[2],
            dt   = self.dt,
        )

        # Construct a thrust vector from the direct phi/theta commands
        # phi_x tilts in x (pitch in NED), phi_y tilts in y (roll in NED)
        thr_x = np.sin(phi_x) * thr_z   # horizontal x thrust component
        thr_y = np.sin(phi_y) * thr_z  # horizontal y thrust component
        #thr_sp = np.array([thr_xy[0], thr_xy[1], thr_z])
        thr_sp = np.array([thr_x, thr_y, thr_z])
        q_wxyz, thrust_body_z = thrust_to_attitude(thr_sp, self.yaw_sp)

        # ── Combine into 3-D thrust vector (NED) ──────────────────────
        # PX4 prioritises Z: vertical thrust is set first, then
        # horizontal is added. Total thrust vector in NED world frame:
        #   thr_sp = [tx, ty, thr_z]
        # Note: thr_z is already negative (upward).
        #thr_sp = np.array([thr_xy[0], thr_xy[1], thr_z])

        # ── Convert thrust vector → attitude quaternion + scalar thrust ─
        # This is the exact Python translation of:
        #   ControlMath::thrustToAttitude(thr_sp, yaw_sp, att_sp)
        #q_wxyz, thrust_body_z = thrust_to_attitude(thr_sp, self.yaw_sp)

        # ── Log at 2 Hz ────────────────────────────────────────────────
        if True:
            bige = self.pos_sp-np.array(self.pos)
            # Recover Euler for logging only
            roll, pitch, yaw = self._q_to_euler(q_wxyz)
            self.get_logger().info(
                f'pos=[{self.pos[0]:.2f},{self.pos[1]:.2f},{self.pos[2]:.2f}]  '
                f'sp=[{self.pos_sp[0]:.1f},{self.pos_sp[1]:.1f},{self.pos_sp[2]:.1f}]  '
                f'roll={np.degrees(roll):.1f}°  pitch={np.degrees(pitch):.1f}°  '
                f'thr={thrust_body_z:.3f}\tERROR={bige}'
            )

        # ── Publish ────────────────────────────────────────────────────
        msg = VehicleAttitudeSetpoint()
        msg.q_d[0] = float(q_wxyz[0])   # w
        msg.q_d[1] = float(q_wxyz[1])   # x
        msg.q_d[2] = float(q_wxyz[2])   # y
        msg.q_d[3] = float(q_wxyz[3])   # z
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = float(thrust_body_z)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.att_pub.publish(msg)

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.attitude     = True
        msg.body_rate    = False
        msg.position     = False
        msg.velocity     = False
        msg.acceleration = False
        msg.timestamp    = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _arm(self):
        msg = VehicleCommand()
        msg.command          = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1           = 1.0
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)
        self.get_logger().info('Arm command sent')

    def _switch_offboard(self):
        msg = VehicleCommand()
        msg.command          = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1           = 1.0
        msg.param2           = 6.0
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)
        self.get_logger().info('Offboard mode command sent')

    @staticmethod
    def _q_to_euler(q: np.ndarray):
        """[w,x,y,z] → (roll, pitch, yaw) ZYX — for logging only."""
        w, x, y, z = q
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw


# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = PositionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()