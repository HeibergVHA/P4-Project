import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped # 'Pose' data type.
from std_msgs.msg import String, Header
from mavros_msgs.msg import State, Thrust
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import AttitudeTarget
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.
import numpy as np

# try:
#     from px4_msgs.msg import VehicleOdometry, VehicleAttitudeSetpoint, \
#     OffboardControlMode, VehicleCommand
#     PX4_AVAILABLE = True
# except ImportError:
#     PX4_AVAILABLE = False
# ══════════════════════════════════════════════════════════════════════════════
# Maths helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])

class PID:
    def __init__(
        self,
        kp, ki, kd, dt,
        output_limit=None,
        thr_min=None,
        thr_max=None,
        integral_limit=None,
        hover_thrust=None,
        filter_derivative=False,
        tau=0.15,
    ):
        self.kp              = kp
        self.ki              = ki
        self.kd              = kd
        self.dt              = dt
        self.output_limit    = output_limit
        self.thr_min         = thr_min
        self.thr_max         = thr_max
        self.integral_limit  = integral_limit
        self.hover_thrust    = hover_thrust
        self.filter_derivative = filter_derivative
        self.tau             = tau

        self._integral   = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0
        self._first_run  = True

    def update(self, setpoint, measurement):
        error = setpoint - measurement

        # Proportional
        p = self.kp * error

        # Integral + clamp anti-windup
        self._integral += error * self.dt
        if self.integral_limit is not None:
            self._integral = float(np.clip(self._integral, -self.integral_limit, self.integral_limit))
        i = self.ki * self._integral

        # Derivative
        if self._first_run:
            d = 0.0
            self._first_run = False
        else:
            raw_deriv = (error - self._prev_error) / self.dt
            if self.filter_derivative:
                alpha = self.tau / (self.tau + self.dt)
                self._deriv_filt = alpha * self._deriv_filt + (1.0 - alpha) * raw_deriv
                d = self.kd * self._deriv_filt
            else:
                d = self.kd * raw_deriv
        self._prev_error = error

        output = p + i + d

        # Hover feedforward
        if self.hover_thrust is not None:
            output -= self.hover_thrust

        # Asymmetric thrust clamp (Z / NED)
        if self.thr_min is not None and self.thr_max is not None:
            output = float(np.clip(output, -self.thr_max, -self.thr_min))

        # Symmetric clamp (X / Y angle commands)
        elif self.output_limit is not None:
            output = float(np.clip(output, -self.output_limit, self.output_limit))

        return output

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0
        self._first_run  = True


################ "ros2 run pakage node --ros-args -p input_source:=vicon" #############

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        self.declare_parameter('input_source', 'local')         # Input parameter when node is started. "local" to get current position from the pixhawk
        self.source = self.get_parameter('input_source').value       # "vicon" to get current position from "vicon_posee" topic.
        self.get_logger().info(f"Using input source: {self.source}") # I think like this: "ros2 run pakage node --ros-args -p input_source:=vicon"

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        px4_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 5
        )

        # Timing
        self.RATE_HZ = 50
        self.dt = 1/self.RATE_HZ
        self.counter = 0

        # Controllers
        #self.xy_controller = XYController(hover_thrust=0.73)
        self.x_controller = PID(kp=0.00262, ki=0.000015, kd=0.1, dt=self.dt, output_limit=np.radians(10.0))
        self.y_controller = PID(kp=0.00262, ki=0.000015, kd=0.1, dt=self.dt, output_limit=np.radians(10.0))
        self.z_controller  = PID(kp=0.1, ki=0.05, kd=0.2, dt=self.dt, thr_min=0.3, thr_max=1.0,hover_thrust=0.76, integral_limit = 0.9)


        # self.x_controller = PID(kp=0.00302, ki=0.000020, kd=0.098224, dt=self.dt, output_limit=np.radians(10.0))
        # self.y_controller = PID(kp=0.00302, ki=0.000020, kd=0.098224, dt=self.dt, output_limit=np.radians(10.0))
        # self.z_controller  = PID(kp=0.1, ki=0.05, kd=0.2, dt=self.dt, thr_min=0.3, thr_max=1.0,hover_thrust=0.76, integral_limit = 0.9)

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

        # Mavros subscribers / publishes
        if self.source in ('local', 'vicon'):
            self.state_sub = self.create_subscription(
                State, '/mavros/state', self.state_callback, 10
            )
            if self.source == 'local':
                self.get_logger().info('Subscribing to gps/ekf')
                self.local_pos_sub = self.create_subscription(      # Local position. ArduPilot EKF position.
                    PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos)
                
            if self.source == 'vicon':
                self.get_logger().info('Subscribing to vicon_pose')
                self.vicon_pos_sub = self.create_subscription(      # Vicon topic.
                PoseStamped, 'vicon_pose', self.vicon_pos_callback, 10)
                
            # Publishers 
            self.att_thr_pub = self.create_publisher(
                AttitudeTarget, '/mavros/setpoint_raw/attitude', 10)

            # Service clients
            self.arming_client = self.create_client(                # To arm the drone...
                CommandBool, '/mavros/cmd/arming')

            self.set_mode_client = self.create_client(              # To set mode...
                SetMode, '/mavros/set_mode')
        # elif self.source == 'px4':
        #     if not PX4_AVAILABLE:
        #         self.get_logger().error('PX4 messages not available. Please install px4_msgs package.')
        #         raise RuntimeError('PX4 messages not available')
        #     self.get_logger().info('Using PX4 topics and services')

        #     self.create_subscription(
        #         VehicleOdometry, '/fmu/out/vehicle_odometry', self.px4_odometry_callback, px4_qos)
            
        #     self.px4_att_pub = self.create_publisher(
        #         VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', px4_qos)
            
        #     self.px4_offboard_pub = self.create_publisher(
        #         OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos
        #     )
        #     self.px4_cmd_pub = self.create_publisher(
        #         VehicleCommand, '/fmu/in/vehicle_command', px4_qos
        #     )
        
        # Shared subscribers

        self.target_pos_sub = self.create_subscription(         # Target position topic.
            PoseStamped, 'uav/target_pos', self.target_pos_callback, 10)
        
        self.create_subscription(String, 'uav/radio_in/mission_command', self.mission_command_callback, 10)




        self.timer = self.create_timer(self.dt, self.control_loop) # 0.02 sek = 50 Hz

        self.get_logger().info('DroneController node started')

    # Callbacks
    def state_callback(self, msg):
        self.current_state = msg

    def local_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

    def vicon_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
    
    def px4_odometry_callback(self, msg):
        self.current_x = msg.position[0]
        self.current_y = msg.position[1]
        self.current_z = msg.position[2]

    def target_pos_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.target_P = np.array([self.target_x, self.target_y, self.target_z])

    def mission_command_callback(self, msg):
        return
        command = msg.data
        if command == 'arm':
            self.arm(True)
        elif command == 'disarm':
            self.arm(False)
        elif command == 'guided':
            self.set_mode('GUIDED')
        elif command == 'angle':
            self.set_mode('ANGLE')
        elif command == 'stable':
            self.set_mode('STABLE')
        else:
            self.get_logger().warn(f'Unknown mission command received: {command}')
            return
        self.get_logger().info(f'Mission command received: {command}')

    def convert_euler_to_quaternion_px4(self):

        quat = R.from_euler('xyz', [self.phi_y, -self.phi_x, 0.0], degrees=False)
        x,y,z,w = quat.as_quat()
        return [w,x,y,z]
    
    def convert_euler_to_quaternion(self):
        # On mavros the roll and pitch are not swapped but on px4 they are. 
        # So we use 2 different functions to convert the euler angles to quaternions depending on the source.
        quat = R.from_euler('xyz', [-self.phi_y, self.phi_x, 0.0], degrees=False)
        x,y,z,w = quat.as_quat()
        return [w,x,y,z]

    def control_loop(self): 
        if self.source == 'px4':
            self._control_loop_px4()
        else:
            self._control_loop_mavros()


    def _control_loop_mavros(self):
        if not self.current_state.armed: # Check if drone is ARMED and in correct flight mode # not self.current_state.guided or 
            return
        
        self.phi_x = self.x_controller.update(self.target_x, self.current_x)
        self.phi_y = self.y_controller.update(self.target_y, self.current_y)
        self.thr_z = self.z_controller.update(-self.target_z, -self.current_z)
        
        q_wxyz = self.convert_euler_to_quaternion()

        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Bitmask: ignore body rates (0b00000111 = 7), use attitude + thrust
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE | \
                        AttitudeTarget.IGNORE_YAW_RATE

        msg.orientation.w = q_wxyz[0]
        msg.orientation.x = q_wxyz[1]
        msg.orientation.y = q_wxyz[2]
        msg.orientation.z = q_wxyz[3]

        msg.thrust = float(np.clip(-self.thr_z, 0.0, 1.0))  # Must be 0.0–1.0, positive

        self.att_thr_pub.publish(msg)
    
    # def _control_loop_px4(self):
    #            # Must stream offboard mode every tick or PX4 disengages
    #     self._publish_offboard_mode()

    #     # Startup sequence — arm and switch to offboard after 10 ticks
    #     if self.counter == 100:
    #         self._px4_switch_offboard()
    #         self._px4_arm()
    #     self.counter += 1

    #     if self.counter < 100:
    #         return

    #     self.phi_x = self.x_controller.update(self.target_x, self.current_x)
    #     self.phi_y = self.y_controller.update(self.target_y, self.current_y)
    #     self.thr_z = self.z_controller.update(self.target_z, self.current_z)

    #     q_wxyz = self.convert_euler_to_quaternion_px4()

    #     msg = VehicleAttitudeSetpoint()
    #     msg.q_d[0] = float(q_wxyz[0])
    #     msg.q_d[1] = float(q_wxyz[1])
    #     msg.q_d[2] = float(q_wxyz[2])
    #     msg.q_d[3] = float(q_wxyz[3])
    #     msg.thrust_body[0] = 0.0
    #     msg.thrust_body[1] = 0.0
    #     msg.thrust_body[2] = float(self.thr_z)   # already negative (NED upward)
    #     msg.timestamp = self.get_clock().now().nanoseconds // 1000
    #     self.px4_att_pub.publish(msg)

    # # Px4 Helpers
    # def _publish_offboard_mode(self):
    #     msg = OffboardControlMode()
    #     msg.attitude     = True
    #     msg.body_rate    = False
    #     msg.position     = False
    #     msg.velocity     = False
    #     msg.acceleration = False
    #     msg.timestamp    = self.get_clock().now().nanoseconds // 1000
    #     self.px4_offboard_pub.publish(msg)

    # def _px4_arm(self):
    #     msg = VehicleCommand()
    #     msg.command          = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    #     msg.param1           = 1.0
    #     msg.target_system    = 1
    #     msg.target_component = 1
    #     msg.source_system    = 1
    #     msg.source_component = 1
    #     msg.from_external    = True
    #     msg.timestamp        = self.get_clock().now().nanoseconds // 1000
    #     self.px4_cmd_pub.publish(msg)
    #     self.get_logger().info('PX4 arm command sent')

    # def _px4_switch_offboard(self):
        # msg = VehicleCommand()
        # msg.command          = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        # msg.param1           = 1.0
        # msg.param2           = 6.0
        # msg.target_system    = 1
        # msg.target_component = 1
        # msg.source_system    = 1
        # msg.source_component = 1
        # msg.from_external    = True
        # msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        # self.px4_cmd_pub.publish(msg)
        # self.get_logger().info('PX4 offboard mode command sent')



    # Mavros Helpers
    def arm(self, x):
        req = CommandBool.Request()
        req.value = x
        future = self.arming_client.call_async(req)
        self.get_logger().info('Arming requested')
        return future

    def set_mode(self, mode):
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