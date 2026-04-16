import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # 'Pose' data type.
from mavros_msgs.msg import State, Thrust
from mavros_msgs.srv import CommandBool, SetMode
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.

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

        # State
        self.current_state = State()        # Stores if the drone is connected, armed, and state.

        # Current postition
        self.current_x = 0.0            # Meters
        self.current_y = 0.0
        self.current_z = 0.0

        # Target position
        self.target_x = 0.0             # Meters
        self.target_y = 0.0
        self.target_z = 0.0

        # Orientation (attitude) setpoints
        self.target_roll  = 0.0         # Degrees
        self.target_pitch = 0.0
        self.target_yaw   = 0.0
        self.target_thrust = 0.0        # Thrust value 0.0-1.0

        self.timer = self.create_timer(0.02, self.control_loop) # 0.02 sek = 50 Hz

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

    def target_pos_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z

    
    def control_loop(self): # Control loop (50 Hz)
        
        if not self.current_state.guided or not self.current_state.armed: # Check if drone is ARMED and in correct flight mode

            return

        ########### Reguleringssystem ############
        self.target_roll  = 0.0 ### In degrees ### PID based on position feedback from VICON or ...
        self.target_pitch = 0.0 #### PID based on position feedback from VICON or ...
        self.target_yaw   = 0.0 #### PID based on position feedback from VICON or ...
        self.target_thrust = 0.05 #### PID based on height over ground (and maby angle because the force vector no longer vertical). 0.0 to 1.0
        ### Lige her #########

        
        stamp = self.get_clock().now().to_msg()
        
        r = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=True) # Euler degrees to quatanions -->
        q = r.as_quat()  # [x, y, z, w]

        att_msg = PoseStamped()             # Create and send the orientation 
        att_msg.header.stamp = stamp
        att_msg.header.frame_id = 'map'
        att_msg.pose.orientation.x = q[0]
        att_msg.pose.orientation.y = q[1]
        att_msg.pose.orientation.z = q[2]
        att_msg.pose.orientation.w = q[3]

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