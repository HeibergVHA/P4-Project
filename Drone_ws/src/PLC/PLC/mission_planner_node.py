import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # 'Pose' data type.
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.

################ "ros2 run pakage node --ros-args -p input_source:=vicon" #############

class MissionPlanner(Node):
    def __init__(self):
        super().__init__('mission_planner')
        
        self.declare_parameter('input_source', 'vicon')         # Input parameter when node is started. "local" to get current position from the pixhawk
        source = self.get_parameter('input_source').value       # "vicon" to get current position from "vicon_posee" topic.
        self.get_logger().info(f"Using input source: {source}") # I think like this: "ros2 run pakage node --ros-args -p input_source:=vicon"

        if source == 'local':
            self.local_pos_sub = self.create_subscription(      # Local position. ArduPilot EKF position.
                PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, 10)
            
        if source == 'vicon':
            self.vicon_pos_sub = self.create_subscription(      # Vicon topic.
                PoseStamped, 'vicon_pose', self.vicon_pos_callback, 10)

        self.create_subscription(String, 'uav/radio_in/target_waypoint', self.target_waypoint_callback, 10)
            
        self.target_pos_pub = self.create_publisher(         # Target position topic.
            PoseStamped, 'uav/target_pos', 10)
        

        # Current postition
        self.current_x = 0.0            # Meters
        self.current_y = 0.0
        self.current_z = 0.0

        # Target position
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_qw = 0.0
        self.target_qx = 0.0
        self.target_qy = 0.0
        self.target_qz = 0.0

        self.timer = self.create_timer(0.02, self.control_loop) # 0.02 sek = 50 Hz

        self.get_logger().info('vicon node started')

    def local_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

    def vicon_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
    
    def target_waypoint_callback(self, msg):
        x = 123
        # Waypoints.append(msg) # Or something like that.
    
    def control_loop(self): # Control loop at or around 50 Hz. Does not strictly need to be 50 Hz, just need a new position before the old is reached.
        
        stamp = self.get_clock().now().to_msg()
        
        ##################### DETERMINE TARGET POSITION HERE ##########################
        #### Mby bezier curve or b-spline based on waypoints.
        ## Then pure-persuit algorithm
        # self.target_x = 0.0
        # self.target_y = 0.0
        # self.target_z = 0.0

        msg = PoseStamped()             # Create and send 
        msg.header.stamp = stamp
        msg.header.frame_id = 'targetPos'
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        msg.pose.orientation.w = self.target_qw
        msg.pose.orientation.x = self.target_qx
        msg.pose.orientation.y = self.target_qy
        msg.pose.orientation.z = self.target_qz

        self.target_pos_pub.publish(msg)       # Send


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()