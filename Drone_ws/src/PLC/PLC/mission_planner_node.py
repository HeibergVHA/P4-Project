import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # 'Pose' data type.
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
            
        self.target_pos_pub = self.create_publisher(         # Target position topic.
            PoseStamped, 'target_pos', 10)
        

        # Current postition
        self.current_x = 0.0            # Meters
        self.current_y = 0.0
        self.current_z = 0.0

        # Target position
        self.target_x = 0.0             # Meters
        self.target_y = 0.0
        self.target_z = 0.0

        self.timer = self.create_timer(0.02, self.control_loop) # 0.02 sek = 50 Hz

        self.get_logger().info('vicon node started')


    def vicon_pos_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
        
    
    def control_loop(self): # Control loop (50 Hz)
        
        stamp = self.get_clock().now().to_msg()
        
        ##################### DETERMINE TARGET POSITION HERE ##########################

        r = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=True) # Euler degrees to quatanions -->
        q = r.as_quat()  # [x, y, z, w]

        att_msg = PoseStamped()             # Create and send the orientation 
        att_msg.header.stamp = stamp
        att_msg.header.frame_id = 'map'
        att_msg.pose.orientation.x = q[0]
        att_msg.pose.orientation.y = q[1]
        att_msg.pose.orientation.z = q[2]
        att_msg.pose.orientation.w = q[3]

        self.target_pos_pub.publish(att_msg)       # Send the orientation


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()