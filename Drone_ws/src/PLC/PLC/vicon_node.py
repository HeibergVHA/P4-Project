import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped # 'Pose' data type
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.


class Vicon(Node):
    def __init__(self):
        super().__init__('vicon')
        
        self.vicon_pos_pub = self.create_publisher(      # Vicon topic.
            PoseStamped, 'vicon_pose', 10)

        # Current postition
        self.current_x = 0.0            # Meters
        self.current_y = 0.0
        self.current_z = 0.0

        # Orientation (attitude) setpoints
        self.target_roll  = 0.0         # Degrees
        self.target_pitch = 0.0
        self.target_yaw   = 0.0

        self.timer = self.create_timer(0.02, self.control_loop) # 0.02 sek = 50 Hz

        self.get_logger().info('vicon node started')

    
    def control_loop(self): # Control loop (50 Hz)
        
        stamp = self.get_clock().now().to_msg()
        
        ##################### GET DATA FROM VICON HERE ##########################

        r = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=True) # Euler degrees to quatanions -->
        q = r.as_quat()  # [x, y, z, w]

        msg = PoseStamped()             # Create and send the orientation 
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.current_x
        msg.pose.position.y = self.current_y
        msg.pose.position.z = self.current_z
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.vicon_pos_pub.publish(msg)       # Send the orientation


def main(args=None):
    rclpy.init(args=args)
    node = Vicon()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()