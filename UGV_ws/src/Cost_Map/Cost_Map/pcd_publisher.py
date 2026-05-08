# publish_pcd.py
import sys, time
import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

class PCDPublisher(Node):
    def __init__(self, path):
        super().__init__('pcd_publisher')

        self.pub = self.create_publisher(PointCloud2, '/scene_cloud', 10)

        # Declare parameter with default
        self.declare_parameter('pcd_file_path', '')

        # Read parameter
        pcd_path = self.get_parameter(
            'pcd_file_path'
        ).get_parameter_value().string_value

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points, dtype=np.float32)

        msg = PointCloud2()
        msg.header = Header(frame_id='map')
        msg.height, msg.width = 1, len(pts)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian, msg.point_step = False, 12
        msg.row_step = 12 * len(pts)
        msg.data = pts.tobytes()
        msg.is_dense = True
        self.msg = msg

        self.create_timer(0.1, self.publish)  # 10 Hz
        self.get_logger().info(f'Publishing {len(pts)} points from {path}')

    def publish(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)

def main():
    rclpy.init()
    rclpy.spin(PCDPublisher(sys.argv[1]))

if __name__ == '__main__':
    main()