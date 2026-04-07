import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Imu

import rosbag2_py

class LivoxBagRecorder(Node):
    def __init__(self):
        super().__init__('livox_bag_recorder')
        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py.StorageOptions(
            uri='my_bag',
            storage_id='sqlite3')
        
        converter_options = rosbag2_py.ConverterOptions('', '')

        self.writer.open(storage_options, converter_options)

        lidar_info = rosbag2_py.TopicMetadata(
            name='/livox/lidar',
            type='std_msgs/msg/PointCloud2',
            serialization_format='cdr')
        self.writer.create_topic(lidar_info)

        imu_topic = rosbag2_py.TopicMetadata(
            name='/livox/imu',
            type='std_msgs/msg/Imu',
            serialization_format='cdr')
        self.writer.create_topic(imu_topic)

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/livox/imu',
            self.imu_callback,
            10
        )

    def lidar_callback(self, msg):
        self.writer.write(
            '/livox/lidar',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)

    def imu_callback(self, msg):
        self.writer.write(
            '/livox/imu',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)


def main(args=None):
    rclpy.init(args=args)
    lbr = LivoxBagRecorder()
    rclpy.spin(lbr)
    rclpy.shutdown()


if __name__ == '__main__':
    main()