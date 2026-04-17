import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu
from livox_interfaces.msg import CustomMsg
from std_srvs.srv import Trigger
import datetime
import rosbag2_py

class LivoxBagRecorder(Node):
    def __init__(self):
        super().__init__('livox_bag_recorder')


        self.writer = None
        self.recording = False
        self.bag_name = None

        self.start_srv = self.create_service(
            Trigger,
            'start_recording',
            self.start_callback
        )

        self.stop_srv = self.create_service(
            Trigger,
            'stop_recording',
            self.stop_callback
        )

        self.lidar_sub = self.create_subscription(
            CustomMsg,
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
    def start_callback(self, request, response):
        if self.recording:
            response.success = False
            response.message = 'Already recording'
            return response
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.bag_name = f'/ros2_ws/bags/scan_{timestamp}'

        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_name,
            storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        # Register recording topics
        lidar_info = rosbag2_py.TopicMetadata(
            name='/livox/lidar',
            type='livox_interfaces/msg/CustomMsg',
            serialization_format='cdr')
        self.writer.create_topic(lidar_info)
    
        imu_topic = rosbag2_py.TopicMetadata(
            name='/livox/imu',
            type='std_msgs/msg/Imu',
            serialization_format='cdr')
        self.writer.create_topic(imu_topic)

        self.recording = True
        self.get_logger().info(f'Recording started: {self.bag_name}')
        response.success = True
        response.message = f'Recording to {self.bag_name}'
        return response

    def stop_callback(self, request, response):
        if not self.recording:
            response.success = False
            response.message = 'Not recording'
            return response
        self.writer = None
        self.recording = False
        self.get_logger().info(f'Recording stopped: {self.bag_name}')
        response.success = True
        response.message = f'Saved to {self.bag_name}'

    def lidar_callback(self, msg):
        if self.recording and self.writer:
            self.writer.write(
                '/livox/lidar',
                serialize_message(msg),
                self.get_clock().now().nanoseconds)

    def imu_callback(self, msg):
        if self.recording and self.writer:
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