import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu
from livox_interfaces.msg import CustomMsg
from std_srvs.srv import Trigger
import datetime
import rosbag2_py
import subprocess
import time
import os 
import signal

class LivoxBagRecorder(Node):
    def __init__(self):
        super().__init__('livox_bag_recorder')
        self.writer = None
        self.recording = False
        self.bag_name = None
        self.livox_process = None

        self.start_srv = self.create_service(
            Trigger,
            '/start_recording',
            self.start_callback
        )

        self.stop_srv = self.create_service(
            Trigger,
            '/stop_recording',
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
        self._pending_send = False
        self.create_timer(1.0, self._check_send)

        self.send_client = self.create_client(Trigger, '/send')

    def start_livox_driver(self):
        self.get_logger().info('Starting Livox driver...')
        self.livox_process = subprocess.Popen(
            ['ros2','launch','livox_ros2_driver','livox_lidar_msg_launch.py'],
            preexec_fn=os.setsid # Start the process in a new session to allow killing the entire process group later
        )
        time.sleep(3.0) # Wait for driver startup
        self.get_logger().info('Livox driver started.')
    
    def stop_livox_driver(self):
        if self.livox_process:
            self.get_logger().info('Stopping Livox driver...')
            os.killpg(os.getpgid(self.livox_process.pid), signal.SIGTERM) # Kill the entire process group
            self.livox_process = None
            self.get_logger().info('Livox driver stopped.')


    def start_callback(self, request, response):
        if self.recording:
            response.success = False
            response.message = 'Already recording'
            return response
        
        self.start_livox_driver()

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
            type='sensor_msgs/msg/Imu',
            serialization_format='cdr')
        self.writer.create_topic(imu_topic)

        self.recording = True
        self.get_logger().info(f'Recording started: {self.bag_name}')
        response.success = True
        response.message = f'{self.bag_name}'
        return response

    def stop_callback(self, request, response):
        if not self.recording:
            response.success = False
            response.message = 'Not recording'
            return response
        self.writer = None
        self.recording = False
        self.get_logger().info(f'Recording stopped: {self.bag_name}')
        self._pending_send = True
        self.stop_livox_driver()
        response.success = True
        response.message = f'{self.bag_name}'
        return response
    
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
            
    def trigger_send(self):
        request = Trigger.Request()
        future = self.send_client.call_async(request)
        future.add_done_callback(self.send_response_cb)
    
    def send_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info('send triggered successfully')
        else:
            self.get_logger().error(f'Failed to trigger send: {result.message}')
    
    def _check_send(self):
        if self._pending_send:
            self._pending_send = False
            self.trigger_send()


def main(args=None):
    rclpy.init(args=args)
    lbr = LivoxBagRecorder()
    rclpy.spin(lbr)
    rclpy.shutdown()


if __name__ == '__main__': 
    main() 