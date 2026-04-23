import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from livox_interfaces.msg import CustomMsg
from std_srvs.srv import Trigger
import datetime
import rosbag2_py
from rclpy.serialization import deserialize_message
import os
import subprocess
import time

class LivoxBagReader(Node):
    def __init__(self):
        super().__init__('livox_bag_reader')
        self.bag_process = None
        self.fastlio_process = None

        self.start_srv = self.create_service(
            Trigger,
            '/start_reading',
            self.start_callback
        )

        self.stop_srv = self.create_service(
            Trigger,
            '/stop_reading',
            self.stop_callback
        )


    def get_latest_bag(self):
        bags_dir = "/ros2_ws/bags"

        # Get all subdirectories
        subdirs = [
            os.path.join(bags_dir, d)
            for d in os.listdir(bags_dir)
            if os.path.isdir(os.path.join(bags_dir, d))
        ]
        if not subdirs:
            raise RuntimeError(f'No bag folders found')
        self.get_logger().info(f'founder rosbag {max(subdirs, key=os.path.getmtime)}')
        return max(subdirs, key=os.path.getmtime)
    
    def start_callback(self, request, response):
        if self.bag_process:
            response.success = False
            response.message = 'Already Recording..'
            return response

        try:
            bag_name = self.get_latest_bag()
        except RuntimeError as e:
            response.success = False
            response.message = str(e)
            return response
        
        # Launch FastLio mapping
        self.fastlio_process = subprocess.Popen(
            ['ros2', 'launch', 'fast_lio', 'mapping.launch.py', 'config_file:=avia.yaml'],
        )
        self.get_logger().info('Launched FAST-LIO mapping.launch.py')

        # Following starts the reading loop after delay.
        self.bag_name = bag_name
        self.read_timer = self.create_timer(5.0, self._start_bag_play)

        response.success = True
        response.message = f'Reading {self.bag_name}'
        return response

    def _start_bag_play(self):
        self.read_timer.cancel()

        self.get_logger().info(f'playing: {self.bag_name}')

        self.bag_process = subprocess.Popen(
            ['ros2', 'bag', 'play', self.bag_name, '--clock'],
        )

        self.poll_timer = self.create_timer(1.0, self._check_bag_finished)

    def _check_bag_finished(self):
        if self.bag_process and self.bag_process.poll() is not None:
            self.poll_timer.cancel()
            self.get_logger().info('Bag play finished')
            self._shutdown_processes()
    
    def _shutdown_processes(self):
        if self.bag_process:
            self.bag_process.terminate()
            self.bag_process = None
            self.get_logger().info('Terminated bag play process')
        if self.fastlio_process:
            self.fastlio_process.terminate()
            self.fastlio_process = None
            self.get_logger().info('Terminated FAST-LIO process')
    
    def stop_callback(self, request, response):
        if not self.bag_process:
            response.success = False
            response.message = 'Not reading..'
            return response
        self._shutdown_processes()

        self.get_logger().info(f'Reading stopped: {self.bag_name}')
        response.success = True
        response.message = f'Stopped reading {self.bag_name}'
        return response
        
def main(args=None):
    rclpy.init(args=args)
    node = LivoxBagReader()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':    
    main()