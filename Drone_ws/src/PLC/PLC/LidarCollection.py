import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import datetime
import subprocess
import time
import os 
import signal
from control_interfaces.srv import SendBag

class LivoxBagRecorder(Node):
    def __init__(self):
        super().__init__('livox_bag_recorder')
        self._pending_send = False
        self.bag_process = None
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

        self.create_timer(1.0, self._check_send)

        self.send_client = self.create_client(SendBag, '/send')

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
        
        self.bag_process = subprocess.Popen(
            ['ros2', 'bag', 'record',
             '-o', self.bag_name,
             '/livox/lidar',
             '/livox/imu',
             '/mavros/global_position/global']
        )

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
        
        if self.bag_process:
            self.bag_process.terminate()
            self.bag_process.wait()
            self.bag_process = None

        self.recording = False
        self.get_logger().info(f'Recording stopped: {self.bag_name}')

        self.stop_livox_driver()
        self._pending_send = True

        response.success = True
        response.message = f'{self.bag_name}'
        return response
    
    def trigger_send(self):
        request = SendBag.Request()
        request.bag_path = self.bag_name
        future = self.send_client.call_async(request)
        future.add_done_callback(self.send_response_cb)
    
    def send_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info('send triggered successfully')
        else:
            self.get_logger().error(f'Client couldnt connect, Trying again in 5 sec: {result.message}')
            self.create_timer(5.0, self._retry_send_once)
    
    def _retry_send_once(self):
        self.destroy_timer(self._timers[-1])
        self.trigger_send()

    
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