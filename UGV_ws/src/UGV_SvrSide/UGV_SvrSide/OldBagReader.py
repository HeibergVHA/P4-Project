# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Imu
# from livox_interfaces.msg import CustomMsg
# from std_srvs.srv import Trigger
# import datetime
# import rosbag2_py
# from rclpy.serialization import deserialize_message
# import os
# import subprocess
# import time

# class LivoxBagReader(Node):
#     def __init__(self):
#         super().__init__('livox_bag_reader')
#         self.reader = None
#         self.reading = False
#         self.bag_name = None
#         self.read_timer = None

#         self.start_srv = self.create_service(
#             Trigger,
#             '/start_reading',
#             self.start_callback
#         )

#         self.stop_srv = self.create_service(
#             Trigger,
#             '/stop_reading',
#             self.stop_callback
#         )
#         self.lidar_pub = self.create_publisher(
#             CustomMsg,
#             '/livox/lidar',
#             10
#         )
#         self.imu_pub = self.create_publisher(
#             Imu,
#             '/livox/imu',
#             10
#         )

#     def start_callback(self, request, response):
#         if self.reading:
#             response.success = False
#             response.message = 'Already Recording..'
#             return response

#         self.fastlio_process = subprocess.Popen(
#             ['ros2', 'launch', 'fast_lio', 'mapping.launch.py', 'config_file:=avia.yaml', 'rviz:=false'],
#         )
#         self.get_logger().info('Launched FAST-LIO mapping.launch.py')
#         self.create_timer
#         self.reader = rosbag2_py.SequentialReader()

#         bags_dir = "/ros2_ws/bags"

#         # Get all subdirectories
#         subdirs = [
#             os.path.join(bags_dir, d)
#             for d in os.listdir(bags_dir)
#             if os.path.isdir(os.path.join(bags_dir, d))
#         ]

#         if subdirs:
#             self.bag_name = max(subdirs, key=os.path.getmtime)
#         else:
#             raise RuntimeError("No bag folders found in /ros2_ws/bags")
#         #self.bag_name = '/ros2_ws/bags/latest_scan'

#         storage_options = rosbag2_py.StorageOptions(
#             uri = self.bag_name,
#             storage_id='sqlite3'
#         )
#         converter_options = rosbag2_py.ConverterOptions('','')
#         self.reader.open(storage_options, converter_options)

#         self.reading = True
#         # Following starts the reading loop after delay.
#         self.read_timer = self.create_timer(5.0, self.start_reading_after_delay)

#         response.success = True
#         response.message = f'Reading {self.bag_name}'
#         return response

#     def stop_callback(self, request, response):
#         if not self.reading:
#             response.success = False
#             response.message = 'Not reading..'
#             return response
#         if hasattr(self, 'fastlio_process'):
#             self.fastlio_process.terminate()
#             self.fastlio_process = None
#             self.get_logger().info('Terminated FAST-LIO process')
        
#         self.reading = False
#         self.reader = None

#         # Stop the timer if its running
#         if self.read_timer:
#             self.read_timer.cancel()
#             self.read_timer = None

#         self.get_logger().info(f'Reading stopped: {self.bag_name}')
#         response.success = True
#         response.message = f'Stopped reading {self.bag_name}'
#         return response
        

#     def start_reading_after_delay(self):
#         self.read_timer.cancel()
#         self.get_logger().info(f'Reading started: {self.bag_name}')
#         self.last_msg_time = None
#         self.read_timer = self.create_timer(0.001, self.read_next_message)



#     def read_next_message(self):
#         if not self.reading or self.reader is None:
#             return
        
#         if self.reader.has_next():
#             topic, data, timestamp = self.reader.read_next()

#             if self.last_msg_time is not None:
#                 # Calculate timing difference from previous message
#                 delay = (timestamp - self.last_msg_time) / 1e9  # Convert nanoseconds to seconds
#                 # sleep for the required delay to mimic original timing
#                 if delay >0:
#                     time.sleep(delay)
            
#             self.last_msg_time = timestamp

#             if topic == '/livox/lidar':
#                 msg = deserialize_message(data, CustomMsg)
#                 self.lidar_pub.publish(msg)
#             elif topic == '/livox/imu':
#                 msg = deserialize_message(data, Imu)
#                 self.imu_pub.publish(msg)
#         else:
#             # Bag finished reading
#             self.get_logger().info('Bag finished')
#             self.on_bag_finished()
#             self.reading = False
#             self.reader = None
#             if self.read_timer:
#                 self.read_timer.cancel()
#                 self.read_timer = None
    
#     def on_bag_finished(self):
#         self.get_logger().info('Shutting down FAST-LIO')
#         if hasattr(self, 'fastlio_process'):
#             self.fastlio_process.terminate()
#             self.fastlio_process = None
# def main(args=None):
#     rclpy.init(args=args)
#     node = LivoxBagReader()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':    
#     main()