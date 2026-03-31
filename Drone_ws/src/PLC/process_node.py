import rclpy                          # ROS2 core library
from rclpy.node import Node           # Base Node class
from std_msgs.msg import String  # stays the same, String handles XML too
import socket
import pandas as pd
import xml.etree.ElementTree as ET

class MyNode(Node):

    def __init__(self):
        super().__init__('process_node')   # Register node name with ROS2
        self.sub = self.create_subscription(String, 'tcp_topic', self.tcp_callback, 10)
        self.pub = self.create_publisher(String, 'process_topic', 10)

        self.df = pd.read_csv('/home/heiberg/plc_ws/src/PLC/PLC/procssing_times_table.csv', sep=';', index_col=0)        
        self.data = self.df.values  # converts to numpy 2D array
        self.latest_msg = None  # will hold the latest message from tcp_node

    def tcp_callback(self, msg):
        self.latest_msg = msg.data  # fires when message arrives, stores it


    def process_code(self):
        if self.latest_msg is None:
            return  # no message received yet
        
        root = ET.fromstring(self.latest_msg)
        carrier_id = root.find('CarrierID').text
        date_time  = root.find('DateTime').text
        station_id = root.find('StationID').text
        self.get_logger().info(f'Received CarrierID: {carrier_id}, StationID: {station_id}, DateTime: {date_time}')
        if(int(station_id)<(10)):
            value = self.df.loc[f'Carrier#{carrier_id}', f'Station#0{station_id}']
        else:
            value = self.df.loc[f'Carrier#{carrier_id}', f'Station#{station_id}']

        self.get_logger().info(f'Proccessing Time: {value}, for CarrierID: {carrier_id}, StationID: {station_id}, DateTime: {date_time}')

        self.publisher(value)


        return
    def publisher(self, value):
        root = ET.Element('root')
        ET.SubElement(root, 'value').text = str(value)
        processed_msg = ET.tostring(root, encoding='unicode')
        
        msg = String()
        msg.data = processed_msg
        self.get_logger().info(f"Publishing: {msg.data}")
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    while rclpy.ok():
        node.process_code()           # your main loop
        rclpy.spin_once(node)    # process callbacks

if __name__ == '__main__':
    main()
