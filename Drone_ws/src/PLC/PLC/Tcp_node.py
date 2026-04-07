import rclpy                          # ROS2 core library
from rclpy.node import Node           # Base Node class
from std_msgs.msg import String  # stays the same, String handles XML too
import socket


class MyNode(Node):

    def __init__(self):
        super().__init__('tcp_node')   # Register node name with ROS2
        self.pub = self.create_publisher(String, 'tcp_topic', 10)
        self.sub = self.create_subscription(String, 'process_topic', self.process_callback, 10)

        self.latest_msg = None  # add this

        self.host = '0.0.0.0'
        self.port = 12343
        self.conn = None        
        
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen()
        self.s.settimeout(0.05)   # 50 ms accept timeout

        self.get_logger().info(f"TCP server started on {self.host}:{self.port}")

    def process_callback(self, msg):
        self.latest_msg = msg.data


    def plc_code(self):

        if self.conn is None:
            try:
                self.conn, addr = self.s.accept()
                self.conn.settimeout(0.01)  # 10 ms recv timeout
                self.get_logger().info(f'connected by {addr}')
            except socket.timeout:
                self.get_logger().info("No connection, retrying...")
                return
        try:
            data = self.conn.recv(1024)

            if data:
                value = data.decode('utf-8')
                self.get_logger().info(f"received: {value}")

                msg = String()
                msg.data = value
                self.get_logger().info(f"Publishing: {msg.data}")
                self.pub.publish(msg)

        except socket.timeout:
            return
        except Exception as e:
            self.get_logger().error(f"Error Receiving Data: {str(e)}")
            self.conn = None

        if self.latest_msg is not None:
            self.get_logger().info(f"Received from process_node: {self.latest_msg}")
            msg = (self.latest_msg + "\r\n").encode('utf-8')
            self.conn.sendall(msg)
            self.get_logger().info(f"Sending to client: {msg}")

            self.latest_msg = None  # reset after sending

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    while rclpy.ok():
        node.plc_code()           # your main loop
        rclpy.spin_once(node)    # process callbacks

if __name__ == '__main__':
    main()
    