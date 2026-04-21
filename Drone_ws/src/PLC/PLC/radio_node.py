import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import serial
import json
import threading
import time


class RadioNode(Node):
    def __init__(self):
        super().__init__('mission_bridge')

        # Serial to 433 MHz telemetry radio
        self.radio = serial.Serial('/dev/ttyUSB0', baudrate=57600, timeout=1.0)

        # Subscribers
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.position_callback, 10)

        self.create_subscription(String, 'mission_status', self.mission_status_callback, 10)

        # Publishers (messages from the radio)
        self.command_pub = self.create_publisher(String, 'mission_command', 10)

        self.waypoint_pub = self.create_publisher(PoseStamped, 'target_waypoint', 10)


        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()
        self.lastSend = 0

        self.get_logger().info('RadioNode started')

    # callbacks
    def position_callback(self, msg):
        if self.lastSend < time.time() + 1:
            packet = {
                'type': 'drone_position',
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
            }
            self.send(packet)
            self.lastSend = time.time()

    def mission_status_callback(self, msg):
        packet = {
            'type': 'mission_status',
            'status': msg.data
        }
        self.send(packet)


    def send(self, packet: dict):
        try:
            line = (json.dumps(packet) + '\n').encode()
            self.radio.write(line)
        except serial.SerialException as e:
            self.get_logger().warn(f'Radio write failed: {e}')

    def read_loop(self):
        while rclpy.ok():
            try:
                line = self.radio.readline().decode().strip()
                if not line:
                    continue
                packet = json.loads(line)
                self.handle_incoming(packet)
            except json.JSONDecodeError:
                self.get_logger().warn(f'Malformed packet received')
            except serial.SerialException as e:
                self.get_logger().warn(f'Radio read failed: {e}')

    def handle_incoming(self, packet: dict):
        msg_type = packet.get('type')

        if msg_type == 'status':
            msg = String()
            msg.data = packet.get('status', '')
            self.command_pub.publish(msg)
            self.get_logger().info(f'UGV status: {msg.data}')

        else:
            self.get_logger().warn(f'Unknown packet type: {msg_type}')


def main(args=None):
    rclpy.init(args=args)
    node = RadioNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()