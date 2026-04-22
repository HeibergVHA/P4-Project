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
        self.radio = serial.Serial('/dev/ttyUSB0', baudrate=57600, timeout=1.0) # Find correct USB port. ######################################

        self.create_subscription(String, 'ugv/radio_out/target_waypoint', self.target_waypoint_callback, 10)

        self.create_subscription(String, 'ugv/radio_out/mission_command', self.mission_command_callback, 10)

        self.uav_pos_pub = self.create_publisher(PoseStamped, 'ugv/radio_in/uav_pos', 10)


        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()
        self.lastSend = 0

        self.get_logger().info('RadioNode started')

    # callbacks
    def target_waypoint_callback(self, msg):
        try:
            self.radio.write(msg)
        except serial.SerialException as e:
            self.get_logger().warn(f'Radio write failed: {e}')

    def mission_command_callback(self, msg):
        packet = {
            'type': 'command',
            'command': msg.data
        }
        self.send(packet)


    def send(self, packet):
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

    def handle_incoming(self, packet):
        msg_type = packet.get('type')

        if msg_type == 'status':
            msg = String()
            msg.data = packet.get('status', '')
            self.command_pub.publish(msg)
            self.get_logger().info(f'UAV status: {msg.data}')
        if msg_type == 'drone_position':
            stamp = self.get_clock().now().to_msg()
            msg = PoseStamped()             # Create and send the orientation 
            msg.header.stamp = stamp
            msg.header.frame_id = 'UAVPosition'
            msg.pose.position.x = packet.get('x', 0.0)
            msg.pose.position.y = packet.get('y', 0.0)
            msg.pose.position.z = packet.get('z', 0.0)
            self.uav_pos_pub.publish(msg)

        else:
            self.get_logger().warn(f'Unknown packet type: {msg_type}')


def main(args=None):
    rclpy.init(args=args)
    node = RadioNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()