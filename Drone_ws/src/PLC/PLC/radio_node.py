import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R # To convert between Euler degrees and quatanions.
import serial
import json
import threading
import time


class RadioNode(Node):
    def __init__(self):
        super().__init__('radio_node')

        # Serial to 433 MHz telemetry radio
        self.radio = serial.Serial('/dev/ttyUSB0', baudrate=57600, timeout=1.0) # Should be correct USB port ###########################

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        # Subscribers
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.position_callback, qos)

        self.create_subscription(String, 'uav/radio_out/mission_status', self.status_callback, 10)

        # Publishers (messages from the radio)
        self.command_pub = self.create_publisher(String, 'uav/radio_in/mission_command', 10)

        self.waypoint_pub = self.create_publisher(PoseStamped, 'uav/radio_in/target_waypoint', 10)


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

    def status_callback(self, msg):
        packet = {
            'type': 'status',
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

        if msg_type == 'command':
            msg = String()
            msg.data = packet.get('command', '')
            self.command_pub.publish(msg)
            self.get_logger().info(f'Command from UGV: {msg.data}')
        if msg_type == 'waypoint':
            stamp = self.get_clock().now().to_msg()
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'waypoint'
            msg.pose.position.x = packet.get('x', 0.0)
            msg.pose.position.y = packet.get('y', 0.0)
            msg.pose.position.z = packet.get('z', 0.0)
            r = R.from_euler('xyz', [0, 0, packet.get('yaw', 0.0)], degrees=True) # Euler degrees to quatanions -->
            q = r.as_quat()  # [x, y, z, w]
            msg.pose.orientation.x = q[1] # Always 0. Independent of yaw.
            msg.pose.orientation.y = q[2] # Always 0. Independent of yaw.
            msg.pose.orientation.z = q[3]
            msg.pose.orientation.w = q[0]
            self.waypoint_pub.publish(msg)


        else:
            self.get_logger().warn(f'Unknown packet type: {msg_type}')


def main(args=None):
    rclpy.init(args=args)
    node = RadioNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()