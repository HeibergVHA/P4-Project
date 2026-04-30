import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading


class ManualCommandPublisher(Node):

    def __init__(self):
        super().__init__('manual_command_publisher')

        self.publisher_ = self.create_publisher( String, 'ugv/radio_out/mission_command', 10)

        input_thread = threading.Thread(target=self.input_loop)
        input_thread.daemon = True
        input_thread.start()

        self.get_logger().info('Manual command publisher started. Type commands:')

    def input_loop(self):
        while rclpy.ok():
            try:
                command = input("> ")
                msg = String()
                msg.data = command.strip()
                self.publisher_.publish(msg)
                self.get_logger().info(f'Published: {msg.data}')
            except Exception as e:
                self.get_logger().error(f'Error reading input: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ManualCommandPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()