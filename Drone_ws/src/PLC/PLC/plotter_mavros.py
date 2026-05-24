import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading


class PlotNode(Node):
    def __init__(self):
        super().__init__('plot_node')

        # Data buffers
        self.xs = []
        self.ys = []
        self.zs = []

        # MAVROS topics are usually BEST_EFFORT
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.local_pos_callback,
            qos
        )

    def local_pos_callback(self, msg):
        self.xs.append(msg.pose.position.x)
        self.ys.append(msg.pose.position.y)
        self.zs.append(msg.pose.position.z)


def main():
    rclpy.init()

    node = PlotNode()

    # ROS2 spin thread
    thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    thread.start()

    # Plot setup
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('MAVROS Local Position')

    def update(frame):

        if len(node.xs) == 0:
            return

        # X
        axes[0].cla()
        axes[0].plot(node.xs)
        axes[0].set_ylabel('X [m]')
        axes[0].grid(True)

        # Y
        axes[1].cla()
        axes[1].plot(node.ys)
        axes[1].set_ylabel('Y [m]')
        axes[1].grid(True)

        # Z
        axes[2].cla()
        axes[2].plot(node.zs)
        axes[2].set_ylabel('Z [m]')
        axes[2].set_xlabel('Samples')
        axes[2].grid(True)

        plt.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=100
    )

    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()