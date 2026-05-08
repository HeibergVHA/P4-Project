import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import numpy as np

class PlotNode(Node):
    def __init__(self):
        super().__init__('plot_node')

        # Data buffers
        self.xs = []
        self.ys = []
        self.zs = []

        self.create_subscription(
            PoseStamped, 'uav/target_pos', self.pos_callback, 10)

    def pos_callback(self, msg):
        self.xs.append(msg.pose.position.x)
        self.ys.append(msg.pose.position.y)
        self.zs.append(msg.pose.position.z)


def main():
    rclpy.init()
    node = PlotNode()

    # Spin ROS2 in background thread so it doesn't block matplotlib
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    # Set up plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('UAV Position')

    def update(frame):
        if not node.xs:
            return

        # X
        axes[0].cla()
        axes[0].plot(node.xs, label='x')
        axes[0].set_ylabel('X [m]')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)

        # Y
        axes[1].cla()
        axes[1].plot(node.ys, label='y', color='orange')
        axes[1].set_ylabel('Y [m]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # Z
        axes[2].cla()
        axes[2].plot(node.zs, label='z', color='green')
        axes[2].set_ylabel('Z [m]')
        axes[2].set_xlabel('Tick')
        axes[2].legend(loc='upper left')
        axes[2].grid(True)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, interval=100)  # update every 100ms
    plt.show()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()