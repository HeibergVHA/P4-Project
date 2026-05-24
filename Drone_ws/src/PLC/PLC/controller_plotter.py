import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mavros_msgs.msg import AttitudeTarget

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

from scipy.spatial.transform import Rotation as R


class AttitudePlotNode(Node):
    def __init__(self):
        super().__init__('attitude_plot_node')

        self.roll = []
        self.pitch = []
        self.yaw = []

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            self.attitude_callback,
            qos
        )

    def attitude_callback(self, msg):
        q = msg.orientation  # geometry_msgs/Quaternion

        rpy = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True)

        self.roll.append(rpy[0])
        self.pitch.append(rpy[1])
        self.yaw.append(rpy[2])


def main():
    rclpy.init()

    node = AttitudePlotNode()

    thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    thread.start()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('MAVROS Attitude (Roll / Pitch / Yaw)')

    def update(_):

        if len(node.roll) == 0:
            return

        axes[0].cla()
        axes[0].plot(node.roll)
        axes[0].set_ylabel('Roll [rad]')
        axes[0].grid(True)

        axes[1].cla()
        axes[1].plot(node.pitch)
        axes[1].set_ylabel('Pitch [rad]')
        axes[1].grid(True)

        axes[2].cla()
        axes[2].plot(node.yaw)
        axes[2].set_ylabel('Yaw [rad]')
        axes[2].set_xlabel('Samples')
        axes[2].grid(True)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, interval=100)

    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()