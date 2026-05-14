import rclpy
from rclpy.node import Node
from px4_msgs.msg import (ActuatorMotors, OffboardControlMode, VehicleCommand, 
                          VehicleAttitudeSetpoint, VehicleOdometry, TrajectorySetpoint)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from control import TransferFunction, step_response

class MotorCommander(Node):
    def __init__(self):
        super().__init__('motor_commander')

        self.x, self.y, self.z = 0, 0, 0
        self.dx, self.dy, self.dz = 0, 0, 0

        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

        # Configure QoS for PX4 topics
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribe to vehicle odometry to get position and velocity feedback
        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos
        )

        self.vel_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.command_pub  = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.counter = 0
        # Run at 50 Hz
        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory()

        # Arm and switch mode
        if self.counter == 10:
            self.arm()
            self.switch_to_offboard_mode()

        self.counter += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()

        # Configure control mode
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False # True
        msg.velocity = True # False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        
        self.offboard_pub.publish(msg)

    # Print odometry feedback and update internal state
    def odometry_callback(self, msg):
        self.get_logger().info(f'{msg.position}\t{msg.velocity}')
        self.x, self.y, self.z = msg.position[0], msg.position[1], msg.position[2]
        self.dx, self.dy, self.dz = msg.velocity[0], msg.velocity[1], msg.velocity[2]

    # Publish trajectory setpoint to command the drone's position
    def publish_trajectory(self):
 
        msg = TrajectorySetpoint()
        
        #if self.counter % 2000 == 1000 or self.counter % 2000 == 0:


        # Alternate between two positions to create a back-and-forth motion in the x-axis
        if self.counter % 2000 < 1000:
            msg.acceleration = [float('nan')] * 3# [0.0, 0.0, 0.0]
            msg.velocity = [float('nan')] * 3 # [-1.0, 0.0, 0.0] # [float('nan')] * 3
            msg.position = [0.0, 0.0, -2.0] # [self.counter*0.02, 0.0, -2.5]
        else:
            msg.acceleration = [float('nan')] * 3 # [0.0, 0.0, 0.0]
            msg.velocity = [float('nan')] * 3 # [0.0, 0.0, 0.0] # [float('nan')] * 3
            msg.position = [0.0, 0.0, -1.0] # [self.counter*0.02, 0.0, -2.5]

        msg.yaw = 0.0
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.vel_pub.publish(msg)

    # Arms the drone automatically
    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.command_pub.publish(msg)
        self.get_logger().info('Arm command sent')

    # Switch to offboard mode to allow external control
    def switch_to_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.command_pub.publish(msg)
        self.get_logger().info('Offboard mode command sent')


def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()