#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from geometry_msgs.msg import PoseArray

class StateSubscriber(Node):

    def __init__(self):
        super().__init__('state_subscriber')
        self.joints_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_listener_callback,
            10)
        self.joints_state_subscription  # prevent unused variable warning

        self.box_subscriber = self.create_subscription(
            PoseArray,
            '/world/environment/dynamic_pose/info',
            self.box_listener_callback,
            10
        )

    def joint_listener_callback(self, msg):
        # self.get_logger().info('Joints state: "%s"' % msg.position)
        pass

    def box_listener_callback(self, msg):
        box = msg.poses[0].position
        self.get_logger().info('Box state: "%s"' % box)

def main(args=None):
    rclpy.init(args=args)

    state_subscriber = StateSubscriber()

    rclpy.spin(state_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    state_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()