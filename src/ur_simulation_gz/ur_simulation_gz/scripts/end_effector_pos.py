#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class EndEffectorPos(Node):
    def __init__(self):
        super().__init__('end_effector_pos')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_end_effector_transform(self):
        from_frame_rel = "pointy"
        to_frame_rel = "world"

        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time()
            )

            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z

            print(f"EE x: {x}, y: {y}, z: {z}")
        
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {to_frame_rel} to {from_frame_rel}: {ex}")
            return
        
def main(args=None):
    rclpy.init(args=args)
    EEPos = EndEffectorPos()
    while rclpy.ok() :
        rclpy.spin_once(EEPos)
        EEPos.get_end_effector_transform()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    EEPos.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()