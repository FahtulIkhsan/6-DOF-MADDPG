#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import random

class SimpleActionClient(Node):

    def __init__(self):
        super().__init__('simple_action_client')
        
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        
        # Joint configuration
        self._joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint',
            'elbow_joint', 'wrist_1_joint',
            'wrist_2_joint', 'wrist_3_joint'
        ]
        self._trajectory_duration = 2.0
        self.processing = False

    def send_trajectory(self):

        self.processing = True
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self._joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        for i in goal_msg.trajectory.joint_names :
            point.positions.append(random.uniform(-2.9, 2.9))
        point.time_from_start.sec = int(self._trajectory_duration)
        point.time_from_start.nanosec = int((self._trajectory_duration % 1) * 1e9)
        goal_msg.trajectory.points.append(point)

        self._action_client.wait_for_server()
        
        self.get_logger().info(f"Sending trajectory goal... \n {goal_msg}")
        
        # Send goal and keep future reference
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        """Handle goal acceptance"""
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self.get_logger().error("Goal rejected by server!")
                return
                
            self.get_logger().info(f"Goal accepted with ID: {self.goal_handle.goal_id}")
            
            # Get result and keep future reference
            self._get_result_future = self.goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.goal_result_callback)
            
        except Exception as e:
            self.get_logger().error(f"Goal response error: {str(e)}")

    def goal_result_callback(self, future):
        """Handle trajectory completion"""
        try:
            result = future.result().result
            self.get_logger().info(
                f"Trajectory completed with:\n"
                f"Error code: {result.error_code}\n"
                f"Message: '{result.error_string}'")
                
        except Exception as e:
            self.get_logger().error(f"Result error: {str(e)}")

        self.processing = False

    def feedback_callback(self, feedback_msg):
        """Handle feedback messages"""
        self.get_logger().info(
            f"Current positions: {feedback_msg.feedback.actual.positions}")



def main(args=None):
    rclpy.init(args=args)

    action_client = SimpleActionClient()

    while rclpy.ok() :
        rclpy.spin_once(action_client)

        if not action_client.processing :
            action_client.send_trajectory()

    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()