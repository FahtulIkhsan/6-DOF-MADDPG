#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.time
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseArray

from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ur_simulation_gz.maddpg import MADDPG

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import traceback

import time

class Robot_Arm(Node) :
    def __init__(self):
        super().__init__('robot_arm')

        # Setup callback groups
        self._sub_cb_group = ReentrantCallbackGroup()
        self._action_cb_group = MutuallyExclusiveCallbackGroup()
        
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self._action_cb_group)
        
        self._joints_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_listener_callback,
            10,
            callback_group=self._sub_cb_group)
        self._joints_state_subscription  # prevent unused variable warning

        self._box_subscriber = self.create_subscription(
            PoseArray,
            '/world/environment/dynamic_pose/info',
            self.box_listener_callback,
            10,
            callback_group=self._sub_cb_group)
        self._box_subscriber

        self.maddpg = MADDPG()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint configuration
        self._joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint',
            'elbow_joint', 'wrist_1_joint',
            'wrist_2_joint', 'wrist_3_joint'
        ]
        self._joint_limit = [1.57, 0.27, 0.75, 3.2, 3.2, 3.2]

        self._trajectory_duration = 3.0
        self._timeout_sec = 100.0
        self._joint_states = None
        self._box_states = None
        self._processing = False
        self._resetting = False
        self._reset_tf = False
        self.result_code = None

    def joint_listener_callback(self, msg):
        self._joint_states = msg

    def box_listener_callback(self, msg):
        self._box_states = msg

    def cleanup_action_client(self):
        # Cancel all possible futures
        self.get_logger().warn("Cancelling futures")
        if self._result_timeout_timer:
            self._result_timeout_timer.cancel()

        if self._get_result_future:
            if not self._get_result_future.done():
                self._get_result_future.cancel()
                self._goal_handle.cancel_goal_async()
        elif self._goal_handle:
            self._goal_handle.cancel_goal_async()

        if self._send_goal_future:
            if not self._send_goal_future.done():
                self._send_goal_future.cancel()

        # Destroy existing action client then create new action client
        self.get_logger().warn(f"Destroying action client : {self._action_client}")
        self._action_client.destroy()

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self._action_cb_group)
        self.get_logger().info(f"New action client created : {self._action_client}")

    def send_trajectory(self, positions):      
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self._joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(self._trajectory_duration)
        point.time_from_start.nanosec = int((self._trajectory_duration % 1) * 1e9)
        goal_msg.trajectory.points.append(point)

        while not self._action_client.wait_for_server():
            self.get_logger().info('Waiting for scaled joint trajectory controller...')
        
        # Send goal and keep future reference
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        # Set a timer to cancel if no response
        self._result_timeout_timer = self.create_timer(self._timeout_sec, self.cleanup_action_client)

    def goal_response_callback(self, future):
        if self._result_timeout_timer:
            self._result_timeout_timer.cancel()

        """Handle goal acceptance"""
        if future.done():
            try:
                self._goal_handle = future.result()
                if not self._goal_handle.accepted:
                    self.get_logger().error("Goal rejected by server!")
                    return

                # Get result and keep future reference
                self._get_result_future = self._goal_handle.get_result_async()
                self._get_result_future.add_done_callback(self.goal_result_callback)

                # Set a timer to cancel if no response
                self._result_timeout_timer = self.create_timer(self._timeout_sec, self.cleanup_action_client)
            
            except Exception as e:
                self.get_logger().error(f"Goal response error: {str(e)}")
                self.cleanup_action_client()

    def goal_result_callback(self, future):
        if self._result_timeout_timer:
            self._result_timeout_timer.cancel()
        
        """Handle trajectory completion"""
        if future.done():
            try:
                if self._resetting:
                    if self._reset_tf:
                        self.tf_buffer.clear()
                        self._reset_tf = False
                    current_positions = self.observe_states()
                    current_distance = self.maddpg.distance_from_observations(current_positions)
                    self.get_logger().info(f"Current Positions : {current_positions}")
                    self.get_logger().info(f"Current Distance : {current_distance}")
                    self.cleanup_action_client()
                    self._resetting = False
                    self._processing = False
                else:
                    result = future.result().result
                    self.result_code = result.error_code
                    self.get_logger().info(f"Result: {self.result_code}")

                    current_positions = self.observe_states()
                    self.get_logger().info(f"Current Positions : {current_positions}")

                    self.maddpg.store(current_positions, self.result_code)
                    self.maddpg.step()

                    check = self.maddpg.end_frame_check()
                    self.get_logger().info(f"Check : {check}")

                    if check == 1:
                        self.maddpg.new_episode()
                        self._resetting = True
                        self._reset_tf = True
                        self._processing = False
                    elif check == 2:
                        self.maddpg.terminate()
                        self.destroy_node()
                        time.sleep(10)
                        rclpy.shutdown()
                    elif check == 3:
                        self.maddpg.new_episode()
                        self.tf_buffer.clear()
                        self.cleanup_action_client()
                        self._processing = False
                    elif check == 0:
                        self._processing = False
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().info(f"Goal result transform not ready: {str(e)}")
                self._processing = False
            except Exception as e:
                self.get_logger().error(f"Goal result error: {str(e)}")
                self.cleanup_action_client()

    def feedback_callback(self, feedback_msg):
        """Handle feedback messages"""
        # self.get_logger().info(
        #     f"Current positions: {feedback_msg.feedback.actual.positions}")

    def get_end_effector_transform(self):
        from_frame_rel = "pointy"
        to_frame_rel = "world"

        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z

            return [x, y, z]
        
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info(f"Transform not ready: {e}")
            raise
        except TransformException as e:
            self.get_logger().error(f"Could not transform {to_frame_rel} to {from_frame_rel}: {e}")
            return
        
    def get_links_transform(self):
        from_wrist_3_frame_rel = "wrist_3_link"
        from_wrist_2_frame_rel = "wrist_2_link"
        from_wrist_1_frame_rel = "wrist_1_link"
        from_forearm_frame_rel = "forearm_link"
        to_frame_rel = "world"

        try:
            t_wrist_3 = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_wrist_3_frame_rel,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x_wrist_3 = t_wrist_3.transform.translation.x
            y_wrist_3 = t_wrist_3.transform.translation.y
            z_wrist_3 = t_wrist_3.transform.translation.z

            t_wrist_2 = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_wrist_2_frame_rel,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x_wrist_2 = t_wrist_2.transform.translation.x
            y_wrist_2 = t_wrist_2.transform.translation.y
            z_wrist_2 = t_wrist_2.transform.translation.z

            t_wrist_1 = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_wrist_1_frame_rel,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x_wrist_1 = t_wrist_1.transform.translation.x
            y_wrist_1 = t_wrist_1.transform.translation.y
            z_wrist_1 = t_wrist_1.transform.translation.z
            
            t_forearm = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_forearm_frame_rel,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x_forearm = t_forearm.transform.translation.x
            y_forearm = t_forearm.transform.translation.y
            z_forearm = t_forearm.transform.translation.z

            return [x_wrist_3, y_wrist_3, z_wrist_3,
                    x_wrist_2, y_wrist_2, z_wrist_2,
                    x_wrist_1, y_wrist_1, z_wrist_1,
                    x_forearm, y_forearm, z_forearm]
        
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info(f"Transforms not ready: {e}")
            raise
        except TransformException as e:
            self.get_logger().error(f"Could not transforms: {e}")
            return

    def observe_states(self):
        # Process current state
        try:
            current_positions = [
                self._joint_states.position[self._joint_states.name.index(name)] 
                for name in self._joint_names
            ]

            links_position = self.get_links_transform()
            current_positions.extend(links_position)

            end_effector_position = self.get_end_effector_transform()
            current_positions.extend(end_effector_position)

            return current_positions
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            raise
        except AttributeError as e:
            self.get_logger().info(f"Data not ready : {e}")
            raise
        except Exception as e:
            self.get_logger().warn(f"Observe not ready : {e}")
            return
        
    def process_states(self):
        self._processing = True

        try:
            if self._resetting :
                target_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
                self.get_logger().info(f"Sending trajectory {target_positions}")
                self.send_trajectory(target_positions)            
            else:
                # Process current state
                current_positions = self.observe_states()

                # Your processing logic here
                calculation_result = self.maddpg.act(current_positions)

                target_positions = []
                for i, joint_limit in enumerate(self._joint_limit) :
                    if i == 1 :
                        target_position = (calculation_result[i] * joint_limit) - 1.30
                    elif i == 2 :
                        target_position = (calculation_result[i] * joint_limit) + 0.74
                    elif i == 3 :
                        target_position = (calculation_result[i] * joint_limit) - 1.57
                    else :
                        target_position = calculation_result[i] * joint_limit

                    target_positions.append(target_position)

                self.get_logger().info(f"Sending trajectory {target_positions}")

                # Send trajectory
                self.send_trajectory(target_positions)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._processing = False
        except AttributeError as e:
            self._processing = False
        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")
            traceback.print_exc()
            self._processing = False

def main(args=None):
    rclpy.init(args=args)

    action_client = Robot_Arm()

    while rclpy.ok() :
        rclpy.spin_once(action_client)

        if not action_client._processing:
            action_client.process_states()

    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()