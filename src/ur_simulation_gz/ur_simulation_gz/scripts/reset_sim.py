import ur_simulation_gz.reset_subporcess

class Reset_Call(Node):
    def __init__(self):
        super().__init__('reset_call')

        # self._robot_description_subscriber = self.create_subscription(
        #     String,
        #     '/robot_description',
        #     self.robot_description_listener_callback,
        #     10,
        #     callback_group=self._sub_cb_group)
        # self._robot_description_subscriber

        # --- Service Clients ---
        # Gazebo World Control
        self.world_control_client = self.create_client(ControlWorld, '/world/environment/control')
        self.spawn_client = self.create_client(SpawnEntity, '/world/environment/create')

        # Controller Manager
        self.load_controller_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.configure_controller_client = self.create_client(ConfigureController, '/controller_manager/configure_controller')
        self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        self.reset = False
        self.gazebo_resetting = False
        self.gazebo_resetted = False
        self.robot_spawning = False
        self.robot_spawned = False
        self.broadcaster_configuring_and_loading = False
        self.broadcaster_configured_and_loaded = False
        self.trajectory_control_configuring_and_loading = False
        self.trajectory_control_configured_and_loaded = False
        self.controllers_activating = False
        self.controllers_activated = False

        self._robot_description = None

        self.package_name = "ur_simulation_gz"
        self.robot_name = 'ur' # This must match the name in your URDF and controller config
        self.robot_type = 2 # Entity.msg encoding, with '2' refers to MODEL type

    def reset_sim_world(self):
        """Publishes the command to reset the world."""
        self.get_logger().info("Resetting Gazebo world")
        reset = WorldReset()
        reset.all = True
        world_control = WorldControl()
        world_control.reset = reset
        reset_req = ControlWorld.Request(world_control=world_control)
        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('World control service not available, waiting again...')
        future_reset = self.world_control_client.call_async(reset_req)
        future_reset.add_done_callback(self.reset_sim_world_callback)

    def reset_sim_world_callback(self, future):
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"World resetted successfully")
            self.gazebo_resetted = True
        else:
            self.get_logger().info(f"Failed to reset world")

    def spawn_robot(self):
        pkg_share_path = Path(get_package_share_directory(self.package_name))
        sdf_filename = pkg_share_path / "urdf" / "ur_gz.urdf.xacro.urdf.sdf"
        self.get_logger().info(f"sdf_filename : {sdf_filename}")

        point = Point()
        point.x = 0.
        point.y = 0.
        point.z = 0.

        quaternion = Quaternion()
        quaternion.w = 1.
        quaternion.x = 0.
        quaternion.y = 0.
        quaternion.z = 0.

        pose = Pose()
        pose.position = point
        pose.orientation = quaternion

        entity = EntityFactory()
        entity.name = self.robot_name
        entity.sdf_filename = str(sdf_filename)
        entity.pose = pose

        spawn_req = SpawnEntity.Request(entity_factory=entity)
        self.get_logger().info(f"spawn_req : {spawn_req}")
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn entity service not available, waiting again...')
        future_spawn = self.spawn_client.call_async(spawn_req)
        future_spawn.add_done_callback(self.spawn_robot_callback)

    def spawn_robot_callback(self, future):
        if future.result() is not None and future.result().success:
            self.robot_spawned = True
            self.get_logger().info(f"Robot spawned")
        else :
            self.get_logger().error("Failed to spawn robot")

    def switch_controllers(self):
        controllers_to_reload = ['joint_state_broadcaster', 'scaled_joint_trajectory_controller']
        activate_req = SwitchController.Request(
            activate_controllers=controllers_to_reload,
            strictness=SwitchController.Request.STRICT,
            timeout=Duration(sec=10)
        )
        while not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Switch controller service not available, waiting again...')
        future_unload = self.switch_controller_client.call_async(activate_req)
        future_unload.add_done_callback(self.switch_controllers_callback)

    def switch_controllers_callback(self, future):
        if future.result() is not None and future.result().ok:
            self.get_logger().info(f"Controllers activated successfully")
            self.loaded = True
        else :
            self.get_logger().error("Failed to activate controllers. You may need to manually use `ros2 control` tools to debug")

    def load_and_configure_joint_state_broadcaster(self):
        self.get_logger().info("Reloading and configuring Joint State Broadcaster")
        load_broadcaster_req = LoadController.Request(name="joint_state_broadcaster")
        while not self.load_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Load controller service not available, waiting again...')
        future_broadcaster_load = self.load_controller_client.call_async(load_broadcaster_req)
        future_broadcaster_load.add_done_callback(self.load_broadcaster_callback)
    
    def load_broadcaster_callback(self, future):
        if future.result() is not None and future.result().ok:
            self.get_logger().info("Joint State Broadcaster reloaded successfully")
            conf_broadcaster_req = ConfigureController.Request(name="joint_state_broadcaster")
            while not self.load_controller_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Configure controller service not available, waiting again...')
            future_broadcaster_conf = self.configure_controller_client.call_async(conf_broadcaster_req)
            future_broadcaster_conf.add_done_callback(self.configure_broadcaster_callback)
        else:
            self.get_logger().error("Failed to load Joint State Broadcaster. You may need to manually use `ros2 control` tools to debug")

    def configure_broadcaster_callback(self, future):
        if future.result() is not None and future.result().ok:
            self.get_logger().info("Joint State Broadcaster configured successfully")
            self.broadcaster_configured_and_loaded = True
        else:
            self.get_logger().error("Failed to configure Joint State Broadcaster. You may need to manually use `ros2 control` tools to debug")

    def load_and_configure_scaled_joint_trajectory_controller(self):
        self.get_logger().info("Reloading and configuring Scaled Joint Trajectory Controller")
        load_trajectory_control_req = LoadController.Request(name="scaled_joint_trajectory_controller")
        while not self.load_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Load controller service not available, waiting again...')
        future_trajectory_control_load = self.load_controller_client.call_async(load_trajectory_control_req)
        future_trajectory_control_load.add_done_callback(self.load_trajectory_control_callback)

    def load_trajectory_control_callback(self, future):
        if future.result() is not None and future.result().ok:
            self.get_logger().info("Scaled Joint Trajectory Controller reloaded successfully")
            conf_trajectory_control_req = ConfigureController.Request(name="scaled_joint_trajectory_controller")
            while not self.load_controller_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Configure controller service not available, waiting again...')
            future_trajectory_control_conf = self.configure_controller_client.call_async(conf_trajectory_control_req)
            future_trajectory_control_conf.add_done_callback(self.configure_trajectory_control_callback)
        else:
            self.get_logger().error("Failed to load Scaled Joint Trajectory Controller. You may need to manually use `ros2 control` tools to debug")

    def configure_trajectory_control_callback(self, future):
        if future.result() is not None and future.result().ok:
            self.get_logger().info("Scaled Joint Trajectory Controller configured successfully")
            self.trajectory_control_configured_and_loaded = True
        else:
            self.get_logger().error("Failed to configure Scaled Joint Trajectory Controller. You may need to manually use `ros2 control` tools to debug")

    def reset_process(self):
        if not self.gazebo_resetting and not self.gazebo_resetted:
            self.gazebo_resetting = True
            self.reset_sim_world()
        if self.gazebo_resetted and not self.robot_spawning and not self.robot_spawned:
            self.robot_spawning = True
            self.robot_spawned = ur_simulation_gz.reset_subporcess.reset_node_subprocesses()
        # if self.robot_spawned :
        #     if not self.broadcaster_configuring_and_loading and not self.broadcaster_configured_and_loaded :
        #         self.broadcaster_configuring_and_loading = True
        #         self.broadcaster_configured_and_loaded = ur_simulation_gz.reset_subporcess.run_ros_gz_sim_spawner("joint_state_broadcaster")
        # if self.broadcaster_configured_and_loaded :
        #     if not self.trajectory_control_configuring_and_loading and not self.trajectory_control_configured_and_loaded:
        #         self.trajectory_control_configuring_and_loading = True
        #         self.trajectory_control_configured_and_loaded = ur_simulation_gz.reset_subporcess.run_ros_gz_sim_spawner("scaled_joint_trajectory_controller")

        if self.robot_spawned:
            self.gazebo_resetting = False
            self.gazebo_resetted = False
            self.robot_spawning = False
            self.robot_spawned = False
            self.broadcaster_configuring_and_loading = False
            self.broadcaster_configured_and_loaded = False
            self.trajectory_control_configuring_and_loading = False
            self.trajectory_control_configured_and_loaded = False
            self.controllers_activating = False
            self.controllers_activated = False
            self.reset = False
