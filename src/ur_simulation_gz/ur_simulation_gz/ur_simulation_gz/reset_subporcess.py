import subprocess
import os
import sys
import time

def run_ros_gz_sim_create(topic="/robot_description", name="ur", z_position=0):
    """
    Runs the ros_gz_sim create command with the given parameters.

    Args:
        topic (str): The topic for robot description.
        name (str): The name of the robot.
        z_position (float): The initial z-position of the robot.
    """
    # It's crucial to have the ROS 2 environment sourced.
    # If your Python script is being run from a terminal where ROS 2 is already sourced,
    # you might not need this. However, for robustness, especially if this script
    # is part of a larger application or deployed, explicitly sourcing can be necessary.
    # The exact path to setup.bash might vary based on your ROS 2 distribution (e.g., Foxy, Humble)
    # and installation location. Common paths are /opt/ros/<distro>/setup.bash
    # or inside your workspace: ~/ros2_ws/install/setup.bash.
    ros2_setup_script = "/opt/ros/jazzy/setup.bash"  # Adjust this to your ROS 2 distro and path
    if not os.path.exists(ros2_setup_script):
        print(f"Warning: ROS 2 setup script not found at {ros2_setup_script}. "
              f"Please ensure ROS 2 is sourced in your environment or update the path.", file=sys.stderr)

    # Construct the command list. Each part of the command should be a separate item.
    command = [
        "ros2",
        "run",
        "ros_gz_sim",
        "create",
        "-topic", topic,
        "-name", name,
        "-z", str(z_position)  # Convert z_position to string
    ]

    # When running a command that relies on environment variables set by sourcing a script,
    # you often need to run it within a shell where that script has been sourced.
    # This is typically done by passing a single string to `subprocess.run` with `shell=True`,
    # and prepending the sourcing command.
    full_command = f"source {ros2_setup_script} && {' '.join(command)}"

    print(f"Executing command: {full_command}")

    try:
        # Use subprocess.run for a robust way to run external commands.
        # `shell=True` is necessary here because we're using `source` and `&&`.
        # `capture_output=True` captures stdout and stderr.
        # `text=True` decodes stdout and stderr as text.
        # `check=True` raises a CalledProcessError if the command returns a non-zero exit code.
        process = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable="/bin/bash" # Explicitly use bash to source the script
        )

        print("Command executed successfully!")
        print("STDOUT:")
        print(process.stdout)
        print("STDERR:")
        print(process.stderr)

        time.sleep(10)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: The 'ros2' command or 'ros_gz_sim' executable was not found. "
              f"Ensure ROS 2 and ros_gz_sim are installed and sourced correctly.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

def run_ros_gz_sim_spawner(controller_name):
    ros2_setup_script = "/opt/ros/jazzy/setup.bash"  # Adjust this to your ROS 2 distro and path
    if not os.path.exists(ros2_setup_script):
        print(f"Warning: ROS 2 setup script not found at {ros2_setup_script}. "
              f"Please ensure ROS 2 is sourced in your environment or update the path.", file=sys.stderr)
        
    command = [
        "ros2",
        "run",
        "controller_manager",
        "spawner",
        controller_name,
        "--controller-manager",
        "/controller_manager"  # Convert z_position to string
    ]

    full_command = f"source {ros2_setup_script} && {' '.join(command)}"

    print(f"Executing command: {full_command}")

    try:
        process = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable="/bin/bash" # Explicitly use bash to source the script
        )

        print("Command executed successfully!")
        print("STDOUT:")
        print(process.stdout)
        print("STDERR:")
        print(process.stderr)

        time.sleep(5)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: The 'ros2' command or 'ros_gz_sim' executable was not found. "
              f"Ensure ROS 2 and ros_gz_sim are installed and sourced correctly.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

def change_lifecycle_state(node_name, transition):
    """
    Uses a subprocess to call the ros2 lifecycle command.
    
    Args:
        node_name (str): The full name of the lifecycle node (e.g., '/my_node').
        transition (str): The desired transition (e.g., 'configure', 'activate').
    """
    command = ['ros2', 'lifecycle', 'set', node_name, transition]
    print(f"Executing: {' '.join(command)}")
    try:
        # Execute the command. check=True will raise an exception on a non-zero exit code.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully transitioned '{node_name}' to '{transition}' state.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to transition '{node_name}' to '{transition}' state.")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error:\n{e.stderr}")
    except FileNotFoundError:
        print("Error: 'ros2' command not found. Is your ROS 2 environment sourced?")

def node_set_param(node_name, param_name, param_value):
    command = ['ros2', 'param', 'set', node_name, param_name, param_value]
    print(f"Executing: {' '.join(command)}")
    try:
        # Execute the command. check=True will raise an exception on a non-zero exit code.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully set '{param_name}' to '{param_value}'.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to set '{param_name}' to '{param_value}'.")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error:\n{e.stderr}")
    except FileNotFoundError:
        print("Error: 'ros2' command not found. Is your ROS 2 environment sourced?")

def reset_node_subprocesses():
    time.sleep(2)
    run_ros_gz_sim_create()
    time.sleep(6)
    run_ros_gz_sim_spawner("joint_state_broadcaster")
    time.sleep(1)
    run_ros_gz_sim_spawner("scaled_joint_trajectory_controller")
    time.sleep(1)

    return True

if __name__ == "__main__":
    pass
    

    # You can call this function from anywhere in your Python module.
    # For example, if you had another function that needed to spawn this robot:
    # def initialize_simulation():
    #     print("Initializing simulation...")
    #     run_ros_gz_sim_create("/my_robot_description", "my_robot", 1.0)
    #     print("Simulation initialized.")
    #
    # initialize_simulation()