cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gz)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Default to off as starting gzsim doesn't shut down correctly at the moment
option(
  UR_SIM_INTEGRATION_TESTS
  "Run ur_simulation_gz integration tests"
  OFF
)

install(DIRECTORY config launch urdf resources ur_simulation_gz
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS 
  scripts/end_effector_pos.py
  scripts/maddpg_single_object.py
  scripts/simple_subscriber.py
  scripts/simple_action_client.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake)
  find_package(ament_cmake_pytest REQUIRED)

  if(${UR_SIM_INTEGRATION_TESTS})
    add_launch_test(test/test_gz.py
      TIMEOUT
        180
    )
  endif()
  ament_add_pytest_test(description test/test_description.py)
endif()

ament_package()
