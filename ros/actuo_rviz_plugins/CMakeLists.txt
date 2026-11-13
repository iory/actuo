cmake_minimum_required(VERSION 3.16)
project(actuo_rviz_plugins)

find_package(ament_cmake REQUIRED)
find_package(actuo_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rviz_common REQUIRED)
find_package(trajectory_msgs REQUIRED)

# Qt's signals/slots need moc run over the header that declares Q_OBJECT.
set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} SHARED
  src/servo_panel.cpp
  include/actuo_rviz_plugins/servo_panel.hpp
  src/diagnostics_panel.cpp
  include/actuo_rviz_plugins/diagnostics_panel.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} Qt5::Widgets)

ament_target_dependencies(${PROJECT_NAME}
  actuo_msgs
  control_msgs
  controller_manager_msgs
  diagnostic_msgs
  pluginlib
  rclcpp
  rclcpp_action
  rviz_common
  trajectory_msgs
)

# rviz_common looks the plugin up through pluginlib at runtime.
pluginlib_export_plugin_description_file(rviz_common plugins_description.xml)

# No ament_export_targets: nothing links against a panel. RViz loads it through
# pluginlib at runtime, so it only has to be installed where the class loader
# looks.
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES plugins_description.xml DESTINATION share/${PROJECT_NAME})

ament_package()
