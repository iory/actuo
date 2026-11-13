cmake_minimum_required(VERSION 3.16)
project(actuo_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

add_library(actuo_hardware SHARED src/actuo_system.cpp)
target_include_directories(actuo_hardware PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(actuo_hardware
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle)

pluginlib_export_plugin_description_file(hardware_interface actuo_hardware.xml)

install(TARGETS actuo_hardware
  EXPORT export_actuo_hardware
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(actuo_hardware)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

ament_package()
