cmake_minimum_required(VERSION 3.16)
project(actuo_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointServoState.msg"
  "msg/ServoState.msg"
  "action/SetServo.action"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_package()
