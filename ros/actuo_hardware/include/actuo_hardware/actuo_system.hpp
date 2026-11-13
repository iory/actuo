// actuo_hardware -- thin, motor-agnostic ros2_control SystemInterface.
//
// This C++ layer knows NO motor protocol. It runs in the controller's RT
// read/write loop and exchanges only flat joint command/state vectors with the
// Python bus servicer over a shared-memory seqlock (ShmBridge). Keeping protocol
// knowledge OUT of here means there is no second implementation of any device
// to keep in step with the Python one.
//
// One component is one MODULE: any subset of joints, each living in a slot of
// some servicer's segment. Per joint, `<param name="slot">` picks the slot
// (default: the joint's index in this component) and `<param name="shm_name">`
// the segment (default: the component's `shm_name`), so a module can span
// buses. Several components can share one segment; each writes only its own
// slots. When a slot's actuator stops answering, write() returns DEACTIVATE
// and controller_manager deactivates exactly the controllers commanding this
// module, leaving every other module running.
#ifndef ACTUO_HARDWARE__ACTUO_SYSTEM_HPP_
#define ACTUO_HARDWARE__ACTUO_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "actuo_hardware/shm_bridge.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_component_interface_params.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace actuo_hardware
{

class ActuoSystem : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // One segment this component has joints in.
  struct Segment
  {
    std::string name;
    std::vector<size_t> joints;  // indices into info_.joints
    std::vector<size_t> slots;   // the same joints' slots in the segment
    std::unique_ptr<ShmBridge> shm;
    uint64_t epoch = 0;
    std::vector<double> state;   // scratch: the whole state block
    std::vector<double> cmd;     // scratch: kCmdStride per joint, in `joints` order
  };

  void detach_all();
  // Names of joints whose actuator does not answer, comma separated.
  std::string absent_joint_names() const;

  std::string shm_name_ = "actuo";
  bool use_sim_ = true;
  size_t n_ = 0;

  static constexpr size_t kCmdStride = ShmBridge::kCmdStride;
  // Seconds on_activate() waits for a live servicer and its first measured
  // state before refusing. Overridable via the `state_timeout` hardware param.
  double state_timeout_ = 10.0;

  std::vector<double> init_pos_;       // per-joint initial_value from the URDF
  std::vector<double> pos_state_;
  std::vector<double> vel_state_;
  std::vector<bool> present_;
  std::vector<double> cmd_;            // flat [pos,vel,eff] per joint
  std::vector<Segment> segments_;
  // Set by read() when a segment's epoch changes under us: the servicer was
  // restarted and the state we adopted at activation belongs to another life.
  bool segment_recreated_ = false;
};

}  // namespace actuo_hardware

#endif  // ACTUO_HARDWARE__ACTUO_SYSTEM_HPP_
