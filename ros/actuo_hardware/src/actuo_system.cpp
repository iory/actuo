#include "actuo_hardware/actuo_system.hpp"
#include <thread>
#include <chrono>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"

namespace
{
bool parse_bool(const std::string & s, bool fallback)
{
  if (s.empty()) { return fallback; }
  std::string v = s;
  std::transform(v.begin(), v.end(), v.begin(),
    [](unsigned char c) { return std::tolower(c); });
  return v == "true" || v == "1" || v == "yes";
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

namespace actuo_hardware
{

hardware_interface::CallbackReturn ActuoSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  n_ = info_.joints.size();
  // Kept for the log line only: the command is seeded from the MEASURED state
  // at activation (see on_activate), never from this.
  init_pos_.assign(n_, 0.0);
  for (size_t i = 0; i < n_; ++i) {
    for (const auto & si : info_.joints[i].state_interfaces) {
      if (si.name == hardware_interface::HW_IF_POSITION && !si.initial_value.empty()) {
        try {
          init_pos_[i] = std::stod(si.initial_value);
        } catch (const std::exception &) {}
      }
    }
  }
  pos_state_ = init_pos_;
  vel_state_.assign(n_, 0.0);
  present_.assign(n_, false);
  cmd_.assign(n_ * kCmdStride, kNaN);

  auto it = info_.hardware_parameters.find("shm_name");
  if (it != info_.hardware_parameters.end()) { shm_name_ = it->second; }
  it = info_.hardware_parameters.find("use_sim");
  if (it != info_.hardware_parameters.end()) {
    use_sim_ = parse_bool(it->second, true);
  }
  it = info_.hardware_parameters.find("state_timeout");
  if (it != info_.hardware_parameters.end()) {
    try {
      state_timeout_ = std::stod(it->second);
    } catch (const std::exception &) {
      RCLCPP_ERROR(get_logger(), "state_timeout '%s' is not a number", it->second.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  // Which segment and slot each joint lives in. Defaults reproduce the
  // single-component layout: every joint in `shm_name`, slot = its index.
  segments_.clear();
  for (size_t i = 0; i < n_; ++i) {
    const auto & jp = info_.joints[i].parameters;
    std::string segment = shm_name_;
    size_t slot = i;
    auto p = jp.find("shm_name");
    if (p != jp.end()) { segment = p->second; }
    p = jp.find("slot");
    if (p != jp.end()) {
      try {
        const long v = std::stol(p->second);
        if (v < 0) { throw std::out_of_range("negative"); }
        slot = static_cast<size_t>(v);
      } catch (const std::exception &) {
        RCLCPP_ERROR(
          get_logger(), "joint %s: slot '%s' is not a non-negative integer",
          info_.joints[i].name.c_str(), p->second.c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
    }
    auto seg = std::find_if(
      segments_.begin(), segments_.end(),
      [&](const Segment & s) { return s.name == segment; });
    if (seg == segments_.end()) {
      segments_.emplace_back();
      segments_.back().name = segment;
      seg = segments_.end() - 1;
    }
    if (std::find(seg->slots.begin(), seg->slots.end(), slot) != seg->slots.end()) {
      RCLCPP_ERROR(
        get_logger(), "joint %s: slot %zu of segment '%s' is already used in this component",
        info_.joints[i].name.c_str(), slot, segment.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    seg->joints.push_back(i);
    seg->slots.push_back(slot);
  }
  for (auto & seg : segments_) {
    seg.cmd.assign(seg.joints.size() * kCmdStride, kNaN);
    std::string where;
    for (size_t k = 0; k < seg.joints.size(); ++k) {
      where += (k ? ", " : "") + info_.joints[seg.joints[k]].name + "->" +
        std::to_string(seg.slots[k]);
    }
    RCLCPP_INFO(
      get_logger(), "actuo_hardware on_init: segment '%s': %s", seg.name.c_str(), where.c_str());
  }
  RCLCPP_INFO(
    get_logger(), "actuo_hardware on_init: %zu joints in %zu segment(s), use_sim=%s",
    n_, segments_.size(), use_sim_ ? "true" : "false");
  return hardware_interface::CallbackReturn::SUCCESS;
}

void ActuoSystem::detach_all()
{
  for (auto & seg : segments_) { seg.shm.reset(); }
}

std::string ActuoSystem::absent_joint_names() const
{
  std::string out;
  for (size_t i = 0; i < n_; ++i) {
    if (!present_[i]) { out += (out.empty() ? "" : ", ") + info_.joints[i].name; }
  }
  return out;
}

hardware_interface::CallbackReturn ActuoSystem::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Attach to every segment, adopt the pose the robot is ACTUALLY in, and
  // command exactly that for this module's slots.
  //
  // The servicer creates each segment; we only attach. A segment is believed
  // only while its servicer holds the lock: the file is under /tmp and
  // outlives every process, so one left by an earlier run holds that
  // run's state, and seeding a command from it is the lurch this whole
  // handshake exists to prevent. Commanding the measured pose is also the
  // ros2_control convention (goal = present before torque, as on Dynamixel).
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration<double>(state_timeout_);
  segment_recreated_ = false;
  for (auto & seg : segments_) {
    std::string last_error = "servicer not running";
    while (!seg.shm) {
      if (ShmBridge::servicer_alive(seg.name)) {
        try {
          auto shm = std::make_unique<ShmBridge>(seg.name);
          if (shm->state_ever_written()) {
            seg.shm = std::move(shm);
            break;
          }
          last_error = "no state published yet";
        } catch (const std::exception & e) {
          last_error = e.what();
        }
      }
      if (std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(
          get_logger(),
          "actuo_hardware: segment '%s' not ready within %.1fs (%s). Refusing to "
          "activate: without knowing where the robot IS, any command we issue is a "
          "guess. Start the servicer (actuo_bus_node --real --shm-name %s) first, or "
          "use backend:=mock.", seg.name.c_str(), state_timeout_, last_error.c_str(),
          seg.name.c_str());
        detach_all();
        return hardware_interface::CallbackReturn::ERROR;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (size_t k = 0; k < seg.slots.size(); ++k) {
      if (seg.slots[k] >= seg.shm->slots()) {
        RCLCPP_ERROR(
          get_logger(), "joint %s: slot %zu but segment '%s' has %u slots",
          info_.joints[seg.joints[k]].name.c_str(), seg.slots[k], seg.name.c_str(),
          seg.shm->slots());
        detach_all();
        return hardware_interface::CallbackReturn::ERROR;
      }
    }
    seg.epoch = seg.shm->epoch();
    if (!seg.shm->read_state(seg.state)) {
      RCLCPP_ERROR(get_logger(), "actuo_hardware: state seqlock of '%s' did not converge",
        seg.name.c_str());
      detach_all();
      return hardware_interface::CallbackReturn::ERROR;
    }
    for (size_t k = 0; k < seg.joints.size(); ++k) {
      const size_t i = seg.joints[k];
      const double * s = &seg.state[seg.slots[k] * ShmBridge::kStateStride];
      present_[i] = s[ShmBridge::kStatePresent] > 0.5;
      pos_state_[i] = s[0];
      vel_state_[i] = s[1];
    }
  }

  // A module whose actuators are not all there does not come up. FAILURE, not
  // ERROR: the component stays inactive and can be activated again once they
  // answer -- which is how a module that is plugged in later joins.
  const std::string absent = absent_joint_names();
  if (!absent.empty()) {
    RCLCPP_WARN(
      get_logger(), "actuo_hardware: not activating, no answer from: %s", absent.c_str());
    detach_all();
    return hardware_interface::CallbackReturn::FAILURE;
  }

  for (auto & seg : segments_) {
    for (size_t k = 0; k < seg.joints.size(); ++k) {
      const size_t i = seg.joints[k];
      // Command position = measured position. Velocity/effort start at 0.
      cmd_[kCmdStride * i + 0] = pos_state_[i];
      cmd_[kCmdStride * i + 1] = 0.0;
      cmd_[kCmdStride * i + 2] = 0.0;
      RCLCPP_INFO(
        get_logger(), "actuo_hardware: %s adopted measured %.4f rad (URDF said %.4f)",
        info_.joints[i].name.c_str(), pos_state_[i], init_pos_[i]);
      std::copy_n(&cmd_[kCmdStride * i], kCmdStride, &seg.cmd[kCmdStride * k]);
    }
    seg.shm->write_command_slots(seg.slots, seg.cmd.data());
  }
  RCLCPP_INFO(get_logger(), "actuo_hardware activated from MEASURED state");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ActuoSystem::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Give the slots back: NaN is "no controller owns this", on which the
  // servicer holds a position joint where it is and stops a velocity one.
  for (auto & seg : segments_) {
    if (!seg.shm) { continue; }
    std::fill(seg.cmd.begin(), seg.cmd.end(), kNaN);
    seg.shm->write_command_slots(seg.slots, seg.cmd.data());
  }
  detach_all();
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> ActuoSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &pos_state_[i]);
    interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &vel_state_[i]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface> ActuoSystem::export_command_interfaces()
{
  // Export whichever of position/velocity/effort the URDF declares, each backed
  // by its slot in the flat command block. Controllers claim the one they drive;
  // the servicer applies the active mode's slot to the plant.
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    for (const auto & ci : info_.joints[i].command_interfaces) {
      size_t slot = 0;
      if (ci.name == hardware_interface::HW_IF_VELOCITY) { slot = 1; }
      else if (ci.name == hardware_interface::HW_IF_EFFORT) { slot = 2; }
      interfaces.emplace_back(
        info_.joints[i].name, ci.name, &cmd_[kCmdStride * i + slot]);
    }
  }
  return interfaces;
}

hardware_interface::return_type ActuoSystem::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (auto & seg : segments_) {
    if (!seg.shm) { continue; }
    if (seg.shm->epoch() != seg.epoch) {
      segment_recreated_ = true;
      continue;
    }
    // A non-converged read keeps the previous state rather than faking one.
    if (!seg.shm->read_state(seg.state)) { continue; }
    for (size_t k = 0; k < seg.joints.size(); ++k) {
      const size_t i = seg.joints[k];
      const double * s = &seg.state[seg.slots[k] * ShmBridge::kStateStride];
      present_[i] = s[ShmBridge::kStatePresent] > 0.5;
      if (present_[i]) {
        pos_state_[i] = s[0];
        vel_state_[i] = s[1];
      }
    }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ActuoSystem::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // DEACTIVATE from write() -- not read(), where it counts as ERROR -- makes
  // controller_manager deactivate the controllers commanding this component
  // and only those; every other module keeps running.
  if (segment_recreated_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "actuo_hardware: a servicer re-created its segment; deactivating this module");
    return hardware_interface::return_type::DEACTIVATE;
  }
  const std::string absent = absent_joint_names();
  if (!absent.empty()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "actuo_hardware: no answer from %s; deactivating this module", absent.c_str());
    return hardware_interface::return_type::DEACTIVATE;
  }
  for (auto & seg : segments_) {
    if (!seg.shm) { continue; }
    for (size_t k = 0; k < seg.joints.size(); ++k) {
      std::copy_n(&cmd_[kCmdStride * seg.joints[k]], kCmdStride, &seg.cmd[kCmdStride * k]);
    }
    seg.shm->write_command_slots(seg.slots, seg.cmd.data());
  }
  return hardware_interface::return_type::OK;
}

}  // namespace actuo_hardware

PLUGINLIB_EXPORT_CLASS(actuo_hardware::ActuoSystem, hardware_interface::SystemInterface)
