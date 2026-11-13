// RViz panel for servo (torque) on/off through an actuo bus servicer.
//
// Design rule, and the reason this file is short: THE PANEL DECIDES NOTHING.
// `would_jump` is precomputed server-side and only displayed here; whether an
// enable is allowed is the action server's answer, never this widget's. A panel
// that re-derived the rule could disagree with the robot, and the operator would
// believe the screen.

#ifndef ACTUO_RVIZ_PLUGINS__SERVO_PANEL_HPP_
#define ACTUO_RVIZ_PLUGINS__SERVO_PANEL_HPP_

#include <map>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <QLabel>
#include <QScrollArea>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/panel.hpp>

#include <controller_manager_msgs/srv/list_controllers.hpp>

#include <actuo_msgs/action/set_servo.hpp>
#include <actuo_msgs/msg/servo_state.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace actuo_rviz_plugins
{

/// One joint's row of widgets. Rebuilt only when the joint list itself changes.
struct JointRow
{
  std::string name;
  QLabel * name_label{nullptr};
  QLabel * mismatch_label{nullptr};  // the jump enabling would cause, or a fault
  /// ONE button showing the state it is IN (green ON / red OFF), which a click
  /// flips. Two buttons with one greyed out made the operator cross-reference a
  /// separate "held/limp" column to work out which of them was live -- reading
  /// two things to answer one question, at the moment they least want to.
  QPushButton * toggle{nullptr};
  bool enabled{false};               // last known, so the click knows which way
};

class ServoPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit ServoPanel(QWidget * parent = nullptr);
  ~ServoPanel() override;

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private Q_SLOTS:
  /// Re-subscribe / re-target the action at whatever namespace was typed.
  void onConnect();
  void onAllOn();
  void onAllOff();
  /// Pulls the latest message and the latest action reply on the GUI thread.
  /// ROS callbacks land on RViz's executor thread and must never touch Qt.
  void onTick();

private:
  void sendGoal(bool enable, const std::vector<std::string> & joints);
  /// Tell the CONTROLLER to adopt the robot's current pose, then enable.
  ///
  /// This is the whole reason ON "just works": the jump guard refuses unless the
  /// controller already commands where the robot is, and the controller -- not
  /// the servicer -- owns that setpoint. The servicer cannot fix it from its side (it
  /// re-reads the command every cycle, so anything it wrote would last one
  /// cycle and then jump). So we ask the controller for a trajectory to where
  /// the robot ALREADY is: a move that moves nothing, whose only effect is to
  /// leave the setpoint on the measured pose. Same idea as `on_activate` at
  /// startup, and as Dynamixel practice: write goal = present, then torque on.
  ///
  /// The trajectory carries EVERY joint of each controller involved: the
  /// joints being enabled at their measured angle, the others at their current
  /// command (ServoState `commanded`), so they keep it. A goal with only some of
  /// a controller's joints is refused by a JTC with `allow_partial_joints_goal:
  /// false` (a common configuration), and then
  /// nothing could be enabled one joint at a time. It does NOT leave other
  /// joints' motion alone: JTC holds one trajectory, so this replaces it and
  /// they stop at their existing command. They do not jump; they do stop.
  void resyncThenEnable(const std::vector<std::string> & joints);
  // Which controller commands which joint, asked of controller_manager rather
  // than guessed from a name. A robot with one arm has one; a humanoid may have five
  // (head, each arm, each gripper), and a trajectory naming joints a
  // controller does not own is rejected outright.
  void resyncWithControllers(
    const std::vector<std::string> & selection,
    const std::vector<std::string> & limp,
    const std::map<std::string, double> & targets);
  void sendResyncGoals(
    const std::map<std::string, std::vector<std::string>> & by_controller,
    const std::map<std::string, double> & targets,
    const std::vector<std::string> & selection);
  void rebuildRows(const std::vector<std::string> & names);
  void applyState(const actuo_msgs::msg::ServoState & msg);
  void report(bool ok, const std::string & text);

  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<actuo_msgs::msg::ServoState>::SharedPtr sub_;
  rclcpp_action::Client<actuo_msgs::action::SetServo>::SharedPtr client_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr
    controllers_client_;
  std::map<std::string,
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr> resync_clients_;
  // Goals still outstanding in one re-sync. Torque is enabled when it reaches
  // zero, and NOT once per controller: enabling after the first would leave
  // the others still commanding where the robot used to be.
  std::shared_ptr<std::atomic<int>> resync_pending_;

  // Written by ROS callbacks, read by onTick(). The mutex is the whole
  // thread-safety story: nothing else crosses.
  std::mutex mutex_;
  actuo_msgs::msg::ServoState::SharedPtr latest_;
  bool have_latest_{false};
  std::string pending_reason_;
  bool pending_ok_{false};
  bool have_pending_{false};

  QLineEdit * ns_edit_{nullptr};
  QLineEdit * manager_edit_{nullptr};
  QLabel * status_label_{nullptr};
  QLabel * banner_label_{nullptr};
  QTextEdit * reason_view_{nullptr};
  QVBoxLayout * rows_layout_{nullptr};
  /// The rows' scroll area, so a joint that stops answering can be scrolled
  /// into view: with 22 joints most rows are below the fold, and a missing
  /// servo whose row is hidden is reported by nothing the operator can see.
  QScrollArea * rows_scroll_{nullptr};
  /// The not-answering joints last scrolled to; scroll again only on a change,
  /// never every message (that would fight an operator scrolling the list).
  std::vector<std::string> shown_missing_;
  /// The row still to scroll into view ("" = none), see applyState.
  std::string scroll_to_;
  QWidget * rows_host_{nullptr};
  QPushButton * all_on_{nullptr};
  QPushButton * all_off_{nullptr};
  QTimer * timer_{nullptr};

  std::vector<JointRow> rows_;
  std::string ns_{"/actuo_bus_servicer"};
  // The controller_manager node; its controllers are in its namespace.
  std::string manager_{"/controller_manager"};
};

}  // namespace actuo_rviz_plugins

#endif  // ACTUO_RVIZ_PLUGINS__SERVO_PANEL_HPP_
