#include "actuo_rviz_plugins/servo_panel.hpp"

#include <algorithm>
#include <cmath>

#include <QGridLayout>
#include <QHBoxLayout>
#include <QScrollArea>

#include <rclcpp/duration.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace actuo_rviz_plugins
{

namespace
{
constexpr double kRadToDeg = 57.29577951308232;

QString jointSummary(const actuo_msgs::msg::JointServoState & j)
{
  // Degrees lead because nobody judges an arm in radians; rad follows because
  // the threshold and every reason string the server sends are in rad, and an
  // operator comparing the two must not have to convert in their head.
  return QString("%1°  (%2 rad)")
         .arg(j.mismatch * kRadToDeg, 0, 'f', 1)
         .arg(j.mismatch, 0, 'f', 4);
}

std::string withoutTrailingSlash(std::string path)
{
  while (!path.empty() && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

// "/ns/controller_manager" -> "/ns", "/controller_manager" -> "": where the
// manager's controllers (and so their actions) live.
std::string parentNamespace(const std::string & node)
{
  const auto slash = node.rfind('/');
  return slash == std::string::npos ? "" : node.substr(0, slash);
}

// An older config's TrajectoryNamespace,
// "/robot/fullbody_controller/follow_joint_trajectory", only ever served
// to find the controller manager: "/robot/controller_manager".
std::string managerOfTrajectoryAction(const std::string & action)
{
  return parentNamespace(parentNamespace(action)) + "/controller_manager";
}
}  // namespace

ServoPanel::ServoPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout;

  auto * ns_row = new QHBoxLayout;
  ns_row->addWidget(new QLabel("servicer:"));
  ns_edit_ = new QLineEdit(QString::fromStdString(ns_));
  ns_row->addWidget(ns_edit_);
  auto * connect_button = new QPushButton("Connect");
  ns_row->addWidget(connect_button);
  root->addLayout(ns_row);

  auto * manager_row = new QHBoxLayout;
  manager_row->addWidget(new QLabel("controller_manager:"));
  manager_edit_ = new QLineEdit(QString::fromStdString(manager_));
  manager_edit_->setToolTip(
    "The controller manager whose trajectory controllers command these joints.\n"
    "ON first sends each controller that owns a joint being enabled a\n"
    "trajectory to the pose the robot is already in, so it adopts it before\n"
    "torque is applied and nothing jumps.");
  manager_row->addWidget(manager_edit_);
  root->addLayout(manager_row);

  status_label_ = new QLabel("not connected");
  root->addWidget(status_label_);

  // The one thing an operator must see before touching a button: how far the
  // robot has drifted from what the controller commands.
  banner_label_ = new QLabel("");
  banner_label_->setWordWrap(true);
  banner_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  // Fixed height so the warning appearing does not shove every button down the
  // panel. A control that moves when the danger shows up is a control you can
  // misclick exactly when it matters.
  banner_label_->setFixedHeight(58);
  root->addWidget(banner_label_);

  rows_host_ = new QWidget;
  rows_layout_ = new QVBoxLayout(rows_host_);
  rows_layout_->setContentsMargins(0, 0, 0, 0);
  auto * scroll = new QScrollArea;
  rows_scroll_ = scroll;
  scroll->setWidgetResizable(true);
  scroll->setWidget(rows_host_);
  // The joint rows ARE the panel. Docked narrow in RViz they lose the fight for
  // vertical space against the reason box (a QTextEdit expands by default), and
  // a panel that shows one joint of three while the rest hide behind a
  // scrollbar is how an operator releases the wrong limb.
  scroll->setMinimumHeight(180);
  root->addWidget(scroll, 3);

  auto * all_row = new QHBoxLayout;
  all_on_ = new QPushButton("ALL ON");
  all_off_ = new QPushButton("ALL OFF");
  all_row->addWidget(all_on_);
  all_row->addWidget(all_off_);
  root->addLayout(all_row);

  root->addWidget(new QLabel("last answer from the server:"));
  reason_view_ = new QTextEdit;
  reason_view_->setReadOnly(true);
  // The refusal reason is the safety payload -- it names the joint, both
  // numbers and what to do. It gets real space and stays put (not a toast that
  // vanishes), but capped: it must not crowd out the joint rows. Longer reasons
  // scroll inside it.
  reason_view_->setMinimumHeight(80);
  reason_view_->setMaximumHeight(150);
  root->addWidget(reason_view_, 0);

  setMinimumWidth(320);
  setLayout(root);

  connect(connect_button, &QPushButton::clicked, this, &ServoPanel::onConnect);
  connect(ns_edit_, &QLineEdit::returnPressed, this, &ServoPanel::onConnect);
  connect(manager_edit_, &QLineEdit::returnPressed, this, &ServoPanel::onConnect);
  connect(all_on_, &QPushButton::clicked, this, &ServoPanel::onAllOn);
  connect(all_off_, &QPushButton::clicked, this, &ServoPanel::onAllOff);

  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &ServoPanel::onTick);
}

ServoPanel::~ServoPanel() = default;

void ServoPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  onConnect();
  timer_->start(100);
}

void ServoPanel::onConnect()
{
  ns_ = withoutTrailingSlash(ns_edit_->text().toStdString());
  manager_ = withoutTrailingSlash(manager_edit_->text().toStdString());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    have_latest_ = false;
    latest_.reset();
  }
  rows_.clear();
  QLayoutItem * item;
  while ((item = rows_layout_->takeAt(0)) != nullptr) {
    if (item->widget()) {
      item->widget()->deleteLater();
    }
    delete item;
  }

  // transient_local to match the publisher: a panel opened long after the
  // servicer started must learn the state immediately, not sit blank until
  // something changes. A mismatched QoS here silently receives nothing.
  auto qos = rclcpp::QoS(1).reliable().transient_local();
  sub_ = node_->create_subscription<actuo_msgs::msg::ServoState>(
    ns_ + "/servo/state", qos,
    [this](actuo_msgs::msg::ServoState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_ = msg;
      have_latest_ = true;
    });

  client_ = rclcpp_action::create_client<actuo_msgs::action::SetServo>(
    node_, ns_ + "/servo/set");

  controllers_client_ =
    node_->create_client<controller_manager_msgs::srv::ListControllers>(
    manager_ + "/list_controllers");
  resync_clients_.clear();

  status_label_->setText(
    QString::fromStdString("listening on " + ns_ + "/servo/state"));
}

void ServoPanel::report(bool ok, const std::string & text)
{
  std::lock_guard<std::mutex> lock(mutex_);
  pending_ok_ = ok;
  pending_reason_ = text;
  have_pending_ = true;
}

void ServoPanel::rebuildRows(const std::vector<std::string> & names)
{
  rows_.clear();
  QLayoutItem * item;
  while ((item = rows_layout_->takeAt(0)) != nullptr) {
    if (item->widget()) {
      item->widget()->deleteLater();
    }
    delete item;
  }

  auto * grid = new QGridLayout;
  grid->addWidget(new QLabel("<b>joint</b>"), 0, 0);
  grid->addWidget(new QLabel("<b>away from command</b>"), 0, 1);
  grid->addWidget(new QLabel("<b>torque</b>"), 0, 2);
  // Column 1 absorbs the slack so a narrow dock eats into the numbers, never
  // into the button: a torque control pushed off the edge is one an operator
  // cannot reach in the moment they most want it.
  grid->setColumnStretch(1, 1);

  int row = 1;
  for (const auto & name : names) {
    JointRow r;
    r.name = name;
    r.name_label = new QLabel(QString::fromStdString(name));
    r.mismatch_label = new QLabel("-");
    r.mismatch_label->setMinimumWidth(0);
    r.toggle = new QPushButton("-");
    r.toggle->setFixedWidth(74);

    grid->addWidget(r.name_label, row, 0);
    grid->addWidget(r.mismatch_label, row, 1);
    grid->addWidget(r.toggle, row, 2);

    const size_t index = rows_.size();
    connect(r.toggle, &QPushButton::clicked, this, [this, index]() {
        if (index >= rows_.size()) {
          return;
        }
        // Read the state off the row, not off the button's text: the text is
        // for the human, and deciding from it would make a stale repaint able
        // to invert what the click does.
        const auto & clicked = rows_[index];
        if (clicked.enabled) {
          sendGoal(false, {clicked.name});
        } else {
          resyncThenEnable({clicked.name});
        }
      });

    rows_.push_back(r);
    ++row;
  }
  rows_layout_->addLayout(grid);
  rows_layout_->addStretch(1);
}

void ServoPanel::applyState(const actuo_msgs::msg::ServoState & msg)
{
  std::vector<std::string> names;
  names.reserve(msg.joints.size());
  for (const auto & j : msg.joints) {
    names.push_back(j.name);
  }

  std::vector<std::string> current;
  current.reserve(rows_.size());
  for (const auto & r : rows_) {
    current.push_back(r.name);
  }
  if (names != current) {
    rebuildRows(names);
  }

  for (size_t i = 0; i < msg.joints.size() && i < rows_.size(); ++i) {
    const auto & j = msg.joints[i];
    auto & r = rows_[i];
    r.enabled = j.enabled;

    // The button says what the joint IS, and the colour says it twice so it
    // reads at a glance: green = holding, red = limp. A click flips it.
    if (j.enabled) {
      r.toggle->setText("ON");
      r.toggle->setStyleSheet(
        "background-color: #0a7d28; color: white; font-weight: bold;");
      r.toggle->setToolTip(
        QString::fromStdString(j.name + " is holding. Click to release it -- "
        "whatever it is holding up will fall."));
    } else {
      r.toggle->setText("OFF");
      r.toggle->setStyleSheet(
        "background-color: #b00020; color: white; font-weight: bold;");
      r.toggle->setToolTip(
        QString::fromStdString(j.name + " is limp. Click to hold it where it "
        "is now: the controller is re-synced to the robot first, so it will "
        "not jump."));
    }

    // Not answering outranks everything else on the row: its `measured` is the
    // last pose it reported (or NaN), so a mismatch shown beside it would be a
    // number about a servo that is not there.
    if (!j.present) {
      const std::string why = j.absence_reason.empty() ? "no reason given"
                                                       : j.absence_reason;
      r.name_label->setStyleSheet("color: #b00020; font-weight: bold;");
      r.mismatch_label->setText("NOT ANSWERING");
      r.mismatch_label->setStyleSheet("color: #b00020; font-weight: bold;");
      const QString tip = QString::fromStdString(
        j.name + " does not answer: " + why + ". Check its cable and power. "
        "ON is refused until it answers.");
      r.name_label->setToolTip(tip);
      r.mismatch_label->setToolTip(tip);
      r.toggle->setToolTip(tip);
    } else if (j.error != 0) {
      r.name_label->setStyleSheet("");
      r.name_label->setToolTip("");
      r.mismatch_label->setToolTip("");
      r.mismatch_label->setText(QString::fromStdString("FAULT: " + j.error_name));
      r.mismatch_label->setStyleSheet("color: #b00020; font-weight: bold;");
    } else {
      r.name_label->setStyleSheet("");
      r.name_label->setToolTip("");
      r.mismatch_label->setToolTip("");
      r.mismatch_label->setText(jointSummary(j));
      // would_jump comes from the server. This colours it; it does not compute
      // it. Only meaningful while limp -- a held joint is already there.
      if (j.would_jump && !j.enabled) {
        r.mismatch_label->setStyleSheet("color: #b00020; font-weight: bold;");
      } else {
        r.mismatch_label->setStyleSheet("");
      }
    }

    // Never greyed, even when would_jump is true. A dead control explains
    // nothing; let them press it and read the server's answer.
    r.toggle->setEnabled(true);
  }

  // Absent joints are judged first and kept out of the jump banner: a servo
  // that is not there reads as "moved away from its command" (its last pose
  // vs the command), which would send the operator after the wrong remedy.
  std::vector<std::string> missing;
  bool present_would_jump = false;
  for (const auto & j : msg.joints) {
    if (!j.present) {
      missing.push_back(j.name);
    } else if (j.would_jump) {
      present_would_jump = true;
    }
  }
  if (missing != shown_missing_) {
    shown_missing_ = missing;
    scroll_to_ = missing.empty() ? "" : missing.front();
  }
  // Scrolled once the row has been laid out: a freshly built row, or one in a
  // panel not shown yet, has no geometry and the scroll would go nowhere.
  // applyState runs every tick, so this simply waits for that.
  if (!scroll_to_.empty() && rows_scroll_ && rows_scroll_->isVisible()) {
    for (const auto & r : rows_) {
      if (r.name == scroll_to_ && r.name_label->height() > 0 &&
        r.name_label->mapTo(rows_scroll_->widget(), QPoint(0, 0)).y() > 0)
      {
        rows_scroll_->ensureWidgetVisible(r.name_label);
        scroll_to_.clear();
      }
    }
  }

  if (!missing.empty()) {
    std::string names;
    for (const auto & n : missing) {
      names += (names.empty() ? "" : ", ") + n;
    }
    // Names only; the reasons are on each row's tooltip. The banner has a
    // fixed height and a cut-off safety line is worse than a terse one.
    banner_label_->setText(
      QString("<b style='color:#b00020'>Not answering (%1): %2</b><br>"
              "Check cable / power. ON is refused until it answers.")
      .arg(missing.size()).arg(QString::fromStdString(names).toHtmlEscaped()));
  } else if (present_would_jump) {
    // NB what red means now. It does NOT mean "pressing this throws the joint":
    // the button re-syncs the controller first, so it will not. It means the
    // robot has been moved away from what the controller still commands -- the
    // danger a client that enables WITHOUT re-syncing would hit, and which the
    // servicer refuses. It must not say "re-sync it yourself": that would be
    // false, since the button does the re-sync.
    // Kept short enough to fit the fixed height at a narrow dock. It is cut
    // off, not scrolled, if it overflows -- and a safety line that ends
    // mid-sentence is worse than a terse one.
    banner_label_->setText(
      QString("<b style='color:#b00020'>Red: moved &ge;%1 rad from the "
              "controller's command.</b> OFF&rarr;ON re-syncs the controller "
              "first, so the joint holds where it is. No jump.")
      .arg(msg.jump_threshold, 0, 'f', 2));
  } else if (msg.all_enabled) {
    banner_label_->setText("<b style='color:#0a7d28'>All joints held.</b>");
  } else {
    banner_label_->setText("Every joint is within its command.");
  }
}

void ServoPanel::sendGoal(bool enable, const std::vector<std::string> & joints)
{
  if (!client_ || !client_->action_server_is_ready()) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ok_ = false;
    pending_reason_ = "no servo action server at " + ns_ +
      "/servo/set -- is the bus servicer running?";
    have_pending_ = true;
    return;
  }

  actuo_msgs::action::SetServo::Goal goal;
  goal.enable = enable;
  goal.joints = joints;   // empty = every joint, as the action promises
  goal.force = false;     // never from the panel: an override must be a
                          // deliberate, greppable act, not a stray click.

  auto options =
    rclcpp_action::Client<actuo_msgs::action::SetServo>::SendGoalOptions();
  options.result_callback =
    [this](const rclcpp_action::ClientGoalHandle<
        actuo_msgs::action::SetServo>::WrappedResult & result) {
      std::lock_guard<std::mutex> lock(mutex_);
      have_pending_ = true;
      if (result.result) {
        pending_ok_ = result.result->ok;
        pending_reason_ = result.result->reason;
      } else {
        pending_ok_ = false;
        pending_reason_ = "no result from the server";
      }
    };
  // A refusal is an ABORTED goal, so it arrives with no result in some
  // transports; catch the rejection path too rather than leaving the operator
  // staring at a stale reason.
  options.goal_response_callback =
    [this](const rclcpp_action::ClientGoalHandle<
        actuo_msgs::action::SetServo>::SharedPtr & handle) {
      if (!handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ok_ = false;
        pending_reason_ = "the server rejected the goal outright";
        have_pending_ = true;
      }
    };

  client_->async_send_goal(goal, options);
}

void ServoPanel::resyncThenEnable(const std::vector<std::string> & joints)
{
  actuo_msgs::msg::ServoState::SharedPtr state;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    state = latest_;
  }
  if (!state) {
    report(false, "no servo state yet -- cannot tell where the robot is, so "
      "nothing was sent");
    return;
  }

  // Empty selection means every joint, matching the action's contract.
  std::vector<std::string> want = joints;
  if (want.empty()) {
    for (const auto & j : state->joints) {
      want.push_back(j.name);
    }
  }

  // Only joints that are actually limp are re-seeded to where they are.
  // Commanding a held joint to its own measured position would drag it off
  // whatever trajectory it is running, for no benefit -- it is already where
  // the controller wants it.
  std::vector<std::string> names;
  for (const auto & j : state->joints) {
    if (std::find(want.begin(), want.end(), j.name) == want.end()) {
      continue;
    }
    if (j.enabled) {
      continue;
    }
    names.push_back(j.name);
  }

  if (names.empty()) {
    sendGoal(true, joints);   // nothing limp to re-seed; let the server answer
    return;
  }

  // WHICH controller owns which joint is asked of controller_manager, not
  // guessed. A robot may have several trajectory controllers (head, each arm), and
  // a goal naming joints a controller does not own is not a partial goal -- it
  // is rejected outright, which would show up as "torque on does nothing"
  // in the panel.
  // Where each joint goes in the re-sync trajectory: a joint being enabled to
  // where it is, every other joint to what it is commanded now, so its
  // setpoint does not change. A command that is not a number (no controller
  // owns the slot) cannot be kept, and that joint is put where it is.
  std::map<std::string, double> targets;
  for (const auto & j : state->joints) {
    const bool enabling =
      std::find(names.begin(), names.end(), j.name) != names.end();
    targets[j.name] =
      (enabling || !std::isfinite(j.commanded)) ? j.measured : j.commanded;
  }
  resyncWithControllers(joints, names, targets);
}

void ServoPanel::resyncWithControllers(
  const std::vector<std::string> & selection,
  const std::vector<std::string> & limp,
  const std::map<std::string, double> & targets)
{
  if (!controllers_client_ || !controllers_client_->service_is_ready()) {
    report(false, "no controller_manager at " + manager_ +
      " -- cannot tell which controller owns which joint, so nothing was "
      "re-synced and torque was NOT enabled. Set controller_manager and press "
      "Connect.");
    return;
  }

  auto request = std::make_shared<
    controller_manager_msgs::srv::ListControllers::Request>();
  const std::vector<std::string> limp_copy = limp;
  const std::map<std::string, double> targets_copy = targets;
  const std::vector<std::string> selection_copy = selection;
  const std::string prefix = parentNamespace(manager_) + "/";
  controllers_client_->async_send_request(
    request,
    [this, limp_copy, targets_copy, selection_copy, prefix](
      rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedFuture
      future) {
      std::map<std::string, std::vector<std::string>> by_controller;
      for (const auto & controller : future.get()->controller) {
        if (controller.state != "active") {
          continue;
        }
        // Every joint the controller commands, in its order:
        // "rarm_joint0/position" -- the joint is everything before the slash.
        std::vector<std::string> owned;
        bool enables_one = false;
        for (const auto & claimed : controller.claimed_interfaces) {
          const auto slash = claimed.find('/');
          if (slash == std::string::npos) {
            continue;
          }
          const std::string joint = claimed.substr(0, slash);
          if (std::find(owned.begin(), owned.end(), joint) == owned.end()) {
            owned.push_back(joint);
          }
          if (std::find(limp_copy.begin(), limp_copy.end(), joint) !=
            limp_copy.end())
          {
            enables_one = true;
          }
        }
        if (!enables_one) {
          continue;
        }
        for (const auto & joint : owned) {
          if (targets_copy.count(joint) == 0) {
            report(false, controller.name + " commands " + joint + ", which "
              "the servicer does not report, so there is no pose to re-sync "
              "it to. Torque was NOT enabled.");
            return;
          }
        }
        by_controller[prefix + controller.name + "/follow_joint_trajectory"] = owned;
      }
      if (by_controller.empty()) {
        report(false, "no active controller claims any of the joints being "
          "enabled, so there was nothing to re-sync. Torque was NOT enabled.");
        return;
      }
      sendResyncGoals(by_controller, targets_copy, selection_copy);
    });
}

void ServoPanel::sendResyncGoals(
  const std::map<std::string, std::vector<std::string>> & by_controller,
  const std::map<std::string, double> & targets,
  const std::vector<std::string> & selection)
{
  // One counter for the whole re-sync. Torque goes on when the LAST goal
  // succeeds, not the first: enabling earlier would leave the other
  // controllers still commanding where the robot used to be, which is the
  // jump this whole path exists to prevent.
  resync_pending_ = std::make_shared<std::atomic<int>>(
    static_cast<int>(by_controller.size()));
  const auto total = by_controller.size();

  for (const auto & entry : by_controller) {
    const std::string & action = entry.first;
    auto & client = resync_clients_[action];
    if (!client) {
      client = rclcpp_action::create_client<FollowJointTrajectory>(
        node_, action);
    }
    if (!client->action_server_is_ready()) {
      report(false, "no trajectory controller at " + action +
        " -- cannot re-sync it, so torque was NOT enabled. Nothing moved.");
      return;
    }

    FollowJointTrajectory::Goal goal;
    trajectory_msgs::msg::JointTrajectoryPoint point;
    for (const auto & joint : entry.second) {
      goal.trajectory.joint_names.push_back(joint);
      point.positions.push_back(targets.at(joint));
    }
    point.time_from_start = rclcpp::Duration::from_seconds(0.5);
    goal.trajectory.points.push_back(point);

    auto options =
      rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
    auto pending = resync_pending_;
    const std::vector<std::string> selection_copy = selection;
    options.result_callback =
      [this, pending, selection_copy, action, total](
        const rclcpp_action::ClientGoalHandle<
          FollowJointTrajectory>::WrappedResult & result) {
        if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
          *pending = -1;
          report(false, action + " would not adopt the robot's pose, so "
            "torque was NOT enabled. Nothing moved.");
          return;
        }
        if (--(*pending) == 0) {
          // Every controller now commands where the robot is.
          sendGoal(true, selection_copy);
        }
      };
    options.goal_response_callback =
      [this, pending, action](
        const rclcpp_action::ClientGoalHandle<
          FollowJointTrajectory>::SharedPtr & handle) {
        if (!handle) {
          *pending = -1;
          report(false, action + " rejected the re-sync trajectory; torque was "
            "NOT enabled. Nothing moved.");
        }
      };

    // Nothing moves here: the target IS the current pose, and torque is off
    // anyway. The only effect is on the controller's setpoint.
    client->async_send_goal(goal, options);
  }
}

void ServoPanel::onAllOn()
{
  resyncThenEnable({});
}

void ServoPanel::onAllOff()
{
  sendGoal(false, {});
}

void ServoPanel::onTick()
{
  actuo_msgs::msg::ServoState::SharedPtr msg;
  bool reason_ready = false;
  bool ok = false;
  std::string reason;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (have_latest_) {
      msg = latest_;
    }
    if (have_pending_) {
      reason_ready = true;
      ok = pending_ok_;
      reason = pending_reason_;
      have_pending_ = false;
    }
  }

  if (msg) {
    applyState(*msg);
    status_label_->setText(
      QString::fromStdString(ns_) +
      QString(" — %1 joints").arg(static_cast<int>(msg->joints.size())));
  }

  if (reason_ready) {
    reason_view_->setStyleSheet(
      ok ? "color: #0a7d28;" : "color: #b00020; font-weight: bold;");
    reason_view_->setPlainText(
      QString::fromStdString((ok ? "OK: " : "REFUSED: ") + reason));
  }
}

void ServoPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("Namespace", QString::fromStdString(ns_));
  config.mapSetValue("ControllerManager", QString::fromStdString(manager_));
}

void ServoPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  QString ns;
  if (config.mapGetString("Namespace", &ns)) {
    ns_ = ns.toStdString();
    ns_edit_->setText(ns);
  }
  QString manager;
  QString traj_ns;
  if (config.mapGetString("ControllerManager", &manager)) {
    manager_ = manager.toStdString();
  } else if (config.mapGetString("TrajectoryNamespace", &traj_ns)) {
    manager_ = managerOfTrajectoryAction(traj_ns.toStdString());
  }
  manager_edit_->setText(QString::fromStdString(manager_));
}

}  // namespace actuo_rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(actuo_rviz_plugins::ServoPanel, rviz_common::Panel)
