// RViz panel for /diagnostics.
//
// Read-only by design: it renders DiagnosticArray and never offers controls.
// The node that owns diagnostics decides what is healthy; this panel only
// separates publisher vs item name, highlights levels, and shows key/value
// details so the UI cannot diverge from the backend's own report.

#ifndef ACTUO_RVIZ_PLUGINS__DIAGNOSTICS_PANEL_HPP_
#define ACTUO_RVIZ_PLUGINS__DIAGNOSTICS_PANEL_HPP_

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <QDoubleSpinBox>
#include <QLabel>
#include <QTextEdit>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWidget>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>

namespace actuo_rviz_plugins
{

/// One remembered diagnostic item, with the clock that decides if it is stale.
///
/// Held per item name rather than per message because ``/diagnostics`` is a
/// shared topic: every node publishes an array of only ITS OWN statuses, at its
/// own rate. Keeping just the last array received therefore shows whichever
/// node published most recently and hides all the others -- e.g. the
/// controller_manager's rows would replace the bus and per-servo rows,
/// and the panel would look healthy while showing none of the hardware.
struct DiagnosticEntry
{
  diagnostic_msgs::msg::DiagnosticStatus status;
  std::chrono::steady_clock::time_point last_seen{};
  bool rendered_stale{false};
};

class DiagnosticsPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit DiagnosticsPanel(QWidget * parent = nullptr);
  ~DiagnosticsPanel() override = default;

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private Q_SLOTS:
  void onTick();
  void onSelectionChanged();
  void onStaleAfterEdited(double value);

private:
  void render(
    const std::map<std::string, DiagnosticEntry> & items,
    std::chrono::steady_clock::time_point now);
  void setViewFreshness(bool waiting, bool stale, double age_sec);
  void showDetailsFor(QTreeWidgetItem * item);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr sub_;
  QTimer * timer_{nullptr};

  QDoubleSpinBox * stale_after_spin_{nullptr};
  QLabel * status_{nullptr};
  QLabel * age_{nullptr};
  QTreeWidget * tree_{nullptr};
  QTextEdit * details_{nullptr};

  std::mutex mutex_;
  std::map<std::string, DiagnosticEntry> items_;
  std::chrono::steady_clock::time_point last_receive_steady_{};
  std::size_t generation_{0};
  bool ever_received_{false};

  std::size_t rendered_generation_{0};
  bool rendered_any_{false};
  double stale_after_sec_{2.5};
  const std::string topic_{"/diagnostics"};
};

}  // namespace actuo_rviz_plugins

#endif  // ACTUO_RVIZ_PLUGINS__DIAGNOSTICS_PANEL_HPP_
