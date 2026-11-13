#include "actuo_rviz_plugins/diagnostics_panel.hpp"

#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <QAbstractItemView>
#include <QBrush>
#include <QColor>
#include <QFont>
#include <QHeaderView>
#include <QHBoxLayout>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace actuo_rviz_plugins
{

namespace
{

constexpr int kRoleDetails = Qt::UserRole + 1;
constexpr int kRoleKey = Qt::UserRole + 2;

struct LevelVisual
{
  QString label;
  QColor foreground;
  QColor background;
  int rank;
};

struct ParsedStatus
{
  const diagnostic_msgs::msg::DiagnosticStatus * status{nullptr};
  std::string publisher;
  std::string item_name;
  std::string group;
  int rank{0};
  double age_sec{0.0};
  bool stale{false};
};

std::pair<std::string, std::string> splitPublisherAndItem(const std::string & raw_name)
{
  // diagnostic_updater sends "node_name: item_name". Showing the raw string
  // merges publisher and item, which makes cross-node diagnosis unreadable.
  const std::string delimiter = ": ";
  const std::size_t pos = raw_name.find(delimiter);
  if (pos == std::string::npos) {
    return {"", raw_name};
  }
  return {raw_name.substr(0, pos), raw_name.substr(pos + delimiter.size())};
}

std::string firstPathSegment(std::string item_name)
{
  while (!item_name.empty() && item_name.front() == '/') {
    item_name.erase(item_name.begin());
  }
  if (item_name.empty()) {
    return "ungrouped";
  }
  const std::size_t slash = item_name.find('/');
  if (slash == std::string::npos) {
    return item_name;
  }
  return item_name.substr(0, slash);
}

LevelVisual levelVisual(uint8_t level)
{
  switch (level) {
    case diagnostic_msgs::msg::DiagnosticStatus::OK:
      return {"OK", QColor("#14532d"), QColor("#dcfce7"), 1};
    case diagnostic_msgs::msg::DiagnosticStatus::WARN:
      return {"WARN", QColor("#78350f"), QColor("#fef3c7"), 2};
    case diagnostic_msgs::msg::DiagnosticStatus::ERROR:
      return {"ERROR", QColor("#ffffff"), QColor("#b00020"), 4};
    case diagnostic_msgs::msg::DiagnosticStatus::STALE:
      // STALE is intentionally vivid blue, not grey: no data is a fault state.
      return {"STALE", QColor("#ffffff"), QColor("#1d4ed8"), 3};
    default:
      return {
        QString("UNKNOWN(%1)").arg(static_cast<int>(level)),
        QColor("#ffffff"),
        QColor("#4b5563"),
        0
      };
  }
}

QString detailsText(const ParsedStatus & parsed)
{
  std::ostringstream out;
  out << "publisher: "
      << (parsed.publisher.empty() ? "(no publisher prefix)" : parsed.publisher) << "\n";
  out << "item: " << parsed.item_name << "\n";
  out << "level: " << levelVisual(parsed.status->level).label.toStdString();
  if (parsed.stale) {
    // The reported level is kept visible: a row can be STALE *and* have last
    // said ERROR, and hiding the latter would lose why it mattered.
    out << "  (row shown as STALE; this is its last reported level)";
  }
  out << "\n";
  out << "last update: " << std::fixed << std::setprecision(1) << parsed.age_sec
      << " s ago\n";
  if (!parsed.status->hardware_id.empty()) {
    out << "hardware_id: " << parsed.status->hardware_id << "\n";
  }
  out << "\nmessage:\n" << parsed.status->message << "\n";
  out << "\nvalues:\n";
  if (parsed.status->values.empty()) {
    out << "(none)\n";
  } else {
    for (const auto & pair : parsed.status->values) {
      out << pair.key << ": " << pair.value << "\n";
    }
  }
  return QString::fromStdString(out.str());
}

int worstRank(const std::vector<ParsedStatus> & rows)
{
  int worst = 0;
  for (const auto & row : rows) {
    worst = std::max(worst, row.rank);
  }
  return worst;
}

}  // namespace

DiagnosticsPanel::DiagnosticsPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout;

  auto * stale_row = new QHBoxLayout;
  stale_row->addWidget(new QLabel("stale after:"));
  stale_after_spin_ = new QDoubleSpinBox;
  stale_after_spin_->setRange(0.5, 60.0);
  stale_after_spin_->setSingleStep(0.5);
  stale_after_spin_->setDecimals(1);
  stale_after_spin_->setSuffix(" s");
  stale_after_spin_->setValue(stale_after_sec_);
  stale_after_spin_->setToolTip(
    "Panel timeout for '/diagnostics'. Default 2.5 s comes from the servicer's\n"
    "default diagnostics-rate 1 Hz: one period plus tolerance for missed ticks.");
  stale_row->addWidget(stale_after_spin_);
  stale_row->addStretch(1);
  root->addLayout(stale_row);

  status_ = new QLabel("waiting for /diagnostics");
  root->addWidget(status_);
  age_ = new QLabel("last message age: --");
  root->addWidget(age_);

  tree_ = new QTreeWidget;
  tree_->setColumnCount(4);
  tree_->setHeaderLabels({"Level", "Item", "Message", "Publisher"});
  tree_->setRootIsDecorated(true);
  tree_->setAlternatingRowColors(true);
  tree_->setUniformRowHeights(true);
  tree_->setSelectionBehavior(QAbstractItemView::SelectRows);
  tree_->setSelectionMode(QAbstractItemView::SingleSelection);
  tree_->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  tree_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  tree_->header()->setSectionResizeMode(2, QHeaderView::Stretch);
  tree_->header()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
  // The rows ARE the panel. Without a floor the details pane's own minimum wins
  // whenever the dock is cramped, and the tree collapses to two visible rows --
  // which on a robot with many servos means the bus group fills it and not one servo row is
  // reachable without scrolling, i.e. the fault you opened the panel for.
  tree_->setMinimumHeight(220);
  root->addWidget(tree_, 3);

  root->addWidget(new QLabel("details (selected row values):"));
  details_ = new QTextEdit;
  details_->setReadOnly(true);
  details_->setLineWrapMode(QTextEdit::NoWrap);
  details_->setMinimumHeight(90);
  details_->setPlainText("Select a diagnostic row to view key/value details.");
  {
    QFont mono("monospace");
    mono.setStyleHint(QFont::TypeWriter);
    details_->setFont(mono);
  }
  root->addWidget(details_, 1);

  setMinimumWidth(520);
  setLayout(root);

  connect(
    stale_after_spin_, qOverload<double>(&QDoubleSpinBox::valueChanged),
    this, &DiagnosticsPanel::onStaleAfterEdited);
  connect(tree_, &QTreeWidget::itemSelectionChanged, this, &DiagnosticsPanel::onSelectionChanged);

  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &DiagnosticsPanel::onTick);
}

void DiagnosticsPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  sub_ = node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
    topic_, rclcpp::QoS(10).reliable(),
    [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      // Merge by item name; never replace the whole table. See DiagnosticEntry.
      for (const auto & status : msg->status) {
        auto & entry = items_[status.name];
        entry.status = status;
        entry.last_seen = now;
      }
      last_receive_steady_ = now;
      ever_received_ = true;
      ++generation_;
    });

  timer_->start(200);
}

void DiagnosticsPanel::onTick()
{
  std::map<std::string, DiagnosticEntry> snapshot;
  std::chrono::steady_clock::time_point last_receive;
  std::size_t generation = 0;
  bool received = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    received = ever_received_;
    if (received) {
      snapshot = items_;
      last_receive = last_receive_steady_;
      generation = generation_;
    }
  }

  if (!received) {
    setViewFreshness(true, false, 0.0);
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const double age_sec = std::chrono::duration<double>(now - last_receive).count();
  // Panel-wide: has the TOPIC gone quiet. With several publishers this can stay
  // green while one of them dies, which is why each row also ages on its own
  // clock below -- a live controller_manager must not vouch for a dead bus.
  const bool stale = age_sec > stale_after_sec_;
  setViewFreshness(false, stale, age_sec);

  bool staleness_changed = false;
  for (const auto & entry : snapshot) {
    const double item_age =
      std::chrono::duration<double>(now - entry.second.last_seen).count();
    if ((item_age > stale_after_sec_) != entry.second.rendered_stale) {
      staleness_changed = true;
      break;
    }
  }

  if (!rendered_any_ || generation != rendered_generation_ || staleness_changed) {
    render(snapshot, now);
    rendered_any_ = true;
    rendered_generation_ = generation;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & entry : items_) {
      const double item_age =
        std::chrono::duration<double>(now - entry.second.last_seen).count();
      entry.second.rendered_stale = item_age > stale_after_sec_;
    }
  }
}

void DiagnosticsPanel::onSelectionChanged()
{
  showDetailsFor(tree_->currentItem());
}

void DiagnosticsPanel::onStaleAfterEdited(double value)
{
  stale_after_sec_ = value;
  onTick();
}

void DiagnosticsPanel::setViewFreshness(bool waiting, bool stale, double age_sec)
{
  if (waiting) {
    status_->setText("waiting for /diagnostics");
    status_->setStyleSheet("color: #d97706; font-weight: bold;");
    age_->setText("last message age: --");
    tree_->setStyleSheet("");
    details_->setStyleSheet("");
    return;
  }

  age_->setText(QString("last message age: %1 s").arg(age_sec, 0, 'f', 1));
  if (stale) {
    status_->setText("diagnostics have stopped arriving on /diagnostics");
    status_->setStyleSheet("color: #ef4444; font-weight: bold;");
    // A stale panel must look stale at a glance, or old rows read as live.
    // Forcing a light background WITHOUT a matching foreground leaves the text
    // at the theme's default, which under RViz's dark theme is near-white --
    // i.e. white on pale pink, unreadable. Whenever one is set, set both.
    tree_->setStyleSheet(
      "QTreeWidget { background-color: #fff1f2; color: #1f2937; }");
    details_->setStyleSheet("background-color: #fff1f2; color: #1f2937;");
  } else {
    status_->setText("diagnostics are current");
    status_->setStyleSheet("color: #16a34a; font-weight: bold;");
    tree_->setStyleSheet("");
    details_->setStyleSheet("");
  }
}

void DiagnosticsPanel::showDetailsFor(QTreeWidgetItem * item)
{
  if (!item || item->childCount() > 0) {
    details_->setPlainText("Select a diagnostic row to view key/value details.");
    return;
  }
  details_->setPlainText(item->data(0, kRoleDetails).toString());
}

void DiagnosticsPanel::render(
  const std::map<std::string, DiagnosticEntry> & items,
  std::chrono::steady_clock::time_point now)
{
  QString selected_key;
  if (const auto * current = tree_->currentItem()) {
    if (current->childCount() == 0) {
      selected_key = current->data(0, kRoleKey).toString();
    }
  }

  std::map<std::string, std::vector<ParsedStatus>> groups;
  for (const auto & entry : items) {
    const auto & status = entry.second.status;
    const auto split = splitPublisherAndItem(status.name);
    ParsedStatus parsed;
    parsed.status = &status;
    parsed.publisher = split.first;
    parsed.item_name = split.second;
    parsed.group = firstPathSegment(parsed.item_name);
    parsed.age_sec =
      std::chrono::duration<double>(now - entry.second.last_seen).count();
    parsed.stale = parsed.age_sec > stale_after_sec_;
    // A row nobody is updating any more sorts and paints as STALE whatever it
    // last claimed: "OK, 40 seconds ago" is not OK.
    parsed.rank = parsed.stale
      ? levelVisual(diagnostic_msgs::msg::DiagnosticStatus::STALE).rank
      : levelVisual(status.level).rank;
    groups[parsed.group].push_back(parsed);
  }

  std::vector<std::string> group_names;
  group_names.reserve(groups.size());
  for (const auto & entry : groups) {
    group_names.push_back(entry.first);
  }
  std::sort(group_names.begin(), group_names.end(), [&](const std::string & a, const std::string & b) {
    if (a == "bus" && b != "bus") {
      return true;
    }
    if (b == "bus" && a != "bus") {
      return false;
    }
    const int a_worst = worstRank(groups.at(a));
    const int b_worst = worstRank(groups.at(b));
    if (a_worst != b_worst) {
      return a_worst > b_worst;
    }
    return a < b;
  });

  tree_->clear();
  QTreeWidgetItem * first_leaf = nullptr;
  QTreeWidgetItem * matched_leaf = nullptr;
  for (const auto & group_name : group_names) {
    auto rows = groups.at(group_name);
    std::sort(rows.begin(), rows.end(), [](const ParsedStatus & a, const ParsedStatus & b) {
      if (a.rank != b.rank) {
        return a.rank > b.rank;
      }
      return a.item_name < b.item_name;
    });

    auto * group_item = new QTreeWidgetItem(tree_);
    group_item->setText(
      0, QString::fromStdString(group_name) + QString(" (%1)").arg(static_cast<int>(rows.size())));
    QFont bold = group_item->font(0);
    bold.setBold(true);
    group_item->setFont(0, bold);
    group_item->setFirstColumnSpanned(true);
    // Both colours, explicitly. The leaf rows set both, so a group header that
    // set neither inherited the theme's foreground and vanished the moment the
    // tree's background was repainted for the stale state.
    for (int col = 0; col < 4; ++col) {
      group_item->setForeground(col, QBrush(QColor("#111827")));
      group_item->setBackground(col, QBrush(QColor("#e5e7eb")));
    }

    for (const auto & row : rows) {
      auto * item = new QTreeWidgetItem(group_item);
      const LevelVisual visual = row.stale
        ? levelVisual(diagnostic_msgs::msg::DiagnosticStatus::STALE)
        : levelVisual(row.status->level);
      const QString publisher = row.publisher.empty()
        ? "(no publisher prefix)"
        : QString::fromStdString(row.publisher);
      item->setText(0, visual.label);
      item->setText(1, QString::fromStdString(row.item_name));
      item->setText(
        2, row.stale
        ? QString("no update for %1 s -- last said: %2")
        .arg(row.age_sec, 0, 'f', 1).arg(QString::fromStdString(row.status->message))
        : QString::fromStdString(row.status->message));
      item->setText(3, publisher);
      for (int col = 0; col < 4; ++col) {
        item->setForeground(col, QBrush(visual.foreground));
        item->setBackground(col, QBrush(visual.background));
      }

      const QString key = publisher + "|" + QString::fromStdString(row.item_name);
      item->setData(0, kRoleKey, key);
      item->setData(0, kRoleDetails, detailsText(row));
      if (!first_leaf) {
        first_leaf = item;
      }
      if (!selected_key.isEmpty() && key == selected_key) {
        matched_leaf = item;
      }
    }
    group_item->setExpanded(true);
  }

  if (!first_leaf) {
    details_->setPlainText("No diagnostic rows have been received on /diagnostics.");
    return;
  }

  tree_->setCurrentItem(matched_leaf ? matched_leaf : first_leaf);
  showDetailsFor(tree_->currentItem());
}

void DiagnosticsPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("StaleAfterSec", stale_after_sec_);
}

void DiagnosticsPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  float stale_after_sec = 0.0F;
  if (config.mapGetFloat("StaleAfterSec", &stale_after_sec)) {
    stale_after_sec_ = std::max(0.5, static_cast<double>(stale_after_sec));
    stale_after_spin_->setValue(stale_after_sec_);
  }
}

}  // namespace actuo_rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(actuo_rviz_plugins::DiagnosticsPanel, rviz_common::Panel)
