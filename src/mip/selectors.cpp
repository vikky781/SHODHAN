// Node selectors: best bound, depth first, best estimate. All break ties by the smaller node id.

#include <set>
#include <utility>
#include <vector>

#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

class BestBoundSelector final : public NodeSelector {
 public:
  const char* name() const override { return "bestbound"; }
  void push(const Node& n) override { set_.insert({n.lower_bound, n.id}); }
  NodeId pop() override {
    if (set_.empty()) return kNoNode;
    const NodeId id = set_.begin()->second;
    set_.erase(set_.begin());
    return id;
  }
  bool empty() const override { return set_.empty(); }
  std::size_t size() const override { return set_.size(); }

 private:
  std::set<std::pair<double, NodeId>> set_;
};

class BestEstimateSelector final : public NodeSelector {
 public:
  const char* name() const override { return "bestestimate"; }
  void push(const Node& n) override { set_.insert({n.estimate, n.id}); }
  NodeId pop() override {
    if (set_.empty()) return kNoNode;
    const NodeId id = set_.begin()->second;
    set_.erase(set_.begin());
    return id;
  }
  bool empty() const override { return set_.empty(); }
  std::size_t size() const override { return set_.size(); }

 private:
  std::set<std::pair<double, NodeId>> set_;
};

class DepthFirstSelector final : public NodeSelector {
 public:
  const char* name() const override { return "depth"; }
  void push(const Node& n) override { stack_.push_back(n.id); }
  NodeId pop() override {
    if (stack_.empty()) return kNoNode;
    const NodeId id = stack_.back();
    stack_.pop_back();
    return id;
  }
  bool empty() const override { return stack_.empty(); }
  std::size_t size() const override { return stack_.size(); }

 private:
  std::vector<NodeId> stack_;
};

}  // namespace

void register_selectors(PluginRegistry& reg) {
  reg.selectors.add("bestbound", [] { return std::make_unique<BestBoundSelector>(); });
  reg.selectors.add("bestestimate", [] { return std::make_unique<BestEstimateSelector>(); });
  reg.selectors.add("depth", [] { return std::make_unique<DepthFirstSelector>(); });
}

}  // namespace shodhan::mip
