#include "shodhan/mip/node_tree.hpp"

#include <algorithm>

namespace shodhan::mip {

NodeTree::NodeTree(double root_lower_bound) {
  Node root;
  root.id = 0;
  root.lower_bound = root_lower_bound;
  slots_.push_back(std::move(root));
  slot_of_.push_back(0);
}

NodeId NodeTree::add_child(NodeId parent, Index col, int dir, double value, double lower_bound, double estimate, BoundChange change,
                           std::shared_ptr<const BasisSnapshot> basis) {
  Node n;
  n.id = static_cast<NodeId>(slot_of_.size());
  n.parent = parent;
  n.depth = at(parent).depth + 1;
  n.branch_col = col;
  n.branch_dir = dir;
  n.branch_value = value;
  n.lower_bound = lower_bound;
  n.estimate = estimate;
  n.local.push_back(change);
  n.basis = std::move(basis);
  max_depth_ = std::max(max_depth_, n.depth);
  ++at(parent).live_children;
  std::int32_t slot;
  if (!free_.empty()) {
    slot = free_.back();
    free_.pop_back();
    slots_[static_cast<std::size_t>(slot)] = std::move(n);
  } else {
    slot = static_cast<std::int32_t>(slots_.size());
    slots_.push_back(std::move(n));
  }
  const NodeId id = static_cast<NodeId>(slot_of_.size());
  slot_of_.push_back(slot);
  peak_live_ = std::max(peak_live_, live());
  return id;
}

void NodeTree::finish(NodeId id) {
  if (!alive(id)) return;
  at(id).done = true;
  if (at(id).live_children == 0) release(id);
}

void NodeTree::release(NodeId id) {
  // Release the node, then its parent if that one is finished and has no other live child, and so on.
  while (id != kNoNode && alive(id)) {
    Node& n = at(id);
    if (!n.done || n.live_children != 0) return;
    const NodeId parent = n.parent;
    const std::int32_t slot = slot_of_[static_cast<std::size_t>(id)];
    slots_[static_cast<std::size_t>(slot)] = Node{};  // frees the bound changes and the basis
    slot_of_[static_cast<std::size_t>(id)] = -1;
    free_.push_back(slot);
    if (parent == kNoNode || !alive(parent)) return;
    --at(parent).live_children;
    id = parent;
  }
}

bool NodeTree::path_bounds(NodeId id, std::vector<double>& lo, std::vector<double>& hi) const {
  // Collect the chain, then apply it from the root down.
  std::vector<NodeId> chain;
  for (NodeId k = id; k != kNoNode; k = at(k).parent) chain.push_back(k);
  bool ok = true;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    for (const BoundChange& c : at(*it).local) {
      double& l = lo[static_cast<std::size_t>(c.col)];
      double& h = hi[static_cast<std::size_t>(c.col)];
      l = std::max(l, c.lo);
      h = std::min(h, c.hi);
      if (l > h) ok = false;
    }
  }
  return ok;
}

}  // namespace shodhan::mip
