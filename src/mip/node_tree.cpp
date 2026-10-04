#include "shodhan/mip/node_tree.hpp"

#include <algorithm>

namespace shodhan::mip {

NodeTree::NodeTree(double root_lower_bound) {
  Node root;
  root.id = 0;
  root.lower_bound = root_lower_bound;
  nodes_.push_back(std::move(root));
}

NodeId NodeTree::add_child(NodeId parent, Index col, int dir, double value, double lower_bound, double estimate, BoundChange change,
                           std::shared_ptr<const BasisSnapshot> basis) {
  Node n;
  n.id = static_cast<NodeId>(nodes_.size());
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
  nodes_.push_back(std::move(n));
  return nodes_.back().id;
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
