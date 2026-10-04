#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "shodhan/simplex_engine.hpp"
#include "shodhan/sparse_matrix.hpp"

namespace shodhan::mip {

using NodeId = std::int64_t;
constexpr NodeId kNoNode = -1;

/// New bounds of a column (they replace the current ones inside the node's subtree; the tree intersects them
/// with what the ancestors set).
struct BoundChange {
  Index col = 0;
  double lo = 0.0;
  double hi = 0.0;
};

/// One node of the branch-and-bound tree. A node does not store the bounds of its subproblem: they are
/// reconstructed from the chain of `local` changes up to the root. The bound changes of a node are the
/// branching change that created it plus the tightenings found later at this node (for example by strong
/// branching), which are valid for its whole subtree.
struct Node {
  NodeId id = kNoNode;
  NodeId parent = kNoNode;
  Index depth = 0;
  Index branch_col = -1;     ///< -1 for the root
  int branch_dir = 0;        ///< -1 down branch (upper bound lowered), +1 up branch, 0 root
  double branch_value = 0.0; ///< the fractional LP value of branch_col at the parent
  double lower_bound = 0.0;  ///< the parent's LP objective, strengthened by the objective granularity (minimization form)
  double parent_objective = 0.0;  ///< the parent's raw LP objective (pseudocost updates)
  double estimate = 0.0;     ///< estimated objective of the best integer solution below (best-estimate selection)
  std::vector<BoundChange> local;
  /// Basis of the parent's LP optimum, shared by both children; released once used or when the memory cap is hit.
  std::shared_ptr<const BasisSnapshot> basis;
};

/// Arena of nodes. Ids are assigned in creation order (deterministic) and index the arena.
class NodeTree {
 public:
  /// Creates the root (id 0) with the given lower bound.
  explicit NodeTree(double root_lower_bound = -kInf);

  NodeId add_child(NodeId parent, Index col, int dir, double value, double lower_bound, double estimate, BoundChange change,
                   std::shared_ptr<const BasisSnapshot> basis);
  Node& at(NodeId id) { return nodes_[static_cast<std::size_t>(id)]; }
  const Node& at(NodeId id) const { return nodes_[static_cast<std::size_t>(id)]; }
  std::size_t size() const { return nodes_.size(); }
  Index max_depth() const { return max_depth_; }

  /// Bounds of the node's subproblem: `lo`/`hi` start as the root bounds and are tightened along the path
  /// from the root. Returns false if some column's bounds cross (the node is infeasible).
  bool path_bounds(NodeId id, std::vector<double>& lo, std::vector<double>& hi) const;

 private:
  std::vector<Node> nodes_;
  Index max_depth_ = 0;
};

/// The set of open nodes ordered by (lower bound, id): gives the global bound in O(log n).
class OpenSet {
 public:
  void insert(NodeId id, double bound) { set_.insert({bound, id}); }
  void erase(NodeId id, double bound) { set_.erase({bound, id}); }
  bool empty() const { return set_.empty(); }
  std::size_t size() const { return set_.size(); }
  /// Smallest lower bound among the open nodes (kInf when empty).
  double min_bound() const { return set_.empty() ? kInf : set_.begin()->first; }

 private:
  std::set<std::pair<double, NodeId>> set_;
};

}  // namespace shodhan::mip
