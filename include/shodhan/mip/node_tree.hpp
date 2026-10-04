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

/// New bounds of a column (they replace the current ones inside the subtree of a node; the tree intersects them
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
  double lower_bound = 0.0;  ///< the parent LP objective, strengthened by the objective granularity (minimization form)
  double parent_objective = 0.0;  ///< the raw LP objective of the parent (pseudocost updates)
  double estimate = 0.0;     ///< estimated objective of the best integer solution below (best-estimate selection)
  std::vector<BoundChange> local;
  /// Basis of the parent LP optimum, shared by both children; released once used or when the memory cap is hit.
  std::shared_ptr<const BasisSnapshot> basis;
  int live_children = 0;     ///< children that still exist (not yet released)
  bool done = false;         ///< processed, pruned or dropped: kept only while descendants need its bound changes
};

/// Arena of nodes with slot reuse. Ids are assigned in creation order and never reused (deterministic); the
/// storage of a node is released once it is finished and none of its children exist any more, which cascades
/// towards the root, so the memory is proportional to the nodes still needed (open nodes and their ancestors),
/// not to the number of nodes ever created.
class NodeTree {
 public:
  /// Creates the root (id 0) with the given lower bound.
  explicit NodeTree(double root_lower_bound = -kInf);

  NodeId add_child(NodeId parent, Index col, int dir, double value, double lower_bound, double estimate, BoundChange change,
                   std::shared_ptr<const BasisSnapshot> basis);
  Node& at(NodeId id) { return slots_[static_cast<std::size_t>(slot_of_[static_cast<std::size_t>(id)])]; }
  const Node& at(NodeId id) const { return slots_[static_cast<std::size_t>(slot_of_[static_cast<std::size_t>(id)])]; }
  /// Whether the node still exists (has not been released).
  bool alive(NodeId id) const { return id >= 0 && static_cast<std::size_t>(id) < slot_of_.size() && slot_of_[static_cast<std::size_t>(id)] >= 0; }
  /// Nodes ever created, and nodes whose storage exists now.
  std::size_t created() const { return slot_of_.size(); }
  std::size_t live() const { return slots_.size() - free_.size(); }
  /// The largest number of nodes that existed at the same time.
  std::size_t peak_live() const { return peak_live_; }
  /// Number of slots allocated (the size of the arena).
  std::size_t capacity() const { return slots_.size(); }
  Index max_depth() const { return max_depth_; }

  /// Marks the node finished (processed, pruned or dropped; its children, if any, were created before this call).
  /// Its storage is released as soon as it has no live children, and the release cascades to ancestors that are
  /// finished and have no other live child. The id must not be used afterwards if it was released.
  void finish(NodeId id);

  /// Bounds of the subproblem of the node: `lo`/`hi` start as the root bounds and are tightened along the path
  /// from the root. Returns false if the bounds of some column cross (the node is infeasible).
  bool path_bounds(NodeId id, std::vector<double>& lo, std::vector<double>& hi) const;

 private:
  void release(NodeId id);

  std::vector<Node> slots_;
  std::vector<std::int32_t> free_;      ///< free slot indices (reused last-in first-out: deterministic)
  std::vector<std::int32_t> slot_of_;   ///< id -> slot, -1 once released
  Index max_depth_ = 0;
  std::size_t peak_live_ = 1;
};

/// The set of open nodes ordered by (lower bound, id): gives the global bound in O(log n).
class OpenSet {
 public:
  void insert(NodeId id, double bound) { set_.insert({bound, id}); }
  void erase(NodeId id, double bound) { set_.erase({bound, id}); }
  bool empty() const { return set_.empty(); }
  std::size_t size() const { return set_.size(); }
  /// Number of open nodes whose lower bound is below `threshold` (nodes at or above it cannot improve on an incumbent).
  std::size_t count_below(double threshold) const {
    std::size_t n = 0;
    for (const auto& e : set_) {
      if (e.first < threshold) ++n;
    }
    return n;
  }
  /// Smallest lower bound among the open nodes (kInf when empty).
  double min_bound() const { return set_.empty() ? kInf : set_.begin()->first; }

 private:
  std::set<std::pair<double, NodeId>> set_;
};

}  // namespace shodhan::mip
