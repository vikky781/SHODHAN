#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shodhan/mip/node_tree.hpp"
#include "shodhan/mip/search_state.hpp"

namespace shodhan::mip {

// ---------------------------------------------------------------------------------------------------------
// NodeSelector: chooses the next open node when the search is not plunging.
//
// Contract: push() is called once per open node (in creation order); pop() removes and returns the node the
// rule prefers, kNoNode when empty, and must be deterministic (ties broken by the smaller node id). Pruned
// nodes are not removed: the driver discards a popped node whose bound no longer beats the incumbent.
class NodeSelector {
 public:
  virtual ~NodeSelector() = default;
  virtual const char* name() const = 0;
  virtual void push(const Node& node) = 0;
  virtual NodeId pop() = 0;
  virtual bool empty() const = 0;
  virtual std::size_t size() const = 0;
  /// Whether the driver should continue into a child immediately after branching.
  virtual bool prefers_plunging() const { return true; }
};

// ---------------------------------------------------------------------------------------------------------
// BranchingRule: chooses the column to branch on at a node whose LP solution (state.lp) has fractional
// integer columns. May also report that the node is infeasible or that bounds were tightened.
struct BranchDecision {
  enum class Kind {
    Branch,    ///< branch on `col`
    Prune,     ///< the node cannot contain a better solution (for example both strong-branching children are cut off)
    Tighten,   ///< `tightenings` hold for the node's subtree: apply them, re-solve the node LP and ask again
  };
  Kind kind = Kind::Branch;
  Index col = -1;
  std::vector<BoundChange> tightenings;
  /// Estimated objective increases of the two children (0 when unknown); used for the node estimate.
  double down_gain = 0.0, up_gain = 0.0;
};

class BranchingRule {
 public:
  virtual ~BranchingRule() = default;
  virtual const char* name() const = 0;
  /// Called with a non-empty state.lp.fractional.
  virtual BranchDecision select(SearchState& state) = 0;
};

// ---------------------------------------------------------------------------------------------------------
// PrimalHeuristic: looks for feasible solutions and offers them through state.submit (never accepted
// without verification by the incumbent manager).
enum class HeuristicWhen { BeforeRootLp, AfterRootLp, AtNode };

class PrimalHeuristic {
 public:
  virtual ~PrimalHeuristic() = default;
  virtual const char* name() const = 0;
  /// Whether to run now, given the options (frequencies). `nodes_processed` counts nodes with a solved LP so far.
  virtual bool wants(const MipOptions& options, HeuristicWhen when, long long nodes_processed, Index depth) const = 0;
  /// BeforeRootLp: state.lp is empty (no LP yet). Otherwise state.lp is the LP solution of the node.
  virtual void run(SearchState& state, HeuristicWhen when) = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Separator: interface only (implementations come with the cutting planes of a later step). A separator
// looks at a fractional LP solution and returns inequalities that cut it off.
struct CutRow {
  std::vector<Index> index;
  std::vector<double> value;
  double lower = -kInf, upper = kInf;
};

class Separator {
 public:
  virtual ~Separator() = default;
  virtual const char* name() const = 0;
  /// Appends violated inequalities valid for all integer feasible points; returns how many were added.
  virtual int separate(SearchState& state, std::vector<CutRow>& cuts) = 0;
};

// ---------------------------------------------------------------------------------------------------------
/// Name -> factory registry. The default plugins register themselves in register_default_plugins().
template <typename T>
class Registry {
 public:
  using Factory = std::function<std::unique_ptr<T>()>;
  void add(const std::string& name, Factory factory) { factories_[name] = std::move(factory); }
  bool has(const std::string& name) const { return factories_.count(name) != 0; }
  std::unique_ptr<T> make(const std::string& name) const {
    const auto it = factories_.find(name);
    return it == factories_.end() ? nullptr : it->second();
  }
  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (const auto& kv : factories_) out.push_back(kv.first);
    return out;
  }

 private:
  std::map<std::string, Factory> factories_;
};

struct PluginRegistry {
  Registry<NodeSelector> selectors;
  Registry<BranchingRule> branching;
  Registry<PrimalHeuristic> heuristics;
  Registry<Separator> separators;
};

/// The process-wide registry, with the built-in plugins registered on first use. Tests and later steps may add
/// their own entries.
PluginRegistry& plugin_registry();

}  // namespace shodhan::mip
