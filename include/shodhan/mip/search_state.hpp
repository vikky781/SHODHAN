#pragma once

#include <functional>
#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/node_tree.hpp"
#include "shodhan/mip/options.hpp"
#include "shodhan/mip/random.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"

namespace shodhan::mip {

/// Pseudocosts (objective gain per unit change of a column) per column and direction. Updated after every node
/// LP and by strong branching.
class Pseudocosts {
 public:
  explicit Pseudocosts(Index n_cols = 0) { resize(n_cols); }
  void resize(Index n_cols);
  /// Records one observation: branching `col` in direction `dir` (-1 down, +1 up) moved the objective by
  /// `gain` (>= 0) while the column moved by `delta` (> 0).
  void update(Index col, int dir, double gain, double delta);
  int count(Index col, int dir) const { return n_[dir > 0 ? 1 : 0][static_cast<std::size_t>(col)]; }
  /// Average gain per unit; columns without observations get the global average of that direction (1 if none).
  double value(Index col, int dir) const;
  bool reliable(Index col, int threshold) const { return count(col, -1) >= threshold && count(col, +1) >= threshold; }
  /// The product score max(down, eps) * max(up, eps) with down = value(down) * f, up = value(up) * (1 - f).
  double score(Index col, double frac) const;

 private:
  std::vector<double> sum_[2];
  std::vector<int> n_[2];
  double total_sum_[2] = {0.0, 0.0};
  long long total_n_[2] = {0, 0};
};

/// LP solution of the node being processed, in the presolved (unscaled) space.
struct NodeLp {
  double objective = 0.0;          ///< minimization form, presolved units
  std::vector<double> x;           ///< all columns
  std::vector<Index> fractional;   ///< integer columns farther than int_tol from an integer
};

/// What plugins see of the search. The main engine is exposed mutably so that probes (strong branching, diving)
/// need no copy of it, but a plugin must leave it exactly as it found it: wrap every probe in an EngineProbe.
struct SearchState {
  const LpModel* model = nullptr;       ///< presolved model, unscaled, minimization
  const Scaling* scaling = nullptr;     ///< column scales (integer columns have scale 1)
  SimplexEngine* engine = nullptr;  ///< plugins that change it (strong branching, diving) must restore it: use EngineProbe
  const MipOptions* options = nullptr;
  const NodeTree* tree = nullptr;
  NodeId node = kNoNode;
  Pseudocosts* pseudo = nullptr;
  Random* rng = nullptr;
  const std::vector<Index>* int_cols = nullptr;
  NodeLp lp;
  /// Bounds of the node's subproblem, in the presolved space.
  const std::vector<double>* lo = nullptr;
  const std::vector<double>* hi = nullptr;

  // ---- services provided by the driver ----
  /// Offers a candidate (presolved, unscaled values) to the incumbent manager; true if it is the new incumbent.
  std::function<bool(const std::vector<double>&, const std::string&)> submit;
  /// Smallest objective (minimization form) a node must stay below to be worth exploring; kInf without incumbent.
  std::function<double()> cutoff;
  std::function<bool()> time_up;
  /// Total dual simplex iterations spent so far (node LPs, strong branching, diving): for budgets.
  std::function<long long()> lp_iterations;
  /// Adds iterations spent by a plugin (strong branching, diving) to the totals.
  std::function<void(long long, bool strong)> add_iterations;
  /// Counts one strong-branching child solve.
  std::function<void()> count_strong_solve;

  double obj_scale = 1.0;  ///< the engine's objective is obj_scale times the presolved objective
  bool has_incumbent() const { return cutoff && cutoff() < kInf; }
};

}  // namespace shodhan::mip
