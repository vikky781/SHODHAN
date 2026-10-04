#include "shodhan/mip/mip_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>

#include "shodhan/lp_solver.hpp"
#include "shodhan/mip/granularity.hpp"
#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/node_tree.hpp"
#include "shodhan/mip/plugins.hpp"
#include "shodhan/mip/search_state.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"

namespace shodhan::mip {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

LpModel to_min_form(const LpModel& m) {
  LpModel out = m;
  if (m.sense == Sense::Maximize) {
    for (double& c : out.col_cost) c = -c;
    out.objective_offset = -out.objective_offset;
    out.sense = Sense::Minimize;
  }
  return out;
}

/// Heuristics are created in this fixed order (the order of MipResult::heuristics).
const char* const kHeuristicNames[] = {"trivial", "rounding", "diving-fractional", "diving-coefficient", "feasibility-jump"};

class Search {
 public:
  Search(const LpModel& original, const MipOptions& opt) : original_(original), opt_(opt), rng_(opt.params.seed) {}

  MipResult run();

 private:
  enum class LpOutcome { Optimal, Infeasible, Unbounded, Trouble, TimeLimit };
  enum class Stop { None, Gap, NodeLimit, TimeLimit, Exhausted, RootFailed };

  // ---- setup ----
  bool setup_presolve(MipResult& res);
  void setup_engine();
  void make_plugins();
  SearchState make_state();

  // ---- search ----
  double elapsed() const { return seconds_since(t_start_); }
  bool time_up() const { return !is_inf(opt_.params.time_limit) && elapsed() >= opt_.params.time_limit; }
  double strengthen(double lb) const;
  bool prune(double lb) const;
  double global_bound(NodeId pending) const;
  double incumbent_min() const { return inc_ && inc_->has_incumbent() ? inc_->objective_min() : kInf; }
  void apply_bounds();
  LpOutcome solve_node_lp(bool is_root);
  void capture_lp(NodeLp& lp) const;
  NodeId process(NodeId id, bool plunged);
  void run_heuristics(HeuristicWhen when, NodeId id);
  bool submit(const std::vector<double>& x, const std::string& source);
  void log_progress(bool force);
  void finalize(MipResult& res, Stop stop, bool preset);
  void attach_lp_certificate(MipResult& res);
  std::shared_ptr<const BasisSnapshot> make_snapshot();
  std::size_t heuristic_index(const std::string& name) const;

  const LpModel& original_;
  MipOptions opt_;
  Clock::time_point t_start_ = Clock::now();

  // presolve / scaling
  PresolveResult pre_;
  bool presolved_ = false;
  LpModel pm_;  // presolved (or min-form original), unscaled, minimization
  LpModel sm_;  // scaled copy for the engine
  Scaling sc_;
  std::unique_ptr<SimplexEngine> engine_;
  std::unique_ptr<IncumbentManager> inc_;

  // search structures (live_bases_ before tree_: shared snapshots decrement it when destroyed)
  std::size_t live_bases_ = 0;
  NodeTree tree_;
  OpenSet open_;
  std::unique_ptr<NodeSelector> selector_;
  std::unique_ptr<BranchingRule> rule_;
  std::vector<std::unique_ptr<PrimalHeuristic>> heurs_;
  std::vector<HeuristicStats> hstats_;
  Pseudocosts pseudo_;
  Random rng_;
  std::vector<Index> int_cols_;
  std::vector<double> root_lo_, root_hi_, tlo_, thi_;
  double granularity_ = 0.0;
  double offset_ = 0.0;
  NodeLp node_lp_;
  NodeId current_ = kNoNode;

  // counters
  long long nodes_processed_ = 0, pruned_bound_ = 0, pruned_infeasible_ = 0, trouble_nodes_ = 0, lp_integral_ = 0;
  long long extra_lp_iterations_ = 0, strong_iterations_ = 0, strong_calls_ = 0;
  double dropped_min_ = kInf;
  double t_presolve_ = 0, t_root_ = 0, t_nodes_ = 0, t_strong_ = 0, t_heur_ = 0;
  bool root_lp_infeasible_ = false, root_unbounded_ = false, root_failed_ = false;
  long long last_log_node_ = 0;
  std::string message_;
  Stop pending_stop_ = Stop::None;
  NodeId pending_node_ = kNoNode;
};

std::size_t Search::heuristic_index(const std::string& name) const {
  for (std::size_t k = 0; k < hstats_.size(); ++k) {
    if (hstats_[k].name == name) return k;
  }
  return hstats_.size();
}

// --------------------------------------------------------------------------------------------------------
bool Search::setup_presolve(MipResult& res) {
  const Clock::time_point t0 = Clock::now();
  if (opt_.presolve) {
    PresolveOptions po;
    po.is_mip = true;
    po.need_duals = false;
    pre_ = presolve(original_, po);
    presolved_ = true;
    res.presolve_rows_before = original_.n_rows;
    res.presolve_cols_before = original_.n_cols;
    switch (pre_.status) {
      case PresolveStatus::Infeasible:
        res.status = Status::Infeasible;
        res.message = "presolve proved the model infeasible" + (pre_.note.empty() ? std::string() : " (" + pre_.note + ")");
        t_presolve_ = seconds_since(t0);
        return false;
      case PresolveStatus::Unbounded:
      case PresolveStatus::InfeasibleOrUnbounded:
        res.status = Status::InfeasibleOrUnbounded;
        res.message = "presolve found an improving ray: the model is infeasible or unbounded (not certified)";
        t_presolve_ = seconds_since(t0);
        return false;
      case PresolveStatus::SolvedByPresolve: {
        const Solution full = postsolve(pre_.stack, Solution{});
        inc_ = std::make_unique<IncumbentManager>(original_, &pre_.stack, opt_);
        res.presolve_rows_after = res.presolve_cols_after = 0;
        if (inc_->submit_original(full.x, "presolve") == SubmitOutcome::Accepted) {
          res.status = Status::Optimal;
          res.message = "solved by presolve";
        } else {
          res.status = Status::NumericalError;
          res.message = "presolve solved the model but its point failed verification on the original model";
        }
        t_presolve_ = seconds_since(t0);
        return false;
      }
      case PresolveStatus::Reduced:
        break;
    }
    pm_ = pre_.reduced;
    res.presolve_rows_after = pm_.n_rows;
    res.presolve_cols_after = pm_.n_cols;
    inc_ = std::make_unique<IncumbentManager>(original_, &pre_.stack, opt_);
  } else {
    pm_ = to_min_form(original_);
    inc_ = std::make_unique<IncumbentManager>(original_, nullptr, opt_);
  }
  t_presolve_ = seconds_since(t0);
  return true;
}

void Search::setup_engine() {
  if (opt_.scaling) {
    sc_ = compute_scaling(pm_);
    sm_ = apply_scaling(pm_, sc_);
  } else {
    sc_.row_scale.assign(to_size(pm_.n_rows), 1.0);
    sc_.col_scale.assign(to_size(pm_.n_cols), 1.0);
    sc_.obj_scale = 1.0;
    sm_ = pm_;
  }
  SimplexOptions so;
  so.primal_tol = opt_.params.primal_tol;
  so.dual_tol = opt_.params.dual_tol;
  so.seed = opt_.params.seed;
  so.final_check = true;  // the root LP is checked; node LPs are not (see solve_node_lp)
  engine_ = std::make_unique<SimplexEngine>(sm_, so);
  for (Index j = 0; j < pm_.n_cols; ++j) {
    if (pm_.is_integer(j)) int_cols_.push_back(j);
  }
  root_lo_ = pm_.col_lower;
  root_hi_ = pm_.col_upper;
  // Integer columns have integral bounds: round them (presolve normally did).
  for (const Index j : int_cols_) {
    if (!is_inf(root_lo_[to_size(j)])) root_lo_[to_size(j)] = std::ceil(root_lo_[to_size(j)] - 1e-9);
    if (!is_inf(root_hi_[to_size(j)])) root_hi_[to_size(j)] = std::floor(root_hi_[to_size(j)] + 1e-9);
  }
  tlo_ = root_lo_;
  thi_ = root_hi_;
  pseudo_.resize(pm_.n_cols);
  granularity_ = objective_granularity(pm_);
  offset_ = pm_.objective_offset;
}

void Search::make_plugins() {
  PluginRegistry& reg = plugin_registry();
  selector_ = reg.selectors.make(to_string(opt_.node_select));
  rule_ = reg.branching.make(to_string(opt_.branching));
  if (!rule_) {
    message_ += std::string("branching rule ") + to_string(opt_.branching) + " is not available, using mostfrac; ";
    rule_ = reg.branching.make("mostfrac");
  }
  for (const char* name : kHeuristicNames) {
    HeuristicStats hs;
    hs.name = name;
    hstats_.push_back(hs);
    heurs_.push_back(reg.heuristics.make(name));  // null when not registered
  }
}

SearchState Search::make_state() {
  SearchState s;
  s.model = &pm_;
  s.scaling = &sc_;
  s.engine = engine_.get();
  s.options = &opt_;
  s.tree = &tree_;
  s.node = current_;
  s.pseudo = &pseudo_;
  s.rng = &rng_;
  s.int_cols = &int_cols_;
  s.lp = node_lp_;
  s.lo = &tlo_;
  s.hi = &thi_;
  s.obj_scale = sc_.obj_scale;
  s.submit = [this](const std::vector<double>& x, const std::string& src) { return submit(x, src); };
  s.cutoff = [this]() { return incumbent_min(); };
  s.time_up = [this]() { return time_up(); };
  s.lp_iterations = [this]() { return engine_->stats().iterations + extra_lp_iterations_; };
  s.add_iterations = [this](long long n, bool strong) {
    if (strong) strong_iterations_ += n;
    else extra_lp_iterations_ += n;
  };
  return s;
}

bool Search::submit(const std::vector<double>& x, const std::string& source) {
  const std::size_t k = heuristic_index(source);
  if (k < hstats_.size()) ++hstats_[k].submitted;
  const SubmitOutcome out = inc_->submit(x, source);
  if (out == SubmitOutcome::Accepted) {
    if (k < hstats_.size()) ++hstats_[k].successes;
    if (opt_.log != nullptr && opt_.params.verbosity >= 1) {
      std::ostringstream os;
      os << "  * new incumbent " << std::setprecision(12) << inc_->objective() << " (" << source << ") after " << nodes_processed_ << " nodes, "
         << std::fixed << std::setprecision(2) << elapsed() << " s\n";
      *opt_.log << os.str();
    }
    return true;
  }
  return false;
}

// --------------------------------------------------------------------------------------------------------
double Search::strengthen(double lb) const {
  if (!(granularity_ > 0.0) || is_inf(lb)) return lb;
  const double t = (lb - offset_) / granularity_;
  const double k = std::ceil(t - (1e-6 + 1e-9 * std::fabs(t)));
  return offset_ + k * granularity_;
}

bool Search::prune(double lb) const {
  const double inc = incumbent_min();
  if (inc >= kInf) return false;
  return lb >= inc - 1e-9 * std::max(1.0, std::fabs(inc));
}

double Search::global_bound(NodeId pending) const {
  double lb = std::min(open_.min_bound(), dropped_min_);
  if (pending != kNoNode) lb = std::min(lb, tree_.at(pending).lower_bound);
  return lb;
}

void Search::apply_bounds() {
  for (const Index j : int_cols_) {
    const double lo = tlo_[to_size(j)], hi = thi_[to_size(j)];
    if (engine_->col_lower(j) != lo || engine_->col_upper(j) != hi) engine_->change_col_bounds(j, lo, hi);
  }
}

std::shared_ptr<const BasisSnapshot> Search::make_snapshot() {
  if (live_bases_ >= opt_.max_stored_bases) return nullptr;
  ++live_bases_;
  return std::shared_ptr<const BasisSnapshot>(new BasisSnapshot(engine_->get_basis_snapshot()), [this](const BasisSnapshot* p) {
    --live_bases_;
    delete p;
  });
}

void Search::capture_lp(NodeLp& lp) const {
  lp.objective = engine_->objective() / sc_.obj_scale;
  const std::size_t n = to_size(pm_.n_cols);
  lp.x.resize(n);
  const std::vector<double>& xs = engine_->primal_all();
  for (std::size_t j = 0; j < n; ++j) lp.x[j] = xs[j] * sc_.col_scale[j];
  lp.fractional.clear();
  for (const Index j : int_cols_) {
    const double v = lp.x[to_size(j)];
    if (std::fabs(v - std::round(v)) > opt_.params.int_tol) lp.fractional.push_back(j);
  }
}

// Solves the node LP from the engine's current state. A failure (iteration limit or numerical error) is
// retried once with a refactorization and relaxed ratio-test thresholds; if that fails too the caller marks
// the search incomplete. Nothing is silently discarded.
Search::LpOutcome Search::solve_node_lp(bool is_root) {
  const Clock::time_point t0 = Clock::now();
  auto limit_time = [&]() {
    double remaining = kInf;
    if (!is_inf(opt_.params.time_limit)) remaining = std::max(opt_.params.time_limit - elapsed(), 1e-3);
    engine_->options().time_limit = remaining;
  };
  engine_->options().final_check = is_root;
  limit_time();
  EngineStatus st = engine_->solve_limited(opt_.node_iteration_limit);
  if (st != EngineStatus::Optimal && st != EngineStatus::Infeasible && st != EngineStatus::TimeLimit && st != EngineStatus::Unbounded) {
    SimplexOptions& so = engine_->options();
    const double pa = so.min_pivot_abs, pr = so.min_pivot_rel;
    so.min_pivot_abs = pa * 0.01;
    so.min_pivot_rel = pr * 0.01;
    engine_->refactor(true);
    limit_time();
    st = engine_->solve_limited(opt_.node_iteration_limit);
    so.min_pivot_abs = pa;
    so.min_pivot_rel = pr;
  }
  (is_root ? t_root_ : t_nodes_) += seconds_since(t0);
  switch (st) {
    case EngineStatus::Optimal: return LpOutcome::Optimal;
    case EngineStatus::Infeasible: return LpOutcome::Infeasible;
    case EngineStatus::Unbounded: return LpOutcome::Unbounded;
    case EngineStatus::TimeLimit: return LpOutcome::TimeLimit;
    default: return LpOutcome::Trouble;
  }
}

void Search::run_heuristics(HeuristicWhen when, NodeId id) {
  if (!opt_.heuristics) return;
  const Index depth = id == kNoNode ? 0 : tree_.at(id).depth;
  for (std::size_t k = 0; k < heurs_.size(); ++k) {
    PrimalHeuristic* h = heurs_[k].get();
    if (h == nullptr || !h->wants(opt_, when, nodes_processed_, depth)) continue;
    if (time_up()) return;
    const Clock::time_point t0 = Clock::now();
    SearchState state = make_state();
    h->run(state, when);
    ++hstats_[k].calls;
    const double dt = seconds_since(t0);
    hstats_[k].seconds += dt;
    t_heur_ += dt;
  }
}

void Search::log_progress(bool force) {
  if (opt_.log == nullptr || opt_.params.verbosity < 1) return;
  if (!force && (opt_.log_interval <= 0 || nodes_processed_ - last_log_node_ < opt_.log_interval)) return;
  last_log_node_ = nodes_processed_;
  const double lb = global_bound(kNoNode);
  const double inc = incumbent_min();
  const double sense = original_.sense == Sense::Maximize ? -1.0 : 1.0;
  std::ostringstream os;
  os << "  nodes " << std::setw(8) << nodes_processed_ << "  open " << std::setw(7) << open_.size() << "  incumbent ";
  if (inc >= kInf) os << std::setw(14) << "-";
  else os << std::setw(14) << std::setprecision(8) << sense * inc;
  os << "  bound ";
  if (is_inf(lb)) os << std::setw(14) << "-";
  else os << std::setw(14) << std::setprecision(8) << sense * lb;
  if (inc < kInf && !is_inf(lb)) os << "  gap " << std::fixed << std::setprecision(3) << 100.0 * (inc - lb) / std::max(1.0, std::fabs(inc)) << "%" << std::defaultfloat;
  os << "  iters " << engine_->stats().iterations + extra_lp_iterations_ << "  time " << std::fixed << std::setprecision(2) << elapsed() << " s\n";
  *opt_.log << os.str();
}

// --------------------------------------------------------------------------------------------------------
// Processes one node and returns the child to plunge into (or kNoNode).
NodeId Search::process(NodeId id, bool plunged) {
  const bool is_root = id == 0;
  current_ = id;
  if (!tree_.path_bounds(id, tlo_ = root_lo_, thi_ = root_hi_)) {
    ++pruned_infeasible_;
    return kNoNode;
  }
  // The root bounds are the engine's bounds already; nodes change them.
  apply_bounds();
  if (!plunged && tree_.at(id).basis) {
    engine_->set_basis(*tree_.at(id).basis);
  }
  tree_.at(id).basis.reset();
  ++nodes_processed_;

  for (int round = 0; round < 200; ++round) {
    const LpOutcome lo = solve_node_lp(is_root);
    if (lo == LpOutcome::TimeLimit) {
      pending_stop_ = Stop::TimeLimit;
      pending_node_ = id;
      --nodes_processed_;
      return kNoNode;
    }
    if (lo == LpOutcome::Infeasible) {
      if (is_root) root_lp_infeasible_ = true;
      ++pruned_infeasible_;
      return kNoNode;
    }
    if (lo == LpOutcome::Unbounded && is_root) {
      root_unbounded_ = true;
      return kNoNode;
    }
    if (lo != LpOutcome::Optimal) {
      if (is_root) {
        root_failed_ = true;
        return kNoNode;
      }
      ++trouble_nodes_;
      dropped_min_ = std::min(dropped_min_, tree_.at(id).lower_bound);
      return kNoNode;
    }
    capture_lp(node_lp_);
    const double z = node_lp_.objective;
    // Pseudocost update from the observed gain of the branching that created this node.
    {
      const Node& nd = tree_.at(id);
      if (nd.branch_col >= 0 && round == 0) {
        const double f = nd.branch_value - std::floor(nd.branch_value);
        pseudo_.update(nd.branch_col, nd.branch_dir, std::max(0.0, z - nd.parent_objective), nd.branch_dir < 0 ? f : 1.0 - f);
      }
    }
    tree_.at(id).lower_bound = std::max(tree_.at(id).lower_bound, strengthen(z));
    if (prune(tree_.at(id).lower_bound)) {
      ++pruned_bound_;
      return kNoNode;
    }
    if (is_root) run_heuristics(HeuristicWhen::AfterRootLp, id);
    if (node_lp_.fractional.empty()) {
      ++lp_integral_;
      if (!submit(node_lp_.x, "lp")) {
        // The LP point is integral but was not accepted: either it is not better (fine) or it failed
        // verification (the LP tolerance and the original model disagree): then the node is uncertain.
        const RejectionStats& r = inc_->rejections();
        (void)r;
        if (!(inc_->has_incumbent() && inc_->objective_min() <= z + 1e-6 * std::max(1.0, std::fabs(z)))) {
          ++trouble_nodes_;
          dropped_min_ = std::min(dropped_min_, tree_.at(id).lower_bound);
        }
      }
      return kNoNode;
    }
    if (round == 0 && !is_root) run_heuristics(HeuristicWhen::AtNode, id);
    if (prune(tree_.at(id).lower_bound)) {
      ++pruned_bound_;
      return kNoNode;
    }
    // Heuristics may have changed nothing in the engine (they work on copies); the LP solution stands.
    SearchState state = make_state();
    const BranchDecision dec = rule_->select(state);
    if (dec.kind == BranchDecision::Kind::Prune) {
      ++pruned_bound_;
      return kNoNode;
    }
    if (dec.kind == BranchDecision::Kind::Tighten) {
      for (const BoundChange& c : dec.tightenings) tree_.at(id).local.push_back(c);
      if (!tree_.path_bounds(id, tlo_ = root_lo_, thi_ = root_hi_)) {
        ++pruned_infeasible_;
        return kNoNode;
      }
      apply_bounds();
      continue;
    }
    // ---- branch ----
    const Index j = dec.col;
    const double v = node_lp_.x[to_size(j)];
    const double fl = std::floor(v), ce = std::ceil(v);
    const double f = v - fl;
    const double bound_child = tree_.at(id).lower_bound;
    // Estimates (best-estimate selection): objective + sum over fractional columns of the cheaper rounding.
    double sum_min = 0.0;
    for (const Index k : node_lp_.fractional) {
      const double fk = node_lp_.x[to_size(k)] - std::floor(node_lp_.x[to_size(k)]);
      sum_min += std::min(pseudo_.value(k, -1) * fk, pseudo_.value(k, +1) * (1.0 - fk));
    }
    const double own_min = std::min(pseudo_.value(j, -1) * f, pseudo_.value(j, +1) * (1.0 - f));
    const double est_down = z + sum_min - own_min + pseudo_.value(j, -1) * f;
    const double est_up = z + sum_min - own_min + pseudo_.value(j, +1) * (1.0 - f);
    std::shared_ptr<const BasisSnapshot> snap = make_snapshot();
    const double lo_j = tlo_[to_size(j)], hi_j = thi_[to_size(j)];
    NodeId down = kNoNode, up = kNoNode;
    if (fl >= lo_j) {
      down = tree_.add_child(id, j, -1, v, bound_child, est_down, BoundChange{j, lo_j, fl}, snap);
      tree_.at(down).parent_objective = z;
    }
    if (ce <= hi_j) {
      up = tree_.add_child(id, j, +1, v, bound_child, est_up, BoundChange{j, ce, hi_j}, snap);
      tree_.at(up).parent_objective = z;
    }
    snap.reset();
    // Preferred child: the rounding direction that is nearer; ties go up.
    NodeId first = f > 0.5 || down == kNoNode ? up : down;
    NodeId second = first == up ? down : up;
    if (first == kNoNode) first = second, second = kNoNode;
    if (first == kNoNode) return kNoNode;
    bool plunge = selector_->prefers_plunging() && opt_.plunging;
    if (plunge && inc_->has_incumbent()) {
      const double lb = std::min(global_bound(kNoNode), tree_.at(first).lower_bound);
      const double inc = incumbent_min();
      if (tree_.at(first).lower_bound > lb + opt_.plunge_gap_fraction * (inc - lb)) plunge = false;
    }
    auto enqueue = [&](NodeId n) {
      selector_->push(tree_.at(n));
      open_.insert(n, tree_.at(n).lower_bound);
    };
    if (second != kNoNode) enqueue(second);
    if (plunge) return first;
    enqueue(first);
    return kNoNode;
  }
  // Too many tightening rounds without a decision: treat as numerical trouble rather than looping.
  ++trouble_nodes_;
  dropped_min_ = std::min(dropped_min_, tree_.at(id).lower_bound);
  return kNoNode;
}

// --------------------------------------------------------------------------------------------------------
void Search::attach_lp_certificate(MipResult& res) {
  // An LP-level certificate exists only when the relaxation itself is infeasible.
  LpModel relax = original_;
  for (ColType& t : relax.col_type) t = ColType::Continuous;
  LpOptions lo;
  lo.params = opt_.params;
  lo.params.verbosity = 0;
  const LpResult lr = LpSolver(lo).solve(relax);
  if (lr.status == Status::Infeasible && !lr.farkas_ray.empty()) {
    res.lp_infeasible_certified = true;
    res.lp_farkas = lr.farkas_ray;
  }
}

void Search::finalize(MipResult& res, Stop stop, bool preset) {
  const double sense = original_.sense == Sense::Maximize ? -1.0 : 1.0;
  res.nodes_processed = nodes_processed_;
  res.nodes_open = static_cast<long long>(open_.size()) + (pending_node_ != kNoNode ? 1 : 0);
  res.lp_iterations = (engine_ ? engine_->stats().iterations : 0) + extra_lp_iterations_;
  res.strong_branching_iterations = strong_iterations_;
  res.strong_branching_calls = strong_calls_;
  res.max_depth = tree_.max_depth();
  res.numerical_trouble_nodes = trouble_nodes_;
  res.nodes_pruned_by_bound = pruned_bound_;
  res.nodes_pruned_infeasible = pruned_infeasible_;
  res.lp_solutions_integral = lp_integral_;
  res.heuristics = hstats_;
  if (inc_) {
    res.rejections = inc_->rejections();
    res.incumbents_found = inc_->found();
    res.incumbent_source = inc_->source();
  }
  res.seconds_presolve = t_presolve_;
  res.seconds_root_lp = t_root_;
  res.seconds_node_lps = t_nodes_;
  res.seconds_strong_branching = t_strong_;
  res.seconds_heuristics = t_heur_;
  res.seconds_total = elapsed();

  const bool has_inc = inc_ && inc_->has_incumbent();
  if (has_inc) {
    res.has_solution = true;
    res.solution.x = inc_->x();
    res.objective = inc_->objective();
    res.solution.objective = res.objective;
  }
  if (preset) {  // presolve decided: Infeasible, InfeasibleOrUnbounded, or solved
    if (has_inc && res.status == Status::Optimal) {
      res.has_bound = true;
      res.best_bound = res.objective;
    }
    if (res.status == Status::Infeasible) attach_lp_certificate(res);
    return;
  }

  // The proven bound (minimization form): the smallest bound of every node not yet decided. With no open
  // node left the bound is the incumbent.
  double b = global_bound(pending_node_);
  if (has_inc) b = std::min(b, incumbent_min());
  if (!is_inf(b)) {
    res.has_bound = true;
    res.best_bound = sense * b;
  }
  if (res.has_solution && res.has_bound) {
    res.abs_gap = std::fabs(res.objective - res.best_bound);
    res.rel_gap = res.abs_gap / std::max(1.0, std::fabs(res.objective));
  }

  if (root_unbounded_) {
    res.status = Status::InfeasibleOrUnbounded;
    res.has_bound = false;
    res.message = "the LP relaxation is unbounded: the MIP is infeasible or unbounded (not certified)";
    return;
  }
  if (root_failed_) {
    res.status = Status::NumericalError;
    res.has_bound = false;
    res.message = "the root LP could not be solved reliably";
    return;
  }
  switch (stop) {
    case Stop::TimeLimit: res.status = Status::TimeLimit; break;
    case Stop::NodeLimit: res.status = Status::NodeLimit; break;
    case Stop::Gap:
    case Stop::Exhausted:
      if (trouble_nodes_ > 0) {
        res.status = Status::NumericalError;
        res.message = std::to_string(trouble_nodes_) + " node(s) could not be solved reliably and were dropped; the search is incomplete";
      } else if (has_inc) {
        res.status = Status::Optimal;
      } else {
        res.status = Status::Infeasible;
        res.has_bound = false;
        res.message = root_lp_infeasible_ ? "the LP relaxation is infeasible" : "no integer feasible point exists (proved by branching; no certificate)";
        attach_lp_certificate(res);
      }
      break;
    default: res.status = Status::NumericalError; break;
  }
}

MipResult Search::run() {
  MipResult res;
  if (original_.quadratic.nnz() > 0) {
    res.status = Status::NotImplemented;
    res.message = "quadratic objectives are not supported by the MILP solver";
    return res;
  }
  if (!setup_presolve(res)) {
    finalize(res, Stop::Exhausted, true);
    return res;
  }
  setup_engine();
  make_plugins();
  current_ = kNoNode;

  Stop stop = Stop::None;
  // Heuristics that need no LP run before the root.
  run_heuristics(HeuristicWhen::BeforeRootLp, kNoNode);

  NodeId pending = 0;  // the root, processed first and not in the open set
  while (true) {
    if (pending_stop_ != Stop::None) {
      stop = pending_stop_;
      pending = pending_node_ = (pending_node_ != kNoNode ? pending_node_ : pending);
      break;
    }
    // ---- termination ----
    if (time_up()) {
      stop = Stop::TimeLimit;
      break;
    }
    if (nodes_processed_ >= opt_.node_limit) {
      stop = Stop::NodeLimit;
      break;
    }
    if (inc_->has_incumbent() && nodes_processed_ > 0) {
      const double lb = global_bound(pending);
      const double inc = incumbent_min();
      if (!is_inf(lb) && inc - lb <= std::max(opt_.mip_abs_gap, opt_.mip_gap * std::max(1.0, std::fabs(inc)))) {
        stop = Stop::Gap;
        break;
      }
    }
    NodeId id = pending;
    const bool plunged = id != kNoNode && id != 0;
    pending = kNoNode;
    if (id == kNoNode) {
      id = selector_->pop();
      if (id == kNoNode) {
        stop = Stop::Exhausted;
        break;
      }
      open_.erase(id, tree_.at(id).lower_bound);
      if (prune(tree_.at(id).lower_bound)) {
        ++pruned_bound_;
        continue;
      }
    }
    pending = process(id, plunged);
    log_progress(false);
    if (id == 0 && (root_unbounded_ || root_failed_)) {
      stop = Stop::RootFailed;
      break;
    }
    if (pending_stop_ != Stop::None) {
      stop = pending_stop_;
      pending = kNoNode;
      break;
    }
  }
  if (stop == Stop::Gap || stop == Stop::Exhausted) {
    if (pending != kNoNode && stop == Stop::Gap) pending_node_ = pending;
  } else if (pending != kNoNode && pending_node_ == kNoNode) {
    pending_node_ = pending;
  }
  log_progress(true);
  finalize(res, stop, false);
  return res;
}

}  // namespace

MipResult MipSolver::solve(const LpModel& model) const {
  Search s(model, options_);
  return s.run();
}

}  // namespace shodhan::mip
