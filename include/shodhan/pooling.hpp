#pragma once

// The standard pooling problem in the q-formulation (docs/POOLING.md): a NONCONVEX problem that lives outside the
// LP/MILP/QP core. This module builds LPs from it and solves them with LpSolver; it gives local solutions only (a
// distributive recursion and a sequential LP with a trust region, with multi-start) and a McCormick upper bound. A
// solution is returned only after it has been checked against the original nonlinear equations.
//
//   sources s (cost c_s, supply A_s, qualities q_sk), pools p (capacity C_p), terminals t (price pi_t, demand D_t,
//   maximum qualities Q_tk), flows f_sp, f_pt, f_st >= 0 on the arcs of the network, pool qualities q_pk.
//   max   sum_t pi_t (sum_p f_pt + sum_s f_st) - sum_s c_s (sum_p f_sp + sum_t f_st)
//   s.t.  sum_p f_sp + sum_t f_st <= A_s            sum_s f_sp <= C_p            sum_p f_pt + sum_s f_st <= D_t
//         sum_s f_sp = sum_t f_pt                                                  (pool material balance)
//         sum_s q_sk f_sp = q_pk * sum_t f_pt                                      (pool quality balance, bilinear)
//         sum_p q_pk f_pt + sum_s q_sk f_st <= Q_tk (sum_p f_pt + sum_s f_st)       (terminal quality, bilinear)

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "shodhan/constants.hpp"
#include "shodhan/sparse_matrix.hpp"
#include "shodhan/status.hpp"

namespace shodhan::pooling {

struct Source {
  std::string name;
  double cost = 0.0;
  double supply = kInf;
  std::vector<double> quality;  ///< one value per quality
};

struct Pool {
  std::string name;
  double capacity = kInf;
};

struct Terminal {
  std::string name;
  double price = 0.0;
  double demand = kInf;
  std::vector<double> spec;  ///< maximum blend quality per quality; kInf = no specification
};

struct PoolProblem {
  std::string name;
  bool synthetic = false;  ///< the file says that the data are synthetic (the CLI repeats it)
  int n_qualities = 0;
  std::vector<std::string> quality_names;
  std::vector<Source> sources;
  std::vector<Pool> pools;
  std::vector<Terminal> terminals;
  /// Arcs as (from index, to index): source -> pool, pool -> terminal, source -> terminal.
  std::vector<std::pair<int, int>> arcs_sp, arcs_pt, arcs_st;

  std::size_t n_flows() const { return arcs_sp.size() + arcs_pt.size() + arcs_st.size(); }
  std::size_t n_q() const { return pools.size() * static_cast<std::size_t>(n_qualities); }
  /// Index of the flow variable of an arc in the flow vector: source->pool arcs first, then pool->terminal, then source->terminal.
  std::size_t idx_sp(std::size_t a) const { return a; }
  std::size_t idx_pt(std::size_t a) const { return arcs_sp.size() + a; }
  std::size_t idx_st(std::size_t a) const { return arcs_sp.size() + arcs_pt.size() + a; }
  std::size_t qidx(std::size_t pool, int k) const { return pool * static_cast<std::size_t>(n_qualities) + static_cast<std::size_t>(k); }
  std::string arc_name_sp(std::size_t a) const;
  std::string arc_name_pt(std::size_t a) const;
  std::string arc_name_st(std::size_t a) const;
};

struct PoolParseResult {
  bool ok = false;
  std::string error;  ///< "<source>:<line>: message"
  PoolProblem problem;
};

PoolParseResult parse_pool_string(const std::string& text, const std::string& source_name);
PoolParseResult parse_pool_file(const std::string& path);

/// A point of the nonlinear model: flows (arc order of PoolProblem::idx_*) and pool qualities (pool-major).
struct PoolPoint {
  std::vector<double> flow;
  std::vector<double> q;
};

/// Objective (revenue minus cost) of the flows.
double pool_objective(const PoolProblem& p, const std::vector<double>& flow);

/// Residuals of the ORIGINAL nonlinear model at a point, each relative to the size of the terms behind it
/// (residual / (1 + sum of the absolute terms)); `worst` is the largest of them.
struct PoolCheck {
  double bounds = 0.0;         ///< negative flows, pool qualities outside the range of the source qualities
  double supply = 0.0;
  double capacity = 0.0;
  double demand = 0.0;
  double material = 0.0;       ///< pool material balance
  double quality_balance = 0.0;  ///< pool quality balance (bilinear)
  double terminal_spec = 0.0;  ///< terminal quality (bilinear)
  double worst = 0.0;
  bool ok(double tol) const { return worst <= tol; }
};
PoolCheck check_pool_point(const PoolProblem& p, const PoolPoint& pt);

/// Pool qualities that the flows imply: sum_s q_sk f_sp / sum_s f_sp where the pool has throughput, `previous` elsewhere.
std::vector<double> implied_pool_qualities(const PoolProblem& p, const std::vector<double>& flow, const std::vector<double>& previous);

/// Lower and upper bound of the quality of pool `pool` (the range of the source qualities that can reach it).
std::pair<double, double> pool_quality_range(const PoolProblem& p, std::size_t pool, int k);

enum class PoolMethod { Recursion, Slp };

enum class PoolStatus {
  Converged,     ///< a point that satisfies the nonlinear model within the tolerance was found and the method converged
  NoSolution,    ///< no verified point (the methods failed, or every start cycled)
  Cycling,       ///< the recursion revisited its iterates: no point is returned
  IterationLimit,
  NumericalError,
};
const char* to_string(PoolStatus s) noexcept;

struct PoolOptions {
  PoolMethod method = PoolMethod::Recursion;
  int starts = 1;                 ///< initial pool qualities tried (the first is the midpoint of the ranges, the rest are seeded random)
  double tol = 1e-6;              ///< convergence and verification tolerance (relative residuals, see PoolCheck)
  unsigned long long seed = 1;
  int max_iterations = 200;       ///< per start
  double damping = 0.5;           ///< recursion: q <- q + damping (q_implied - q); 1 = undamped
  double trust_radius = 0.25;     ///< SLP: initial trust region on the pool qualities, relative to the range of each
  double penalty = 0.0;           ///< SLP: penalty of the linearization residual; 0 = chosen from the prices
  bool mccormick = true;          ///< compute the McCormick upper bound
  double time_limit = kInf;       ///< seconds
  std::ostream* log = nullptr;
};

struct PoolRun {
  PoolStatus status = PoolStatus::NoSolution;
  PoolPoint point;
  double objective = 0.0;
  int iterations = 0;
  double violation = 0.0;  ///< PoolCheck::worst of the returned point
  std::string message;
};

struct PoolResult {
  PoolStatus status = PoolStatus::NoSolution;
  PoolPoint point;       ///< the best verified point (valid when status is Converged)
  double objective = 0.0;
  PoolCheck check;       ///< residuals of the returned point
  bool has_bound = false;
  double mccormick_bound = 0.0;   ///< upper bound of the objective (maximization), from the McCormick relaxation
  double gap = 0.0;               ///< (bound - objective) / max(1, |bound|)
  int starts_run = 0, starts_converged = 0, starts_cycled = 0, starts_failed = 0;
  /// Objective values (rounded to the tolerance) of the converged starts with the number of starts that ended there.
  std::vector<std::pair<double, int>> outcomes;
  int iterations = 0;
  std::string message;
};

/// A local method from one initial pool-quality vector (pool-major, size n_q).
PoolRun solve_recursion(const PoolProblem& p, const std::vector<double>& q0, const PoolOptions& options);
PoolRun solve_slp(const PoolProblem& p, const std::vector<double>& q0, const PoolOptions& options);

/// Multi-start of the chosen method; keeps the best verified point; computes the McCormick bound.
PoolResult solve_pool(const PoolProblem& p, const PoolOptions& options);

/// McCormick upper bound alone. Returns false when the LP could not be solved to an accepted optimum.
bool mccormick_upper_bound(const PoolProblem& p, double* bound, std::string* message);

/// A pooling certificate (docs/CERTIFICATES.md): status "feasible" with the flows and pool qualities, the claimed objective,
/// the McCormick bound as `claimed_upper_bound` when there is one, `optimality_certified: false` and `nonconvex: true`;
/// status "other" without a body when no verified point exists. `file_sha256` is the SHA-256 of the exact bytes of the
/// .pool file. Returns the status string written.
std::string write_pool_certificate(const PoolProblem& p, const PoolResult& r, const PoolOptions& options, const std::string& file_sha256,
                                   std::ostream& out);

/// The fixed-quality LP (the pool quality equality is not enforced; q is a constant in the terminal rows): the value
/// and flows of its optimum. Used by the recursion and exposed for tests.
bool solve_fixed_quality_lp(const PoolProblem& p, const std::vector<double>& q, std::vector<double>* flow, double* objective);

}  // namespace shodhan::pooling
