#pragma once

#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/options.hpp"
#include "shodhan/mip/result.hpp"
#include "shodhan/presolve.hpp"

namespace shodhan::mip {

enum class SubmitOutcome { Accepted, NotBetter, Rejected };

/// Largest relative violation of the bounds of `model` by `x` (relative form |violation| / (1 + |bound|), the
/// same measure as KASAUTI and check_kkt), and the largest distance of an integer column from an integer.
struct PointCheck {
  double max_row_violation = 0.0;
  double max_column_violation = 0.0;
  double max_integrality_violation = 0.0;
  bool ok(double primal_tol, double int_tol) const {
    return max_row_violation <= primal_tol && max_column_violation <= primal_tol && max_integrality_violation <= int_tol;
  }
};
PointCheck check_point(const LpModel& model, const std::vector<double>& x);

/// The single entry point for new solutions from every source (node LPs, rounding, diving, Feasibility Jump,
/// the trivial heuristic, presolve). A candidate is
///   1. taken in the presolved (unscaled) space,
///   2. mapped back to the original space (postsolve),
///   3. snapped: every integer column to the nearest integer,
///   4. completed: if the ORIGINAL model has continuous columns, the LP with the integer columns fixed is
///      re-solved so that the continuous values are consistent (skipped when there are none),
///   5. verified against the ORIGINAL model (rows and columns with primal_tol in the relative form, integrality
///      exact after snapping),
///   6. accepted only if verified and strictly better than the incumbent.
/// Rejected candidates are counted with a reason (RejectionStats).
class IncumbentManager {
 public:
  /// `stack` may be null (no presolve: the working space is the original space).
  IncumbentManager(const LpModel& original, const PostsolveStack* stack, const MipOptions& options);

  /// A candidate in the presolved space (values of the presolved model's columns).
  SubmitOutcome submit(const std::vector<double>& working_x, const std::string& source);
  /// A candidate already in the original space.
  SubmitOutcome submit_original(const std::vector<double>& original_x, const std::string& source);

  bool has_incumbent() const { return has_; }
  /// Objective in the minimization form (equal to the presolved model's objective): sense * objective.
  double objective_min() const { return objective_min_; }
  /// Objective in the model's own sense.
  double objective() const { return sense_ * objective_min_; }
  const std::vector<double>& x() const { return x_; }
  const std::string& source() const { return source_; }
  long long found() const { return found_; }
  long long submitted() const { return submitted_; }
  const RejectionStats& rejections() const { return rej_; }
  /// Number of candidates accepted per source name, in first-seen order.
  const std::vector<std::pair<std::string, long long>>& accepted_by_source() const { return by_source_; }

 private:
  SubmitOutcome finish(std::vector<double> x, const std::string& source, bool try_completion);

  const LpModel& original_;
  const PostsolveStack* stack_;
  MipOptions opt_;
  double sense_ = 1.0;
  bool has_ = false;
  double objective_min_ = kInf;
  std::vector<double> x_;
  std::string source_;
  long long found_ = 0, submitted_ = 0;
  RejectionStats rej_;
  std::vector<std::pair<std::string, long long>> by_source_;
  bool has_continuous_ = false;
};

}  // namespace shodhan::mip
