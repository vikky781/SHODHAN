#pragma once

// Cutting planes for the root of the branch and bound (docs/CUTS.md).
//
// A cut is a valid inequality  sum val[k] * x[idx[k]] <= rhs  in the PRESOLVED, UNSCALED, minimization model
// that the search works on. "Valid" means every integer-feasible point of that model (with its continuous part
// free) satisfies it; a cut that cuts off such a point is a bug. Separators produce candidate cuts from the
// current LP solution, the CutPool cleans and filters them, and the root loop adds the selected ones to the
// simplex engine as rows.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"
#include "shodhan/mip/options.hpp"
#include "shodhan/presolve.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"
#include "shodhan/structure.hpp"

namespace shodhan::mip {

struct Cut {
  std::vector<Index> idx;
  std::vector<double> val;
  double rhs = 0.0;
  int separator = -1;     ///< SeparatorId
  double efficacy = 0.0;  ///< violation at the LP point divided by the Euclidean norm of the coefficients
};

enum SeparatorId { kSepGomory = 0, kSepMir, kSepCover, kSepClique, kSepImpliedBound, kNumSeparators };
const char* separator_name(int id) noexcept;

/// Per-separator counters of a whole cut loop.
struct SeparatorStats {
  long long calls = 0;
  long long generated = 0;       ///< raw candidates returned by the separator
  long long rejected_numerics = 0;   ///< empty, dynamism too large, unbounded relaxation of a tiny term
  long long rejected_efficacy = 0;   ///< not violated enough at the LP point
  long long rejected_density = 0;
  long long rejected_parallel = 0;   ///< almost parallel to a better cut of the same round
  long long candidates = 0;      ///< passed cleaning and efficacy (the ones the validity test checks)
  long long added = 0;           ///< rows added to the LP
  double seconds = 0.0;
};

struct CutStats {
  SeparatorStats sep[kNumSeparators];
  int rounds = 0;
  long long cuts_added = 0;
  long long cuts_removed = 0;   ///< removed again because they stayed slack
  long long cuts_kept = 0;      ///< rows still in the LP when the loop ended
  double root_bound_before = 0.0;  ///< LP bound before the first round (minimization form, presolved units)
  double root_bound_after = 0.0;
  bool lp_failed = false;       ///< a resolve failed numerically and the cuts were abandoned
  bool infeasible = false;      ///< the LP with cuts is infeasible
  long long lp_iterations = 0;
  double seconds = 0.0;
  std::string stopped_because;
};

/// What separators see. All pointers are non-null except `structure` and `engine`/`scaling` (null: the
/// separators that need them are skipped).
struct CutData {
  const LpModel* model = nullptr;        ///< presolved, unscaled, minimization, WITHOUT cuts
  const CsrMatrix* rows = nullptr;       ///< row-wise copy of model->A
  const std::vector<double>* lo = nullptr;  ///< global column bounds (integer bounds integral)
  const std::vector<double>* hi = nullptr;
  const std::vector<double>* x = nullptr;   ///< LP solution, unscaled, all columns
  const MipPresolveInfo* structure = nullptr;
  const StructureInfo* detected = nullptr;  ///< variable-upper-bound and balance rows of the model (null: not used)
  SimplexEngine* engine = nullptr;       ///< at the LP optimum; used for tableau rows (Gomory)
  const Scaling* scaling = nullptr;      ///< scaling between the engine's space and the unscaled one
  const MipOptions* options = nullptr;
};

class Separator {
 public:
  virtual ~Separator() = default;
  virtual int id() const = 0;
  /// Appends raw candidate cuts violated (or nearly) by data.x. Must only return valid inequalities.
  virtual void separate(const CutData& data, std::vector<Cut>& out) = 0;
};

/// The separators enabled by the options, in a fixed order.
std::vector<std::unique_ptr<Separator>> make_separators(const MipOptions& options);

/// Cleans one raw cut (see CutPool): sorts, removes zeros, relaxes away tiny coefficients with the column
/// bounds, rejects excessive dynamism, scales to a largest coefficient of 1, relaxes the right-hand side by
/// `options.cut_rhs_relaxation` (relative), computes the efficacy at x. Returns false (and counts the reason in
/// `stats`) when the cut is unusable.
bool clean_cut(Cut& cut, const std::vector<double>& lo, const std::vector<double>& hi, const std::vector<double>& x,
               const MipOptions& options, SeparatorStats& stats);

/// Chooses the cuts to add from cleaned candidates: efficacy first, at most `limit`, skipping cuts that are
/// nearly parallel (cosine above options.cut_max_parallelism) to an already chosen one. Returns indices.
std::vector<std::size_t> select_cuts(const std::vector<Cut>& candidates, Index n_cols, std::size_t limit,
                                     const MipOptions& options, CutStats& stats);

/// The row of the scaled engine for a cut (coefficient on column j times its column scale).
RowSpec cut_to_row(const Cut& cut, const Scaling& scaling);

/// Result of the root loop for the caller.
struct CutLoopResult {
  CutStats stats;
  /// Every candidate (cleaned, efficacious) of every round, only filled when `keep_candidates` was set.
  std::vector<Cut> candidates;
  /// The cuts that were added and are still in the LP at the end.
  std::vector<Cut> kept;
};

/// Outcome of a resolve, reported by the callback of the root loop.
enum class CutLpOutcome { Optimal, Infeasible, Failed };

struct CutLoopInput {
  const LpModel* model = nullptr;        ///< presolved, unscaled, minimization (no cuts)
  const std::vector<double>* lo = nullptr;
  const std::vector<double>* hi = nullptr;
  const MipPresolveInfo* structure = nullptr;
  const StructureInfo* detected = nullptr;
  SimplexEngine* engine = nullptr;       ///< at the optimum of the LP without cuts
  const Scaling* scaling = nullptr;
  const MipOptions* options = nullptr;
  /// Resolves the engine (warm) and returns the outcome. Supplied by the caller so that it can apply its time
  /// limit and retry policy.
  std::function<CutLpOutcome()> resolve;
  std::function<bool()> time_up;
  bool keep_candidates = false;
};

/// The root cut loop (docs/CUTS.md): separate, filter, add, resolve, until no violated cut is found, the bound
/// stalls for options.cut_stall_rounds rounds, or options.cut_rounds is reached. Cuts that stay slack for
/// options.cut_age_limit rounds are removed; at the end all slack cuts are removed. On return the engine holds
/// the LP with the kept cuts and is at its optimum (unless stats.lp_failed / stats.infeasible).
CutLoopResult run_root_cut_loop(const CutLoopInput& in);

}  // namespace shodhan::mip
