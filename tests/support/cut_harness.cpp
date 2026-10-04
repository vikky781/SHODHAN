#include "support/cut_harness.hpp"

#include <cmath>

#include "shodhan/presolve.hpp"
#include "shodhan/scaling.hpp"
#include "shodhan/simplex_engine.hpp"
#include "support/dense_ref_lp.hpp"

namespace shodhan::testing {

LpModel min_form(const LpModel& m) {
  LpModel out = m;
  if (m.sense == Sense::Maximize) {
    for (double& c : out.col_cost) c = -c;
    out.objective_offset = -out.objective_offset;
    out.sense = Sense::Minimize;
  }
  return out;
}

CutRunResult run_cut_loop_on(const LpModel& model, const mip::MipOptions& options, bool use_structure,
                             bool keep_candidates) {
  CutRunResult r;
  r.pm = min_form(model);
  const LpModel& pm = r.pm;
  r.lo = pm.col_lower;
  r.hi = pm.col_upper;
  for (Index j = 0; j < pm.n_cols; ++j) {
    if (!pm.is_integer(j)) continue;
    if (!is_inf(r.lo[to_size(j)])) r.lo[to_size(j)] = std::ceil(r.lo[to_size(j)] - 1e-9);
    if (!is_inf(r.hi[to_size(j)])) r.hi[to_size(j)] = std::floor(r.hi[to_size(j)] + 1e-9);
  }
  Scaling sc = compute_scaling(pm);
  const LpModel sm = apply_scaling(pm, sc);
  SimplexOptions so;
  SimplexEngine engine(sm, so);
  if (engine.solve() != EngineStatus::Optimal) return r;
  r.ran = true;
  r.root_bound = engine.objective() / sc.obj_scale;
  for (Index j = 0; j < pm.n_cols; ++j) {
    const double v = engine.primal_all()[to_size(j)] * sc.col_scale[to_size(j)];
    if (pm.is_integer(j) && std::fabs(v - std::round(v)) > options.params.int_tol) r.root_fractional = true;
  }
  MipPresolveInfo structure;
  if (use_structure) {
    PresolveOptions po;
    po.is_mip = true;
    structure = find_mip_structure(pm, po);
  }
  mip::CutLoopInput in;
  in.model = &pm;
  in.lo = &r.lo;
  in.hi = &r.hi;
  in.structure = use_structure ? &structure : nullptr;
  in.engine = &engine;
  in.scaling = &sc;
  in.options = &options;
  in.keep_candidates = keep_candidates;
  in.resolve = [&]() {
    const EngineStatus st = engine.solve();
    if (st == EngineStatus::Optimal) return mip::CutLpOutcome::Optimal;
    if (st == EngineStatus::Infeasible) return mip::CutLpOutcome::Infeasible;
    return mip::CutLpOutcome::Failed;
  };
  r.loop = mip::run_root_cut_loop(in);
  r.bound_after = r.loop.stats.root_bound_after;
  return r;
}

ValidityResult check_cuts_valid(const LpModel& pm, const std::vector<mip::Cut>& cuts, long long max_assignments) {
  ValidityResult res;
  const Index n = pm.n_cols;
  std::vector<Index> ints, conts;
  long long combos = 1;
  for (Index j = 0; j < n; ++j) {
    if (pm.is_integer(j)) {
      const double lo = std::ceil(pm.col_lower[to_size(j)] - 1e-9), hi = std::floor(pm.col_upper[to_size(j)] + 1e-9);
      if (is_inf(lo) || is_inf(hi)) return res;
      combos *= static_cast<long long>(hi - lo) + 1;
      if (combos > max_assignments || combos <= 0) return res;
      ints.push_back(j);
    } else {
      conts.push_back(j);
    }
  }
  res.enumerated = true;
  const CsrMatrix rows = pm.A.to_csr();
  std::vector<double> x(to_size(n), 0.0);
  for (const Index j : ints) x[to_size(j)] = std::ceil(pm.col_lower[to_size(j)] - 1e-9);
  std::vector<double> act(to_size(pm.n_rows));
  for (long long c = 0; c < combos; ++c) {
    // Feasibility of the integer part and, when there are continuous columns, of the whole point.
    bool feasible = true;
    LpModel fixed;
    if (conts.empty()) {
      for (Index i = 0; i < pm.n_rows && feasible; ++i) {
        double a = 0.0;
        for (Index t = rows.row_start[to_size(i)]; t < rows.row_start[to_size(i) + 1]; ++t) {
          a += rows.value[to_size(t)] * x[to_size(rows.col_index[to_size(t)])];
        }
        const double tol = 1e-9 * (1.0 + std::fabs(a));
        feasible = a >= pm.row_lower[to_size(i)] - tol && a <= pm.row_upper[to_size(i)] + tol;
      }
    } else {
      fixed = pm;
      for (const Index j : ints) {
        fixed.col_lower[to_size(j)] = fixed.col_upper[to_size(j)] = x[to_size(j)];
        fixed.col_type[to_size(j)] = ColType::Continuous;
      }
      for (Index j = 0; j < n; ++j) fixed.col_cost[to_size(j)] = 0.0;
      fixed.objective_offset = 0.0;
      const RefLpResult feas = solve_dense_lp(fixed);
      if (feas.status == Status::Infeasible) feasible = false;
      else if (feas.status != Status::Optimal && feas.status != Status::Unbounded) {
        ++res.inconclusive;
        feasible = false;
      }
    }
    if (feasible) {
      ++res.feasible_points;
      for (std::size_t k = 0; k < cuts.size(); ++k) {
        const mip::Cut& cut = cuts[k];
        bool has_cont = false;
        double lhs_int = 0.0;
        for (std::size_t q = 0; q < cut.idx.size(); ++q) {
          if (pm.is_integer(cut.idx[q])) lhs_int += cut.val[q] * x[to_size(cut.idx[q])];
          else has_cont = true;
        }
        double lhs;
        double tol_scale = 1e-9;
        if (!has_cont) {
          lhs = lhs_int;
        } else {
          LpModel obj = fixed;
          for (Index j = 0; j < n; ++j) obj.col_cost[to_size(j)] = 0.0;
          for (std::size_t q = 0; q < cut.idx.size(); ++q) {
            if (!pm.is_integer(cut.idx[q])) obj.col_cost[to_size(cut.idx[q])] = -cut.val[q];  // maximize the cut
          }
          const RefLpResult r = solve_dense_lp(obj);
          if (r.status == Status::Unbounded) {
            res.violated_cut = static_cast<int>(k);
            res.detail = "the cut is unbounded above over the continuous part of a feasible integer point";
            return res;
          }
          if (r.status != Status::Optimal) {
            ++res.inconclusive;
            continue;
          }
          lhs = lhs_int - r.solution.objective;
          tol_scale = 1e-7;
        }
        ++res.checks;
        const double viol = (lhs - cut.rhs) / (1.0 + std::fabs(cut.rhs));
        if (viol > res.worst_violation) res.worst_violation = viol;
        if (viol > tol_scale) {
          res.violated_cut = static_cast<int>(k);
          res.detail = "violated by " + std::to_string(lhs - cut.rhs);
          return res;
        }
      }
    }
    for (const Index j : ints) {  // odometer over the integer columns
      if (x[to_size(j)] < std::floor(pm.col_upper[to_size(j)] + 1e-9)) {
        x[to_size(j)] += 1.0;
        break;
      }
      x[to_size(j)] = std::ceil(pm.col_lower[to_size(j)] - 1e-9);
    }
  }
  return res;
}

}  // namespace shodhan::testing
