#include "shodhan/presolve.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "mip_reductions.hpp"
#include "reductions.hpp"

namespace shodhan {

using presolve_detail::Activity;
using presolve_detail::Context;
using presolve_detail::WorkModel;

namespace {

std::size_t u(int i) { return static_cast<std::size_t>(i); }

}  // namespace

const char* to_string(PresolveStatus status) noexcept {
  switch (status) {
    case PresolveStatus::Reduced:
      return "Reduced";
    case PresolveStatus::SolvedByPresolve:
      return "SolvedByPresolve";
    case PresolveStatus::Infeasible:
      return "Infeasible";
    case PresolveStatus::Unbounded:
      return "Unbounded";
    case PresolveStatus::InfeasibleOrUnbounded:
      return "InfeasibleOrUnbounded";
  }
  return "Unknown";
}

std::vector<std::pair<std::string, int>> PresolveStats::reduction_counts() const {
  return {
      {"empty rows", empty_rows},
      {"empty columns", empty_columns},
      {"fixed columns", fixed_columns},
      {"singleton rows", singleton_rows},
      {"redundant rows", redundant_rows},
      {"redundant row sides", redundant_row_sides},
      {"forcing rows", forcing_rows},
      {"doubleton equations", doubleton_equations},
      {"dual-fixed columns", dual_fixed_columns},
      {"integer bounds rounded", integer_bounds_rounded},
      {"integer bounds tightened", integer_bounds_tightened},
      {"implied-bound checks", implied_bound_checks},
      {"improving columns dropped", unbounded_columns},
      {"bounds propagated", propagated_bounds},
      {"coefficients tightened", coefficients_tightened},
      {"probing fixings", probing_fixings},
      {"probing bounds", probing_bounds},
      {"implications", implications},
      {"cliques", cliques},
      {"parallel rows merged", parallel_rows},
      {"duplicate columns merged", duplicate_columns},
      {"dominated columns fixed", dominated_columns},
  };
}

namespace {

bool process_col(Context& c, int j) {
  WorkModel& w = c.w;
  const PresolveOptions& o = c.opt;
  if (!w.col_alive[u(j)]) return false;
  bool changed = false;
  if (o.is_mip && o.integer_bounds) {
    changed = presolve_detail::IntegerBoundsReduction::apply(c, j) || changed;
    if (c.infeasible) return true;
  }
  if (o.empty_columns && w.col_cnt[u(j)] == 0) {
    return presolve_detail::EmptyColumnReduction::apply(c, j) || changed;
  }
  if (o.fixed_columns && presolve_detail::FixedColumnReduction::apply(c, j)) return true;
  if (c.infeasible) return true;
  if (o.dual_fixing && presolve_detail::DualFixingReduction::apply(c, j)) return true;
  return changed;
}

bool process_row(Context& c, int i) {
  WorkModel& w = c.w;
  const PresolveOptions& o = c.opt;
  if (!w.row_alive[u(i)]) return false;
  if (w.row_cnt[u(i)] == 0) {
    return o.empty_rows ? presolve_detail::EmptyRowReduction::apply(c, i) : false;
  }
  if (o.singleton_rows && w.row_cnt[u(i)] == 1) {
    return presolve_detail::SingletonRowReduction::apply(c, i);
  }
  const Activity act = w.activity(i);
  if (presolve_detail::RedundantRowReduction::detect_infeasible(c, i, act)) return true;
  bool changed = false;
  if (o.integer_bounds) {
    changed = presolve_detail::ImpliedBoundsReduction::apply(c, i, act) || changed;
    if (c.infeasible) return true;
  }
  if (o.forcing_rows && presolve_detail::ForcingRowReduction::apply(c, i, act)) return true;
  if (o.redundant_rows && presolve_detail::RedundantRowReduction::apply(c, i, act)) return true;
  if (o.doubleton_equations && w.row_cnt[u(i)] == 2 &&
      presolve_detail::DoubletonEquationReduction::apply(c, i)) {
    return true;
  }
  return changed;
}

// Builds the reduced model from the surviving rows and columns.
void build_reduced(const LpModel& original, WorkModel& w, PresolveResult* res) {
  w.compact();
  std::vector<int> row_new(u(w.m), -1);
  std::vector<int> col_new(u(w.n), -1);
  PostsolveStack& st = res->stack;
  for (int i = 0; i < w.m; ++i) {
    if (!w.row_alive[u(i)]) continue;
    row_new[u(i)] = static_cast<int>(st.row_map.size());
    st.row_map.push_back(i);
  }
  for (int j = 0; j < w.n; ++j) {
    if (!w.col_alive[u(j)]) continue;
    col_new[u(j)] = static_cast<int>(st.col_map.size());
    st.col_map.push_back(j);
  }
  LpModel& r = res->reduced;
  r.name = original.name;
  r.objective_name = original.objective_name;
  r.sense = Sense::Minimize;
  r.objective_offset = w.offset;
  r.n_rows = static_cast<Index>(st.row_map.size());
  r.n_cols = static_cast<Index>(st.col_map.size());
  std::vector<Triplet> triplets;
  for (const Index oj : st.col_map) {
    const int j = oj;
    w.for_col(j, [&](int i, double a) { triplets.push_back({row_new[u(i)], col_new[u(j)], a}); });
    r.col_cost.push_back(w.cost[u(j)]);
    r.col_lower.push_back(w.cl[u(j)]);
    r.col_upper.push_back(w.cu[u(j)]);
    ColType t = original.col_type[u(j)];
    if (t == ColType::Binary && (w.cl[u(j)] != 0.0 || w.cu[u(j)] != 1.0)) t = ColType::Integer;
    r.col_type.push_back(t);
  }
  for (const Index oi : st.row_map) {
    r.row_lower.push_back(w.rl[u(oi)]);
    r.row_upper.push_back(w.ru[u(oi)]);
  }
  if (!original.row_names.empty()) {
    for (const Index oi : st.row_map) r.row_names.push_back(original.row_names[u(oi)]);
  }
  if (!original.col_names.empty()) {
    for (const Index oj : st.col_map) r.col_names.push_back(original.col_names[u(oj)]);
  }
  std::string err;
  if (!SparseMatrix::from_triplets(r.n_rows, r.n_cols, std::move(triplets), &r.A, &err)) {
    throw std::logic_error("presolve: internal error building the reduced matrix: " + err);
  }
}

}  // namespace

PresolveResult presolve(const LpModel& model, const PresolveOptions& options) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<std::string> problems = model.validate();
  if (!problems.empty()) throw std::invalid_argument("presolve: invalid model: " + problems.front());

  PresolveResult res;
  PresolveStats& stats = res.stats;
  stats.rows_before = model.n_rows;
  stats.cols_before = model.n_cols;
  stats.nnz_before = model.A.nnz();
  PostsolveStack& st = res.stack;
  st.n_rows = model.n_rows;
  st.n_cols = model.n_cols;
  st.original_cost = model.col_cost;
  st.original_offset = model.objective_offset;
  st.compute_duals = options.need_duals && !options.is_mip;

  WorkModel w(model);
  Context ctx{w, options, st, stats};

  auto finish = [&](PresolveStatus status) {
    res.status = status;
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return std::move(res);
  };

  // The standard passes, to a fixpoint (or max_passes). False when infeasibility was found.
  auto run_standard = [&]() {
    for (int pass = 0; pass < options.max_passes && w.any_dirty(); ++pass) {
      ++stats.passes;
      w.compact();
      for (int j = 0; j < w.n && !ctx.infeasible; ++j) {
        if (!w.col_alive[u(j)] || !w.col_dirty[u(j)]) continue;
        w.col_dirty[u(j)] = 0;
        process_col(ctx, j);
      }
      for (int i = 0; i < w.m && !ctx.infeasible; ++i) {
        if (!w.row_alive[u(i)] || !w.row_dirty[u(i)]) continue;
        w.row_dirty[u(i)] = 0;
        process_row(ctx, i);
      }
      if (ctx.infeasible) return false;
    }
    return true;
  };
  auto infeasible_result = [&]() {
    st.records.clear();
    res.note = ctx.reason;
    return finish(PresolveStatus::Infeasible);
  };

  if (!run_standard()) return infeasible_result();
  presolve_detail::MipWork mip_work;
  if (options.is_mip && !ctx.unbounded) {
    for (int round = 0; round < options.mip_rounds; ++round) {
      const bool changed = presolve_detail::run_mip_round(ctx, mip_work);
      if (ctx.infeasible) return infeasible_result();
      if (!changed) break;
      if (!run_standard()) return infeasible_result();
    }
  }
  MipPresolveInfo structure;
  if (options.is_mip && !ctx.unbounded) structure = presolve_detail::collect_mip_structure(ctx, mip_work);

  const bool all_gone = w.alive_rows() == 0 && w.alive_cols() == 0;
  if (ctx.unbounded) {
    st.records.clear();
    return finish(all_gone ? PresolveStatus::Unbounded : PresolveStatus::InfeasibleOrUnbounded);
  }
  build_reduced(model, w, &res);
  if (options.is_mip) {
    std::vector<int> col_new(u(w.n), -1);
    for (std::size_t k = 0; k < st.col_map.size(); ++k) col_new[u(st.col_map[k])] = static_cast<int>(k);
    for (Implication im : structure.implications) {
      if (col_new[u(im.var)] < 0 || col_new[u(im.other)] < 0) continue;
      im.var = col_new[u(im.var)];
      im.other = col_new[u(im.other)];
      res.mip.implications.push_back(im);
    }
    for (const auto& clique : structure.cliques.cliques) {
      std::vector<int> lits;
      for (const int l : clique) {
        const int cn = col_new[u(l >> 1)];
        if (cn < 0) break;
        lits.push_back(2 * cn + (l & 1));
      }
      if (lits.size() == clique.size()) res.mip.cliques.cliques.push_back(std::move(lits));
    }
    stats.implications = static_cast<int>(res.mip.implications.size());
    stats.cliques = static_cast<int>(res.mip.cliques.cliques.size());
  }
  stats.rows_after = res.reduced.n_rows;
  stats.cols_after = res.reduced.n_cols;
  stats.nnz_after = res.reduced.A.nnz();
  return finish(all_gone ? PresolveStatus::SolvedByPresolve : PresolveStatus::Reduced);
}

Solution postsolve(const PostsolveStack& st, const Solution& reduced) {
  const std::size_t nr = st.row_map.size();
  const std::size_t nc = st.col_map.size();
  if (reduced.x.size() != nc) {
    throw std::invalid_argument("postsolve: reduced solution has " + std::to_string(reduced.x.size()) +
                                " primal values, expected " + std::to_string(nc));
  }
  presolve_detail::PostsolveState s;
  s.x.assign(to_size(st.n_cols), 0.0);
  s.y.assign(to_size(st.n_rows), 0.0);
  s.d.assign(to_size(st.n_cols), 0.0);
  s.duals = st.compute_duals && reduced.y.size() == nr && reduced.d.size() == nc;
  for (std::size_t j = 0; j < nc; ++j) {
    s.x[to_size(st.col_map[j])] = reduced.x[j];
    if (s.duals) s.d[to_size(st.col_map[j])] = reduced.d[j];
  }
  if (s.duals) {
    for (std::size_t i = 0; i < nr; ++i) s.y[to_size(st.row_map[i])] = reduced.y[i];
  }
  for (auto it = st.records.rbegin(); it != st.records.rend(); ++it) (*it)->undo(s);

  Solution out;
  out.x = std::move(s.x);
  if (s.duals) {
    out.y = std::move(s.y);
    out.d = std::move(s.d);
  }
  double obj = st.original_offset;
  for (std::size_t j = 0; j < out.x.size(); ++j) obj += st.original_cost[j] * out.x[j];
  out.objective = obj;
  return out;
}

}  // namespace shodhan
