// Cleaning, safety filters and selection of cuts (docs/CUTS.md, "Safety filters").

#include <algorithm>
#include <cmath>
#include <numeric>

#include "shodhan/mip/cuts.hpp"

namespace shodhan::mip {

const char* separator_name(int id) noexcept {
  switch (id) {
    case kSepGomory: return "gomory-mixed-integer";
    case kSepMir: return "mixed-integer-rounding";
    case kSepCover: return "knapsack-cover";
    case kSepClique: return "clique";
    case kSepImpliedBound: return "implied-bound";
    default: return "unknown";
  }
}

namespace {

std::size_t u(Index i) { return static_cast<std::size_t>(i); }

}  // namespace

bool clean_cut(Cut& cut, const std::vector<double>& lo, const std::vector<double>& hi, const std::vector<double>& x,
               const MipOptions& options, SeparatorStats& stats) {
  // Sort by column and merge duplicates.
  std::vector<std::size_t> order(cut.idx.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return cut.idx[a] < cut.idx[b]; });
  std::vector<Index> idx;
  std::vector<double> val;
  for (const std::size_t k : order) {
    if (!std::isfinite(cut.val[k])) {
      ++stats.rejected_numerics;
      return false;
    }
    if (!idx.empty() && idx.back() == cut.idx[k]) {
      val.back() += cut.val[k];
    } else {
      idx.push_back(cut.idx[k]);
      val.push_back(cut.val[k]);
    }
  }
  double rhs = cut.rhs;
  if (!std::isfinite(rhs)) {
    ++stats.rejected_numerics;
    return false;
  }
  double maxabs = 0.0;
  for (const double v : val) maxabs = std::max(maxabs, std::fabs(v));
  if (maxabs == 0.0) {
    ++stats.rejected_numerics;
    return false;
  }
  // Tiny coefficients are removed by relaxing with the column bound that makes the term smallest:
  // sum_{others} <= rhs - g_j x_j <= rhs - min(g_j x_j). Needs that bound to be finite.
  const double tiny = 1e-9 * maxabs;
  std::vector<Index> idx2;
  std::vector<double> val2;
  for (std::size_t k = 0; k < idx.size(); ++k) {
    const double g = val[k];
    if (g == 0.0) continue;
    if (std::fabs(g) >= tiny) {
      idx2.push_back(idx[k]);
      val2.push_back(g);
      continue;
    }
    const double b = g > 0.0 ? lo[u(idx[k])] : hi[u(idx[k])];
    if (is_inf(b)) {
      ++stats.rejected_numerics;
      return false;
    }
    rhs -= g * b;
  }
  if (idx2.empty()) {
    ++stats.rejected_numerics;
    return false;
  }
  double mx = 0.0, mn = kInf;
  for (const double v : val2) {
    mx = std::max(mx, std::fabs(v));
    mn = std::min(mn, std::fabs(v));
  }
  if (mx / mn > options.cut_max_dynamism) {
    ++stats.rejected_numerics;
    return false;
  }
  const double n_cols = static_cast<double>(x.size());
  if (n_cols > 40.0 && static_cast<double>(idx2.size()) > options.cut_max_density * n_cols) {
    ++stats.rejected_density;
    return false;
  }
  // Scale to a largest coefficient of 1 and relax the right-hand side against rounding.
  for (double& v : val2) v /= mx;
  rhs /= mx;
  rhs += options.cut_rhs_relaxation * (1.0 + std::fabs(rhs));
  double act = 0.0, nrm = 0.0;
  for (std::size_t k = 0; k < idx2.size(); ++k) {
    act += val2[k] * x[u(idx2[k])];
    nrm += val2[k] * val2[k];
  }
  nrm = std::sqrt(nrm);
  const double eff = (act - rhs) / nrm;
  if (!(eff > options.cut_min_efficacy)) {
    ++stats.rejected_efficacy;
    return false;
  }
  cut.idx = std::move(idx2);
  cut.val = std::move(val2);
  cut.rhs = rhs;
  cut.efficacy = eff;
  return true;
}

std::vector<std::size_t> select_cuts(const std::vector<Cut>& cand, Index n_cols, std::size_t limit,
                                     const MipOptions& options, CutStats& stats) {
  std::vector<std::size_t> order(cand.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return cand[a].efficacy > cand[b].efficacy;
  });
  std::vector<std::size_t> chosen;
  std::vector<double> dense(u(n_cols), 0.0);
  std::vector<double> norms;
  for (const std::size_t k : order) {
    if (chosen.size() >= limit) break;
    const Cut& c = cand[k];
    double nrm = 0.0;
    for (const double v : c.val) nrm += v * v;
    nrm = std::sqrt(nrm);
    bool parallel = false;
    for (std::size_t q = 0; q < chosen.size() && !parallel; ++q) {
      const Cut& o = cand[chosen[q]];
      // Cosine of the two coefficient vectors (same orientation only: an opposite pair describes a range).
      double dot = 0.0;
      for (std::size_t t = 0; t < o.idx.size(); ++t) dense[u(o.idx[t])] = o.val[t];
      for (std::size_t t = 0; t < c.idx.size(); ++t) dot += c.val[t] * dense[u(c.idx[t])];
      for (std::size_t t = 0; t < o.idx.size(); ++t) dense[u(o.idx[t])] = 0.0;
      if (dot / (nrm * norms[q]) > options.cut_max_parallelism) parallel = true;
    }
    if (parallel) {
      ++stats.sep[c.separator].rejected_parallel;
      continue;
    }
    chosen.push_back(k);
    norms.push_back(nrm);
  }
  return chosen;
}

RowSpec cut_to_row(const Cut& cut, const Scaling& scaling) {
  RowSpec r;
  r.idx = cut.idx;
  r.val.resize(cut.val.size());
  for (std::size_t k = 0; k < cut.val.size(); ++k) r.val[k] = cut.val[k] * scaling.col_scale[u(cut.idx[k])];
  r.lo = -kInf;
  r.hi = cut.rhs;
  return r;
}

}  // namespace shodhan::mip
