#include "factor_bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/sparse_work.hpp"

namespace shodhan::cli {

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;

// Small deterministic generator (splitmix64) so that runs are reproducible.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
  Index below(Index n) { return static_cast<Index>(next() % static_cast<std::uint64_t>(n)); }

 private:
  std::uint64_t state_;
};

// Products with the basis matrix B (columns of [A | -I]).
class BasisOps {
 public:
  BasisOps(const SparseMatrix& A, const std::vector<Index>& basis) : A_(A), basis_(basis) { rebuild_norms(); }

  void set_basis(const std::vector<Index>& basis) {
    basis_ = basis;
    rebuild_norms();
  }

  // y = B x, x by position, y by row.
  void multiply(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(basis_.size(), 0.0);
    for (std::size_t p = 0; p < basis_.size(); ++p) {
      const Index v = basis_[p];
      if (v < A_.n_cols) {
        for (Index t = A_.col_start[to_size(v)]; t < A_.col_start[to_size(v) + 1]; ++t) {
          y[to_size(A_.row_index[to_size(t)])] += A_.value[to_size(t)] * x[p];
        }
      } else {
        y[to_size(v - A_.n_cols)] -= x[p];
      }
    }
  }

  // y = B^T x, x by row, y by position.
  void multiply_transpose(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(basis_.size(), 0.0);
    for (std::size_t p = 0; p < basis_.size(); ++p) {
      const Index v = basis_[p];
      double s = 0.0;
      if (v < A_.n_cols) {
        for (Index t = A_.col_start[to_size(v)]; t < A_.col_start[to_size(v) + 1]; ++t) {
          s += A_.value[to_size(t)] * x[to_size(A_.row_index[to_size(t)])];
        }
      } else {
        s = -x[to_size(v - A_.n_cols)];
      }
      y[p] = s;
    }
  }

  double norm_inf() const { return norm_inf_; }  // max absolute row sum
  double norm_one() const { return norm_one_; }  // max absolute column sum

 private:
  void rebuild_norms() {
    std::vector<double> row_sum(basis_.size(), 0.0);
    norm_one_ = 0.0;
    for (std::size_t p = 0; p < basis_.size(); ++p) {
      const Index v = basis_[p];
      double col_sum = 0.0;
      if (v < A_.n_cols) {
        for (Index t = A_.col_start[to_size(v)]; t < A_.col_start[to_size(v) + 1]; ++t) {
          const double a = std::fabs(A_.value[to_size(t)]);
          row_sum[to_size(A_.row_index[to_size(t)])] += a;
          col_sum += a;
        }
      } else {
        row_sum[to_size(v - A_.n_cols)] += 1.0;
        col_sum = 1.0;
      }
      norm_one_ = std::max(norm_one_, col_sum);
    }
    norm_inf_ = 0.0;
    for (const double s : row_sum) norm_inf_ = std::max(norm_inf_, s);
  }

  const SparseMatrix& A_;
  std::vector<Index> basis_;
  double norm_inf_ = 0.0;
  double norm_one_ = 0.0;
};

double inf_norm(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// ||b_times_x - a|| / (||B|| ||x|| + ||a||)
double relative_residual(const std::vector<double>& bx, double b_norm, const std::vector<double>& x,
                         const std::vector<double>& a) {
  double r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) r = std::max(r, std::fabs(bx[i] - a[i]));
  const double denom = b_norm * inf_norm(x) + inf_norm(a);
  return denom == 0.0 ? r : r / denom;
}

void load_column(const SparseMatrix& A, Index v, SparseWork& w) {
  w.clear();
  if (v < A.n_cols) {
    for (Index t = A.col_start[to_size(v)]; t < A.col_start[to_size(v) + 1]; ++t) {
      if (A.value[to_size(t)] != 0.0) w.set(A.row_index[to_size(t)], A.value[to_size(t)]);
    }
  } else {
    w.set(v - A.n_cols, -1.0);
  }
}

bool parse_double(const std::string& s, double* out) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

}  // namespace

int run_factor_bench(const std::vector<std::string>& args) {
  std::string path;
  FactorParams params;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--threshold") {
      double u = 0.0;
      if (i + 1 >= args.size() || !parse_double(args[i + 1], &u) || !(u >= 0.0 && u <= 1.0)) {
        std::cerr << "error: --threshold needs a number in [0, 1]\n";
        return kExitUsage;
      }
      params.pivot_threshold = u;
      ++i;
    } else if (a == "--max-updates") {
      double k = 0.0;
      if (i + 1 >= args.size() || !parse_double(args[i + 1], &k) || !(k >= 0.0 && k <= 1e6) || k != std::floor(k)) {
        std::cerr << "error: --max-updates needs a non-negative integer\n";
        return kExitUsage;
      }
      params.max_updates = static_cast<Index>(k);
      ++i;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "error: unknown option '" << a << "'\n";
      return kExitUsage;
    } else if (path.empty()) {
      path = a;
    } else {
      std::cerr << "error: 'factor-bench' takes exactly one file argument\n";
      return kExitUsage;
    }
  }
  if (path.empty()) {
    std::cerr << "error: 'factor-bench' needs a file argument\n";
    return kExitUsage;
  }

  const MpsReadResult read = read_mps_file(path);
  if (!read.ok) {
    std::cerr << "error: " << read.error << "\n";
    return kExitUsage;
  }
  const LpModel& model = read.model;
  const SparseMatrix& A = model.A;
  const Index m = A.n_rows;
  const Index n = A.n_cols;

  std::cout << "Model:            " << (model.name.empty() ? "(none)" : model.name) << "  (" << m << " rows, " << n << " columns)\n";
  std::cout << "Note:             developer diagnostic, not a benchmark; the times and counters below make no claim about solver speed.\n";
  if (m == 0) {
    std::cout << "The model has no rows: nothing to factorize.\n";
    return kExitOk;
  }

  // Structural crash: columns in ascending nonzero-count order (ties by index),
  // empty columns skipped, then logical columns of the first rows if short.
  std::vector<Index> order;
  order.reserve(to_size(n));
  for (Index j = 0; j < n; ++j) {
    if (A.col_start[to_size(j) + 1] > A.col_start[to_size(j)]) order.push_back(j);
  }
  std::stable_sort(order.begin(), order.end(), [&A](Index a, Index b) {
    return A.col_start[to_size(a) + 1] - A.col_start[to_size(a)] < A.col_start[to_size(b) + 1] - A.col_start[to_size(b)];
  });
  std::vector<Index> basis;
  basis.reserve(to_size(m));
  for (Index j : order) {
    if (static_cast<Index>(basis.size()) == m) break;
    basis.push_back(j);
  }
  const Index structural = static_cast<Index>(basis.size());
  for (Index i = 0; static_cast<Index>(basis.size()) < m; ++i) basis.push_back(n + i);
  std::cout << "Crash basis:      " << structural << " structural columns, " << (m - structural) << " logical columns before repair\n";

  BasisFactor f(params);
  const auto t0 = std::chrono::steady_clock::now();
  FactorStatus st = f.factorize(A, basis);
  Index repaired = 0;
  for (int attempt = 0; attempt < 10 && st == FactorStatus::RankDeficient; ++attempt) {
    const std::vector<BasisSubstitution> changes = f.repair(A, basis);
    repaired += static_cast<Index>(changes.size());
    st = f.status();
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (st != FactorStatus::Ok) {
    std::cerr << "error: the basis is still rank deficient after repair attempts\n";
    return kExitUsage;
  }
  const double factor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const FactorStats s = f.stats();
  std::cout << "Pivot threshold:  u = " << params.pivot_threshold << ", max updates " << params.max_updates << "\n";
  std::cout << "m:                " << s.m << "\n";
  std::cout << "nnz(B):           " << s.nnz_basis << "\n";
  std::cout << "rank repaired:    " << repaired << " column(s) replaced by logicals\n";
  std::cout << "nnz(L):           " << s.nnz_l << "\n";
  std::cout << "nnz(U):           " << s.nnz_u << " (including the diagonal)\n";
  std::printf("fill ratio:       %.3f  ((nnz(L)+nnz(U)) / nnz(B))\n", s.fill_ratio);
  std::cout << "pivots:           " << s.singleton_pivots << " singleton (" << s.column_singleton_pivots << " column, " << s.row_singleton_pivots
            << " row), " << s.markowitz_pivots << " Markowitz\n";
  std::printf("factor time:      %.3f ms (including repair)\n", factor_ms);

  // 50 random solves with sparse right-hand sides.
  Rng rng(20260101);
  BasisOps ops(A, basis);
  SparseWork w(m);
  std::vector<double> rhs(to_size(m)), x(to_size(m)), bx;
  double worst_ftran = 0.0, worst_btran = 0.0;
  double sum_result_nnz = 0.0;
  double largest_x = 0.0;
  f.reset_counters();
  for (int k = 0; k < 50; ++k) {
    std::fill(rhs.begin(), rhs.end(), 0.0);
    const Index nz = std::min<Index>(m, 1 + static_cast<Index>(k % 5));
    for (Index q = 0; q < nz; ++q) rhs[to_size(rng.below(m))] = (rng.unit() < 0.5 ? -1.0 : 1.0) * (0.25 + rng.unit());
    w.load_dense(rhs);
    const bool forward = k % 2 == 0;
    if (forward) f.ftran(w); else f.btran(w);
    sum_result_nnz += static_cast<double>(w.count());
    w.to_dense(x);
    largest_x = std::max(largest_x, inf_norm(x));
    if (forward) {
      ops.multiply(x, bx);
      worst_ftran = std::max(worst_ftran, relative_residual(bx, ops.norm_inf(), x, rhs));
    } else {
      ops.multiply_transpose(x, bx);
      worst_btran = std::max(worst_btran, relative_residual(bx, ops.norm_one(), x, rhs));
    }
  }
  const FactorStats s2 = f.stats();
  std::cout << "\n50 random solves (25 ftran, 25 btran; 1 to 5 nonzeros in the right-hand side):\n";
  std::cout << "  triangular stages on the hypersparse path: " << s2.hyper_solves << "\n";
  std::cout << "  triangular stages on the dense path:       " << s2.dense_solves << "\n";
  std::printf("  average nonzeros in a result:              %.1f of %d\n", sum_result_nnz / 50.0, static_cast<int>(m));
  std::printf("  worst relative residual, ftran:            %.2e\n", worst_ftran);
  std::printf("  worst relative residual, btran:            %.2e\n", worst_btran);
  std::printf("  largest entry of any result:               %.2e (a huge value means the basis is badly conditioned:\n"
              "                                             the residual stays small but the digits of the solution do not)\n",
              largest_x);

  // A short run of Forrest-Tomlin updates with random entering columns.
  const Index total = n + m;
  const int attempts = static_cast<int>(std::min<Index>(params.max_updates, 200));
  int accepted = 0, refactors = 0, skipped = 0, repairs_in_run = 0;
  std::vector<char> in_basis(to_size(total), 0);
  for (const Index v : basis) in_basis[to_size(v)] = 1;
  bool ok = true;
  for (int k = 0; k < attempts && ok; ++k) {
    Index q = rng.below(total);
    for (int tries = 0; tries < 20 && in_basis[to_size(q)]; ++tries) q = rng.below(total);
    if (in_basis[to_size(q)]) { ++skipped; continue; }
    load_column(A, q, w);
    f.ftran(w, true);
    const double xmax = w.norm_inf();
    std::vector<Index> cand;
    for (const Index p : w.indices()) {
      if (std::fabs(w[p]) >= 0.1 * xmax && std::fabs(w[p]) > 1e-9) cand.push_back(p);
    }
    if (cand.empty()) { ++skipped; continue; }
    std::sort(cand.begin(), cand.end());
    const Index p = cand[to_size(rng.below(static_cast<Index>(cand.size())))];
    const FactorStatus us = f.update(p);
    in_basis[to_size(basis[to_size(p)])] = 0;
    in_basis[to_size(q)] = 1;
    basis[to_size(p)] = q;
    if (us == FactorStatus::Ok) {
      ++accepted;
    } else {
      ++refactors;
      FactorStatus fs = f.factorize(A, basis);
      for (int attempt = 0; attempt < 10 && fs == FactorStatus::RankDeficient; ++attempt) {
        // A numerically singular basis: substitute logicals as the crash did.
        repairs_in_run += static_cast<int>(f.repair(A, basis).size());
        fs = f.status();
        std::fill(in_basis.begin(), in_basis.end(), static_cast<char>(0));
        for (const Index v : basis) in_basis[to_size(v)] = 1;
      }
      ok = fs == FactorStatus::Ok;
    }
  }
  if (!ok) {
    std::cerr << "error: the basis could not be repaired during the update run\n";
    return kExitUsage;
  }
  ops.set_basis(basis);
  double worst_after = 0.0;
  for (int k = 0; k < 10; ++k) {
    std::fill(rhs.begin(), rhs.end(), 0.0);
    for (Index q = 0; q < std::min<Index>(m, 3); ++q) rhs[to_size(rng.below(m))] = 0.25 + rng.unit();
    w.load_dense(rhs);
    f.ftran(w);
    w.to_dense(x);
    ops.multiply(x, bx);
    worst_after = std::max(worst_after, relative_residual(bx, ops.norm_inf(), x, rhs));
  }
  std::cout << "\nForrest-Tomlin run (up to " << attempts << " random column replacements, pivot element >= 0.1 of the largest):\n";
  std::cout << "  updates accepted: " << accepted << ", NeedRefactor (then refactorized): " << refactors << ", skipped: " << skipped
            << ", columns repaired by a refactorization: " << repairs_in_run << "\n";
  std::printf("  worst relative ftran residual after the run (10 solves): %.2e\n", worst_after);
  return kExitOk;
}

}  // namespace shodhan::cli
