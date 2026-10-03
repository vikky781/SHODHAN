#include "support/lu_testing.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "support/random_lp.hpp"

namespace shodhan::testing {

namespace {

SparseMatrix build(Index rows, Index cols, std::vector<Triplet> t) {
  SparseMatrix A;
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, std::move(t), &A, &err)) {
    // The generators never produce duplicates; fail loudly if that changes.
    throw std::logic_error("lu_testing: " + err);
  }
  return A;
}

std::vector<Index> random_permutation(Index n, Rng& rng) {
  std::vector<Index> p(to_size(n));
  std::iota(p.begin(), p.end(), 0);
  rng.shuffle(p);
  return p;
}

double signed_value(Rng& rng, double lo, double hi) {
  const double v = rng.uniform(lo, hi);
  return rng.chance(0.5) ? v : -v;
}

}  // namespace

const char* basis_family_name(int family) {
  static const char* const names[kNumBasisFamilies] = {
      "random-sparse", "lower-triangular", "upper-triangular", "block-diagonal",
      "arrowhead",     "dense-row",        "dense-column",     "banded",
      "lp-basis",      "signed-permutation", "slack-heavy"};
  return family >= 0 && family < kNumBasisFamilies ? names[family] : "unknown";
}

TestBasis make_family_basis(int family, Index m, std::uint64_t seed) {
  Rng rng(seed * 7919ULL + static_cast<std::uint64_t>(family) * 104729ULL + 17ULL);
  TestBasis out;
  out.family = basis_family_name(family);
  const Index extra = rng.range(0, std::max<Index>(1, m / 2));  // decoy columns outside the basis
  const double off_scale = rng.chance(0.5) ? 1.0 : 0.2;
  const double density = rng.uniform(0.03, 0.5);

  if (family == 8) {
    // Basis taken from a random LP: m rows, columns of [A | -I].
    RandomLpOptions o;
    o.rows = m;
    o.cols = m + extra;
    o.density = std::min(0.8, std::max(0.05, 3.0 / static_cast<double>(std::max<Index>(m, 1)) + rng.uniform(0.0, 0.2)));
    o.wide_coefficients = rng.chance(0.2);
    const RandomLp lp = make_random_lp(seed + 1000003ULL, o);
    out.A = lp.model.A;
    std::vector<Index> all(to_size(out.A.n_cols + m));
    std::iota(all.begin(), all.end(), 0);
    rng.shuffle(all);
    out.basis.assign(all.begin(), all.begin() + m);
    return out;
  }

  std::vector<Triplet> t;
  auto add = [&t](Index i, Index j, double v) { t.push_back({i, j, v}); };
  std::vector<Index> logical_rows;  // for the slack-heavy family
  switch (family) {
    case 0:  // random sparse
      for (Index j = 0; j < m; ++j)
        for (Index i = 0; i < m; ++i)
          if (rng.chance(density)) add(i, j, signed_value(rng, 0.5, 2.0));
      break;
    case 1:  // lower triangular
    case 2:  // upper triangular
      for (Index j = 0; j < m; ++j) {
        add(j, j, signed_value(rng, 0.5, 2.0));
        for (Index i = j + 1; i < m; ++i)
          if (rng.chance(density * 0.5)) {
            const double v = signed_value(rng, 0.05, 1.0) * off_scale;
            if (family == 1) add(i, j, v); else add(j, i, v);
          }
      }
      break;
    case 3: {  // block diagonal
      Index start = 0;
      while (start < m) {
        const Index bs = std::min<Index>(m - start, rng.range(1, 8));
        for (Index i = 0; i < bs; ++i)
          for (Index j = 0; j < bs; ++j)
            if (i == j || rng.chance(0.7)) add(start + i, start + j, signed_value(rng, 0.5, 2.0));
        start += bs;
      }
      break;
    }
    case 4:  // arrowhead: dense first row and column plus the diagonal
      for (Index j = 0; j < m; ++j) {
        add(j, j, signed_value(rng, 1.0, 3.0));
        if (j > 0) {
          add(0, j, signed_value(rng, 0.1, 1.0) * off_scale);
          add(j, 0, signed_value(rng, 0.1, 1.0) * off_scale);
        }
      }
      break;
    case 5:  // dense row on top of a sparse, diagonally heavy matrix
    case 6:  // dense column
      for (Index j = 0; j < m; ++j) {
        add(j, j, signed_value(rng, 1.0, 3.0));
        for (Index i = 0; i < m; ++i)
          if (i != j && rng.chance(density * 0.15)) add(i, j, signed_value(rng, 0.1, 1.0) * off_scale);
      }
      {
        const Index k = rng.range(0, m - 1);
        // Replace row k (or column k) by a dense one.
        std::vector<Triplet> kept;
        for (const Triplet& e : t) {
          const bool hit = family == 5 ? e.row == k : e.col == k;
          if (!hit) kept.push_back(e);
        }
        t.swap(kept);
        for (Index q = 0; q < m; ++q) {
          if (family == 5) add(k, q, signed_value(rng, 0.2, 2.0));
          else add(q, k, signed_value(rng, 0.2, 2.0));
        }
      }
      break;
    case 7: {  // banded
      const Index bw = rng.range(1, 5);
      for (Index j = 0; j < m; ++j)
        for (Index i = std::max<Index>(0, j - bw); i <= std::min<Index>(m - 1, j + bw); ++i)
          if (i == j) add(i, j, signed_value(rng, 2.0, 4.0));
          else if (rng.chance(0.8)) add(i, j, signed_value(rng, 0.1, 1.0));
      break;
    }
    case 9:  // signed permutation
      for (Index j = 0; j < m; ++j) add(j, j, rng.chance(0.5) ? 1.0 : -1.0);
      break;
    case 10:  // slack-heavy: sparse columns plus logical columns
      for (Index j = 0; j < m; ++j) {
        if (rng.chance(0.5)) continue;  // this position will hold a logical
        add(j, j, signed_value(rng, 0.5, 2.0));
        for (Index i = 0; i < m; ++i)
          if (i != j && rng.chance(density * 0.3)) add(i, j, signed_value(rng, 0.1, 1.0));
      }
      break;
    default:
      break;
  }

  // Random row and column permutations; decoy columns are appended to A.
  const std::vector<Index> rp = random_permutation(m, rng);
  const std::vector<Index> cp = random_permutation(m, rng);
  for (Triplet& e : t) {
    e.row = rp[to_size(e.row)];
    e.col = cp[to_size(e.col)];
  }
  for (Index j = 0; j < extra; ++j) {
    std::vector<bool> used(to_size(m), false);
    const Index cnt = rng.range(1, std::min<Index>(m, 4));
    for (Index q = 0; q < cnt; ++q) {
      const Index i = rng.range(0, m - 1);
      if (used[to_size(i)]) continue;
      used[to_size(i)] = true;
      add(i, m + j, signed_value(rng, 0.5, 2.0));
    }
  }
  out.A = build(m, m + extra, std::move(t));
  out.basis.resize(to_size(m));
  std::iota(out.basis.begin(), out.basis.end(), 0);
  if (family == 10) {
    // Positions whose column has no entries hold a logical column instead.
    std::vector<bool> has_entry(to_size(m), false);
    for (Index j = 0; j < m; ++j) has_entry[to_size(j)] = out.A.col_start[to_size(j) + 1] > out.A.col_start[to_size(j)];
    // Distinct rows for the logicals: assign by shuffling the used-row complement.
    std::vector<Index> free_rows(to_size(m));
    std::iota(free_rows.begin(), free_rows.end(), 0);
    rng.shuffle(free_rows);
    Index next = 0;
    for (Index j = 0; j < m; ++j) {
      if (!has_entry[to_size(j)]) out.basis[to_size(j)] = out.A.n_cols + free_rows[to_size(next++)];
    }
  }
  return out;
}

TestBasis make_seeded_basis(std::uint64_t seed) {
  Rng rng(seed ^ 0x5bd1e995ULL);
  const int family = static_cast<int>(seed % static_cast<std::uint64_t>(kNumBasisFamilies));
  Index m = 1;
  const double u = rng.unit();
  if (u < 0.4) m = rng.range(1, 10);
  else if (u < 0.8) m = rng.range(10, 60);
  else m = rng.range(60, 200);
  return make_family_basis(family, m, seed);
}

DenseMatrix basis_to_dense(const SparseMatrix& A, const std::vector<Index>& basis) {
  const Index m = static_cast<Index>(basis.size());
  DenseMatrix B(m);
  for (Index p = 0; p < m; ++p) {
    const Index v = basis[to_size(p)];
    if (v < A.n_cols) {
      for (Index t = A.col_start[to_size(v)]; t < A.col_start[to_size(v) + 1]; ++t) {
        B.at(A.row_index[to_size(t)], p) = A.value[to_size(t)];
      }
    } else {
      B.at(v - A.n_cols, p) = -1.0;
    }
  }
  return B;
}

std::vector<double> basis_multiply(const SparseMatrix& A, const std::vector<Index>& basis,
                                   const std::vector<double>& x) {
  std::vector<double> y(basis.size(), 0.0);
  for (std::size_t p = 0; p < basis.size(); ++p) {
    const Index v = basis[p];
    if (v < A.n_cols) {
      for (Index t = A.col_start[to_size(v)]; t < A.col_start[to_size(v) + 1]; ++t) {
        y[to_size(A.row_index[to_size(t)])] += A.value[to_size(t)] * x[p];
      }
    } else {
      y[to_size(v - A.n_cols)] -= x[p];
    }
  }
  return y;
}

std::vector<double> basis_multiply_transpose(const SparseMatrix& A, const std::vector<Index>& basis,
                                             const std::vector<double>& x) {
  std::vector<double> y(basis.size(), 0.0);
  for (std::size_t p = 0; p < basis.size(); ++p) {
    const Index v = basis[p];
    double s = 0.0;
    if (v < A.n_cols) {
      for (Index t = A.col_start[to_size(v)]; t < A.col_start[to_size(v) + 1]; ++t) {
        s += A.value[to_size(t)] * x[to_size(A.row_index[to_size(t)])];
      }
    } else {
      s = -x[to_size(v - A.n_cols)];
    }
    y[p] = s;
  }
  return y;
}

double norm_inf(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::fabs(x));
  return m;
}

double relative_residual(const std::vector<double>& b_times_x, double b_norm,
                         const std::vector<double>& x, const std::vector<double>& a) {
  double r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) r = std::max(r, std::fabs(b_times_x[i] - a[i]));
  const double denom = b_norm * norm_inf(x) + norm_inf(a);
  return denom == 0.0 ? r : r / denom;
}

std::vector<double> random_dense_vector(Index n, Rng& rng) {
  std::vector<double> v(to_size(n));
  for (double& x : v) x = rng.uniform(-1.0, 1.0);
  return v;
}

std::vector<double> random_sparse_vector(Index n, Index nnz, Rng& rng) {
  std::vector<double> v(to_size(n), 0.0);
  for (Index k = 0; k < nnz; ++k) v[to_size(rng.range(0, n - 1))] = signed_value(rng, 0.2, 1.0);
  return v;
}

SolveCheckResult check_solves(BasisFactor& f, const SparseMatrix& A, const std::vector<Index>& basis,
                              Rng& rng, const SolveCheckOptions& opt) {
  constexpr double kEps = 2.220446049250313e-16;
  SolveCheckResult res;
  const Index m = static_cast<Index>(basis.size());
  const DenseMatrix B = basis_to_dense(A, basis);
  const DenseLu lu(B);
  const double b_inf = B.norm_inf();
  const double b_one = B.norm_one();
  double inv_inf = 0.0;
  double inv_one = 0.0;
  if (!lu.singular()) {
    std::vector<double> col_sum(to_size(m), 0.0);
    std::vector<double> e(to_size(m), 0.0);
    for (Index j = 0; j < m; ++j) {
      e[to_size(j)] = 1.0;
      const std::vector<double> c = lu.solve(e);
      e[to_size(j)] = 0.0;
      double s = 0.0;
      for (const double v : c) s += std::fabs(v);
      inv_one = std::max(inv_one, s);
      if (col_sum.size() == c.size()) {
        for (std::size_t i = 0; i < c.size(); ++i) col_sum[i] += std::fabs(c[i]);
      }
    }
    for (const double v : col_sum) inv_inf = std::max(inv_inf, v);
  }
  res.kappa = lu.singular() ? 0.0 : std::max(b_inf * inv_inf, b_one * inv_one);

  auto fail = [&res](const std::string& what) {
    if (res.failure.empty()) res.failure = what;
  };

  std::vector<std::vector<double>> rhs_list;
  for (int k = 0; k < opt.dense_rhs; ++k) rhs_list.push_back(random_dense_vector(m, rng));
  for (int k = 0; k < opt.sparse_rhs; ++k) rhs_list.push_back(random_sparse_vector(m, 1 + k, rng));
  {
    std::vector<double> unit(to_size(m), 0.0);
    unit[to_size(rng.range(0, m - 1))] = 1.0;
    rhs_list.push_back(unit);
  }

  SparseWork w(m);
  for (std::size_t k = 0; k < rhs_list.size(); ++k) {
    for (int dir = 0; dir < 2; ++dir) {  // 0: ftran, 1: btran
      const std::vector<double>& a = rhs_list[k];
      w.load_dense(a);
      if (dir == 0) f.ftran(w); else f.btran(w);
      ++res.solves;
      if (!work_invariant_holds(w)) fail("SparseWork invariant broken after the solve");
      std::vector<double> x(to_size(m));
      w.to_dense(x);
      const std::vector<double> bx = dir == 0 ? basis_multiply(A, basis, x) : basis_multiply_transpose(A, basis, x);
      const double r = relative_residual(bx, dir == 0 ? b_inf : b_one, x, a);
      res.worst_residual = std::max(res.worst_residual, r);
      if (!(r <= opt.residual_tol)) {
        fail(std::string(dir == 0 ? "ftran" : "btran") + " relative residual " + std::to_string(r) + " exceeds " +
             std::to_string(opt.residual_tol));
      }
      if (!lu.singular()) {
        const std::vector<double> xo = dir == 0 ? lu.solve(a) : lu.solve_transpose(a);
        double diff = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) diff = std::max(diff, std::fabs(x[i] - xo[i]));
        const double bound = opt.forward_factor * res.kappa * kEps * norm_inf(xo) + 1e-13;
        res.worst_forward_ratio = std::max(res.worst_forward_ratio, diff / bound);
        if (!(diff <= bound)) {
          fail(std::string(dir == 0 ? "ftran" : "btran") + " differs from the dense oracle by " +
               std::to_string(diff) + " (bound " + std::to_string(bound) + ", kappa " +
               std::to_string(res.kappa) + ")");
        }
      }
    }
  }
  return res;
}

}  // namespace shodhan::testing
