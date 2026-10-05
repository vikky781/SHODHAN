// Symbolic analysis and numeric LDL^T (docs/LDL.md).
//
// Symbolic: the elimination tree is built with Liu's algorithm (path compression over the ancestors), postordered, and
// the column counts of L are obtained by walking, for every row k, from each nonzero A(i,k), i < k, up the tree until a
// node already visited for row k (the row subtree of Liu, O(nnz(L)) time).
//
// Numeric: up-looking. Row k of L is the solution of the sparse triangular system L(0:k-1,0:k-1) y = A(0:k-1,k) whose
// nonzero pattern is the row subtree of k; then l_ki = y_i / d_i and d_k = a_kk - sum_i l_ki y_i.

#include <algorithm>
#include <cmath>
#include <numeric>

#include "shodhan/constants.hpp"
#include "shodhan/sparse_ldl.hpp"

namespace shodhan {

namespace {

std::size_t u(Index i) { return static_cast<std::size_t>(i); }

// Upper-triangle CSC (column k lists rows < k) of P K P^T plus the diagonal, from the lower triangle of K.
struct Permuted {
  std::vector<Index> up, ui;
  std::vector<double> ux, diag;
  std::vector<Index> map;
};

Permuted permute_upper(const SparseMatrix& lower, const std::vector<Index>& iperm) {
  const Index n = lower.n_cols;
  Permuted r;
  r.up.assign(u(n) + 1, 0);
  r.diag.assign(u(n), 0.0);
  r.map.assign(lower.nnz(), 0);
  for (Index j = 0; j < n; ++j) {
    for (Index p = lower.col_start[u(j)]; p < lower.col_start[u(j) + 1]; ++p) {
      const Index a = iperm[u(lower.row_index[u(p)])], b = iperm[u(j)];
      if (a != b) ++r.up[u(std::max(a, b)) + 1];
    }
  }
  for (Index k = 0; k < n; ++k) r.up[u(k) + 1] += r.up[u(k)];
  r.ui.assign(u(r.up[u(n)]), 0);
  r.ux.assign(u(r.up[u(n)]), 0.0);
  std::vector<Index> fill(r.up.begin(), r.up.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = lower.col_start[u(j)]; p < lower.col_start[u(j) + 1]; ++p) {
      const Index a = iperm[u(lower.row_index[u(p)])], b = iperm[u(j)];
      if (a == b) {
        r.map[u(p)] = -1 - a;
        r.diag[u(a)] = lower.value[u(p)];
      } else {
        const Index col = std::max(a, b), row = std::min(a, b);
        const Index pos = fill[u(col)]++;
        r.ui[u(pos)] = row;
        r.ux[u(pos)] = lower.value[u(p)];
        r.map[u(p)] = pos;
      }
    }
  }
  return r;
}

// Liu's elimination tree of the matrix with upper-triangle pattern (up, ui).
std::vector<Index> etree(Index n, const std::vector<Index>& up, const std::vector<Index>& ui) {
  std::vector<Index> parent(u(n), -1), ancestor(u(n), -1);
  for (Index k = 0; k < n; ++k) {
    for (Index p = up[u(k)]; p < up[u(k) + 1]; ++p) {
      Index r = ui[u(p)];
      while (r != -1 && r < k) {
        const Index next = ancestor[u(r)];
        ancestor[u(r)] = k;
        if (next == -1) {
          parent[u(r)] = k;
          break;
        }
        r = next;
      }
    }
  }
  return parent;
}

std::vector<Index> tree_postorder(Index n, const std::vector<Index>& parent) {
  std::vector<Index> first_child(u(n), -1), next_sibling(u(n), -1);
  for (Index v = n - 1; v >= 0; --v) {
    const Index pa = parent[u(v)];
    if (pa != -1) {
      next_sibling[u(v)] = first_child[u(pa)];
      first_child[u(pa)] = v;
    }
  }
  std::vector<Index> post;
  post.reserve(u(n));
  std::vector<Index> stack;
  for (Index root = 0; root < n; ++root) {
    if (parent[u(root)] != -1) continue;
    stack.push_back(root);
    while (!stack.empty()) {
      const Index v = stack.back();
      const Index c = first_child[u(v)];
      if (c != -1) {
        first_child[u(v)] = next_sibling[u(c)];
        stack.push_back(c);
      } else {
        post.push_back(v);
        stack.pop_back();
      }
    }
  }
  return post;
}

}  // namespace

SymbolicLdl analyze_ldl(const SparseMatrix& lower, const std::vector<Index>& order) {
  const Index n = lower.n_cols;
  SymbolicLdl s;
  s.n = n;
  std::vector<Index> iperm(u(n));
  for (Index k = 0; k < n; ++k) iperm[u(order[u(k)])] = k;
  Permuted pm = permute_upper(lower, iperm);
  std::vector<Index> parent = etree(n, pm.up, pm.ui);
  // Postorder the tree and compose with the ordering (the fill is unchanged; locality improves).
  const std::vector<Index> post = tree_postorder(n, parent);
  s.perm.resize(u(n));
  for (Index k = 0; k < n; ++k) s.perm[u(k)] = order[u(post[u(k)])];
  s.iperm.resize(u(n));
  for (Index k = 0; k < n; ++k) s.iperm[u(s.perm[u(k)])] = k;
  pm = permute_upper(lower, s.iperm);
  s.parent = etree(n, pm.up, pm.ui);
  s.col_count.assign(u(n), 0);
  std::vector<Index> mark(u(n), -1);
  for (Index k = 0; k < n; ++k) {
    mark[u(k)] = k;
    for (Index p = pm.up[u(k)]; p < pm.up[u(k) + 1]; ++p) {
      Index r = pm.ui[u(p)];
      while (mark[u(r)] != k) {
        ++s.col_count[u(r)];
        mark[u(r)] = k;
        r = s.parent[u(r)];
      }
    }
  }
  for (Index k = 0; k < n; ++k) {
    s.nnz_l += s.col_count[u(k)];
    s.flops += static_cast<double>(s.col_count[u(k)]) * static_cast<double>(s.col_count[u(k)]);
  }
  return s;
}

std::vector<std::vector<Index>> ldl_pattern(const SparseMatrix& lower, const SymbolicLdl& sym) {
  const Index n = lower.n_cols;
  const Permuted pm = permute_upper(lower, sym.iperm);
  std::vector<std::vector<Index>> pat(u(n));
  std::vector<Index> mark(u(n), -1);
  for (Index k = 0; k < n; ++k) {
    mark[u(k)] = k;
    for (Index p = pm.up[u(k)]; p < pm.up[u(k) + 1]; ++p) {
      Index r = pm.ui[u(p)];
      while (mark[u(r)] != k) {
        pat[u(r)].push_back(k);
        mark[u(r)] = k;
        r = sym.parent[u(r)];
      }
    }
  }
  return pat;  // rows are appended in increasing k, so every column is sorted
}

// ---------------------------------------------------------------------------------------------------------------
bool SparseLdl::analyze(const SparseMatrix& lower, const std::vector<signed char>& sign, Ordering ordering) {
  analyzed_ = factored_ = false;
  if (lower.n_rows != lower.n_cols || !lower.validate().empty()) return false;
  const Index n = lower.n_cols;
  if (sign.size() != u(n)) return false;
  for (Index j = 0; j < n; ++j) {
    for (Index p = lower.col_start[u(j)]; p < lower.col_start[u(j) + 1]; ++p) {
      if (lower.row_index[u(p)] < j) return false;  // not a lower triangle
    }
  }
  n_ = n;
  nnz_ = lower.nnz();
  const std::vector<Index> order = ordering == Ordering::Amd ? amd_order(lower) : natural_order(n);
  sym_ = analyze_ldl(lower, order);
  psign_.resize(u(n));
  for (Index k = 0; k < n; ++k) psign_[u(k)] = sign[u(sym_.perm[u(k)])];
  Permuted pm = permute_upper(lower, sym_.iperm);
  up_ = std::move(pm.up);
  ui_ = std::move(pm.ui);
  ux_ = std::move(pm.ux);
  diag_ = std::move(pm.diag);
  map_ = std::move(pm.map);
  k_ = lower;
  lp_.assign(u(n) + 1, 0);
  for (Index k = 0; k < n; ++k) lp_[u(k) + 1] = lp_[u(k)] + sym_.col_count[u(k)];
  li_.assign(u(lp_[u(n)]), 0);
  lx_.assign(u(lp_[u(n)]), 0.0);
  lnz_.assign(u(n), 0);
  d_.assign(u(n), 0.0);
  y_.assign(u(n), 0.0);
  w_.assign(u(n), 0.0);
  res_.assign(u(n), 0.0);
  dx_.assign(u(n), 0.0);
  xs_.assign(u(n), 0.0);
  flag_.assign(u(n), -1);
  stack_.assign(u(n), 0);
  pattern_.assign(u(n), 0);
  stats_ = LdlStats();
  stats_.n = n;
  stats_.nnz_l = sym_.nnz_l;
  stats_.flops = sym_.flops;
  analyzed_ = true;
  return true;
}

bool SparseLdl::factorize(const SparseMatrix& lower) {
  factored_ = false;
  if (!analyzed_ || lower.n_cols != n_ || lower.nnz() != nnz_) return false;
  const Index n = n_;
  // Load the new values into the permuted copy and keep them for the residuals.
  k_.value = lower.value;
  double scale = 0.0;
  for (std::size_t p = 0; p < nnz_; ++p) {
    const Index m = map_[p];
    if (m >= 0) {
      ux_[u(m)] = lower.value[p];
    } else {
      diag_[u(-1 - m)] = lower.value[p];
      scale = std::max(scale, std::fabs(lower.value[p]));
    }
  }
  if (scale == 0.0) scale = 1.0;
  stats_.dynamic_regularizations = 0;
  stats_.min_pivot = kInf;
  stats_.min_pivot_column = -1;
  std::fill(lnz_.begin(), lnz_.end(), Index{0});
  std::fill(flag_.begin(), flag_.end(), Index{-1});
  std::fill(y_.begin(), y_.end(), 0.0);
  for (Index k = 0; k < n; ++k) {
    // Row subtree of k in topological order, and y = column k of the upper triangle scattered.
    Index top = n;
    flag_[u(k)] = k;
    y_[u(k)] = diag_[u(k)] + (psign_[u(k)] > 0 ? params_.rho : -params_.delta);
    for (Index p = up_[u(k)]; p < up_[u(k) + 1]; ++p) {
      Index i = ui_[u(p)];
      y_[u(i)] += ux_[u(p)];
      Index len = 0;
      while (flag_[u(i)] != k) {
        pattern_[u(len++)] = i;
        flag_[u(i)] = k;
        i = sym_.parent[u(i)];
      }
      while (len > 0) stack_[u(--top)] = pattern_[u(--len)];
    }
    double dk = y_[u(k)];
    y_[u(k)] = 0.0;
    for (; top < n; ++top) {
      const Index i = stack_[u(top)];
      const double yi = y_[u(i)];
      y_[u(i)] = 0.0;
      const Index end = lp_[u(i)] + lnz_[u(i)];
      for (Index p = lp_[u(i)]; p < end; ++p) y_[u(li_[u(p)])] -= lx_[u(p)] * yi;
      const double lki = yi / d_[u(i)];
      dk -= lki * yi;
      li_[u(end)] = k;
      lx_[u(end)] = lki;
      ++lnz_[u(i)];
    }
    // Dynamic, sign-aware regularization of a pivot that is too small or of the wrong sign.
    const double s = psign_[u(k)] > 0 ? 1.0 : -1.0;
    if (!std::isfinite(dk)) return false;
    if (s * dk < stats_.min_pivot) {
      stats_.min_pivot = s * dk;
      stats_.min_pivot_column = sym_.perm[u(k)];
    }
    if (s * dk < params_.pivot_tol * scale) {
      dk = s * params_.dynamic_delta * scale;
      ++stats_.dynamic_regularizations;
    }
    d_[u(k)] = dk;
  }
  ++stats_.factorizations;
  factored_ = true;
  return true;
}

void SparseLdl::solve_factors(std::span<double> y) const {
  const Index n = n_;
  for (Index i = 0; i < n; ++i) {
    const double yi = y[u(i)];
    if (yi == 0.0) continue;
    const Index end = lp_[u(i)] + lnz_[u(i)];
    for (Index p = lp_[u(i)]; p < end; ++p) y[u(li_[u(p)])] -= lx_[u(p)] * yi;
  }
  for (Index i = 0; i < n; ++i) y[u(i)] /= d_[u(i)];
  for (Index i = n - 1; i >= 0; --i) {
    double s = y[u(i)];
    const Index end = lp_[u(i)] + lnz_[u(i)];
    for (Index p = lp_[u(i)]; p < end; ++p) s -= lx_[u(p)] * y[u(li_[u(p)])];
    y[u(i)] = s;
  }
}

// r = b - K x with K symmetric from its lower triangle; returns ||r||_inf / (||b||_inf + ||K||_inf ||x||_inf).
double SparseLdl::residual(std::span<const double> b, std::span<const double> x, std::vector<double>& r) const {
  const Index n = n_;
  for (Index i = 0; i < n; ++i) r[u(i)] = b[u(i)];
  std::vector<double>& rowabs = w_;
  std::fill(rowabs.begin(), rowabs.end(), 0.0);
  for (Index j = 0; j < n; ++j) {
    for (Index p = k_.col_start[u(j)]; p < k_.col_start[u(j) + 1]; ++p) {
      const Index i = k_.row_index[u(p)];
      const double v = k_.value[u(p)];
      r[u(i)] -= v * x[u(j)];
      rowabs[u(i)] += std::fabs(v);
      if (i != j) {
        r[u(j)] -= v * x[u(i)];
        rowabs[u(j)] += std::fabs(v);
      }
    }
  }
  double rn = 0.0, bn = 0.0, kn = 0.0, xn = 0.0;
  for (Index i = 0; i < n; ++i) {
    rn = std::max(rn, std::fabs(r[u(i)]));
    bn = std::max(bn, std::fabs(b[u(i)]));
    kn = std::max(kn, rowabs[u(i)]);
    xn = std::max(xn, std::fabs(x[u(i)]));
  }
  const double den = bn + kn * xn;
  return den > 0.0 ? rn / den : rn;
}

void SparseLdl::solve(std::span<const double> b, std::span<double> x, int max_refinement, double tol) {
  const Index n = n_;
  stats_.refinement_steps = 0;
  auto base_solve = [&](std::span<const double> rhs, std::span<double> out) {
    for (Index k = 0; k < n; ++k) xs_[u(k)] = rhs[u(sym_.perm[u(k)])];
    solve_factors(xs_);
    for (Index k = 0; k < n; ++k) out[u(sym_.perm[u(k)])] = xs_[u(k)];
  };
  base_solve(b, x);
  double rel = residual(b, x, res_);
  for (int step = 0; step < max_refinement && rel > tol; ++step) {
    base_solve(res_, dx_);
    for (Index i = 0; i < n; ++i) x[u(i)] += dx_[u(i)];
    const double rel2 = residual(b, x, res_);
    if (!(rel2 < rel)) {  // the step did not help: take it back
      for (Index i = 0; i < n; ++i) x[u(i)] -= dx_[u(i)];
      residual(b, x, res_);
      break;
    }
    ++stats_.refinement_steps;
    rel = rel2;
  }
  stats_.residual = rel;
}

}  // namespace shodhan
