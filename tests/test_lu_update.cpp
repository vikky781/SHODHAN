// Forrest-Tomlin update tests: long random replacement sequences compared with a
// fresh factorization of the updated basis, and forced-trigger cases.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "support/lu_testing.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

constexpr int kSequences = 200;
constexpr int kStepsPerSequence = 300;
constexpr double kUpdateTol = 1e-8;  // target

struct Start {
  SparseMatrix A;
  std::vector<Index> basis;
  std::string description;
};

// Half of the sequences start from the all-logical basis (the usual simplex start)
// over a random sparse A, the others from a nonsingular structured basis.
Start make_start(std::uint64_t seed, Rng& rng) {
  const Index m = rng.range(5, 40);
  Start s;
  if (seed % 2 == 0) {
    TestBasis tb = make_family_basis(0, m, seed);
    s.A = tb.A;
    s.basis.resize(to_size(m));
    for (Index p = 0; p < m; ++p) s.basis[to_size(p)] = s.A.n_cols + p;
    rng.shuffle(s.basis);
    s.description = "all-logical start";
  } else {
    static const int families[] = {1, 2, 3, 4, 5, 6, 7, 9};
    const int fam = families[rng.range(0, 7)];
    TestBasis tb = make_family_basis(fam, m, seed);
    s.A = tb.A;
    s.basis = tb.basis;
    s.description = std::string(basis_family_name(fam)) + " start";
  }
  return s;
}

// Column of variable v as a sparse right-hand side (row indexed).
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

double relative_difference(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d / std::max(norm_inf(b), 1e-300);
}

// Compares f (updated) with a fresh factorization g of the same basis on a few
// right-hand sides; returns the largest relative difference.
double compare_with_fresh(BasisFactor& f, BasisFactor& g, Index m, Rng& rng) {
  double worst = 0.0;
  SparseWork wf(m), wg(m);
  for (int k = 0; k < 4; ++k) {
    const std::vector<double> a = k < 2 ? random_dense_vector(m, rng) : random_sparse_vector(m, 1 + k, rng);
    for (int dir = 0; dir < 2; ++dir) {
      wf.load_dense(a);
      wg.load_dense(a);
      if (dir == 0) { f.ftran(wf); g.ftran(wg); } else { f.btran(wf); g.btran(wg); }
      std::vector<double> xf(to_size(m)), xg(to_size(m));
      wf.to_dense(xf);
      wg.to_dense(xg);
      worst = std::max(worst, relative_difference(xf, xg));
    }
  }
  return worst;
}

}  // namespace

// 200 seeded sequences of 300 column replacements. The entering variable is any
// variable outside the basis (LP columns and logicals); the leaving position is
// chosen at random among those whose pivot element is at least a tenth of the
// largest in the ftran'd entering column. After every update the updated
// factorization must agree with a fresh factorization of the new basis.
TEST_CASE(lu_update_sequences_match_fresh_factorization) {
  double worst = 0.0;
  std::uint64_t worst_seed = 0;
  int worst_step = 0;
  long total_updates = 0, total_refactors = 0, total_skipped = 0;
  int failed_sequences = 0;
  int passed_sequences = 0;
  for (int seq = 0; seq < kSequences; ++seq) {
    const std::uint64_t seed = 1000 + static_cast<std::uint64_t>(seq);
    Rng rng(seed * 2654435761ULL);
    Start st = make_start(seed, rng);
    const Index m = st.A.n_rows;
    const Index total_vars = st.A.n_cols + m;
    FactorParams params;
    static const Index limits[] = {10, 30, 100};
    params.max_updates = limits[seq % 3];
    BasisFactor f(params);
    if (f.factorize(st.A, st.basis) != FactorStatus::Ok) {
      std::cerr << "FAILING SEED " << seed << ": start basis not factorizable\n";
      CHECK(false);
      ++failed_sequences;
      continue;
    }
    std::vector<char> in_basis(to_size(total_vars), 0);
    for (const Index v : st.basis) in_basis[to_size(v)] = 1;
    SparseWork w(m);
    bool seq_ok = true;
    for (int step = 0; step < kStepsPerSequence && seq_ok; ++step) {
      // Entering variable: random, not in the basis.
      Index q = rng.range(0, total_vars - 1);
      for (int tries = 0; tries < 50 && in_basis[to_size(q)]; ++tries) q = rng.range(0, total_vars - 1);
      if (in_basis[to_size(q)]) continue;
      load_column(st.A, q, w);
      f.ftran(w, true);
      const double xmax = w.norm_inf();
      std::vector<Index> cand;
      for (const Index p : w.indices()) {
        if (std::fabs(w[p]) >= 0.1 * xmax && std::fabs(w[p]) > 1e-9) cand.push_back(p);
      }
      if (cand.empty()) { ++total_skipped; continue; }
      std::sort(cand.begin(), cand.end());  // deterministic order independent of the work list
      const Index p = cand[to_size(rng.range(0, static_cast<Index>(cand.size()) - 1))];
      const FactorStatus us = f.update(p);
      in_basis[to_size(st.basis[to_size(p)])] = 0;
      in_basis[to_size(q)] = 1;
      st.basis[to_size(p)] = q;
      if (us == FactorStatus::Ok) {
        ++total_updates;
      } else {
        ++total_refactors;
        if (f.factorize(st.A, st.basis) != FactorStatus::Ok) {
          std::cerr << "FAILING SEED " << seed << " step " << step << ": new basis is rank deficient after a pivot of " << w[p] << "\n";
          CHECK(false);
          seq_ok = false;
          break;
        }
      }
      BasisFactor g;
      if (g.factorize(st.A, st.basis) != FactorStatus::Ok) {
        std::cerr << "FAILING SEED " << seed << " step " << step << ": fresh factorization of the updated basis failed\n";
        CHECK(false);
        seq_ok = false;
        break;
      }
      const double d = compare_with_fresh(f, g, m, rng);
      if (d > worst) { worst = d; worst_seed = seed; worst_step = step; }
      if (!(d <= kUpdateTol)) {
        const DenseMatrix B = basis_to_dense(st.A, st.basis);
        const DenseLu lu(B);
        std::cerr << "FAILING SEED " << seed << " (" << st.description << ", m = " << m << ") step " << step
                  << ": updated vs fresh solves differ by " << d << " (condition estimate "
                  << B.norm_inf() * lu.inverse_norm_inf() << ", updates since refactor " << f.update_count() << ")\n";
        CHECK(false);
        seq_ok = false;
      }
    }
    if (seq_ok) ++passed_sequences; else ++failed_sequences;
  }
  std::cout << "    lu update: " << passed_sequences << "/" << kSequences << " sequences of " << kStepsPerSequence
            << " steps passed; " << total_updates << " updates accepted, " << total_refactors
            << " NeedRefactor (refactored and continued), " << total_skipped << " steps skipped (no usable pivot); "
            << "worst updated-vs-fresh relative difference " << std::scientific << std::setprecision(2) << worst
            << " (seed " << worst_seed << ", step " << worst_step << ")\n"
            << std::defaultfloat;
  CHECK_EQ(passed_sequences, kSequences);
  CHECK_EQ(failed_sequences, 0);
}

namespace {

SparseMatrix small_matrix(Index rows, Index cols, const std::vector<Triplet>& t) {
  SparseMatrix A;
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, t, &A, &err)) throw std::runtime_error(err);
  return A;
}

// Arrowhead-like 6x6 basis plus extra columns to enter.
struct Fixture {
  SparseMatrix A;
  std::vector<Index> basis{0, 1, 2, 3, 4, 5};
  Fixture() {
    std::vector<Triplet> t;
    for (Index j = 0; j < 6; ++j) {
      t.push_back({j, j, 3.0 + static_cast<double>(j)});
      if (j > 0) { t.push_back({0, j, 1.0}); t.push_back({j, 0, 0.5}); }
    }
    // Extra columns 6..8.
    t.push_back({0, 6, 1.0}); t.push_back({2, 6, -2.0}); t.push_back({5, 6, 1.5});
    t.push_back({1, 7, 2.0}); t.push_back({3, 7, 1.0});
    t.push_back({4, 8, 1.0}); t.push_back({0, 8, 0.25});
    A = small_matrix(6, 9, t);
  }
};

// ftran of column v with the spike saved; returns the solution (dense, by position).
std::vector<double> ftran_column(BasisFactor& f, const SparseMatrix& A, Index v) {
  SparseWork w(A.n_rows);
  load_column(A, v, w);
  f.ftran(w, true);
  std::vector<double> x(to_size(A.n_rows));
  w.to_dense(x);
  return x;
}

void expect_matches_fresh(BasisFactor& f, const SparseMatrix& A, const std::vector<Index>& basis) {
  BasisFactor g;
  REQUIRE(g.factorize(A, basis) == FactorStatus::Ok);
  Rng rng(99);
  CHECK(compare_with_fresh(f, g, A.n_rows, rng) <= kUpdateTol);
}

// Position with the largest |x| (first on ties).
Index best_position(const std::vector<double>& x) {
  Index best = 0;
  for (Index p = 1; p < static_cast<Index>(x.size()); ++p) {
    if (std::fabs(x[to_size(p)]) > std::fabs(x[to_size(best)])) best = p;
  }
  return best;
}

// Smallest variable index outside the basis.
Index first_free_var(const std::vector<Index>& basis, Index total) {
  for (Index v = 0; v < total; ++v) {
    if (std::find(basis.begin(), basis.end(), v) == basis.end()) return v;
  }
  return -1;
}

// One simplex-like step: variable v (default: the smallest free variable) enters at the position with
// the largest pivot element. The basis is updated only when the update succeeds.
FactorStatus enter_one(BasisFactor& f, const SparseMatrix& A, std::vector<Index>& basis, Index v = -1) {
  if (v < 0) v = first_free_var(basis, A.n_cols + A.n_rows);
  const std::vector<double> x = ftran_column(f, A, v);
  const Index p = best_position(x);
  if (!(std::fabs(x[to_size(p)]) > 1e-6)) throw std::runtime_error("test fixture: no usable pivot");
  const FactorStatus st = f.update(p);
  if (st == FactorStatus::Ok) basis[to_size(p)] = v;
  return st;
}

}  // namespace

TEST_CASE(lu_update_limit_forces_refactor_then_continues) {
  Fixture fx;
  FactorParams params;
  params.max_updates = 3;
  BasisFactor f(params);
  REQUIRE(f.factorize(fx.A, fx.basis) == FactorStatus::Ok);
  for (int k = 0; k < 3; ++k) {
    REQUIRE(enter_one(f, fx.A, fx.basis) == FactorStatus::Ok);
    expect_matches_fresh(f, fx.A, fx.basis);
  }
  CHECK_EQ(f.update_count(), 3);
  // The 4th update hits the limit: NeedRefactor, factorization untouched.
  const std::vector<Index> before = fx.basis;
  CHECK(enter_one(f, fx.A, fx.basis) == FactorStatus::NeedRefactor);
  CHECK(fx.basis == before);
  CHECK_EQ(f.update_count(), 3);
  CHECK(f.valid());
  expect_matches_fresh(f, fx.A, fx.basis);  // still valid for the 3-times-updated basis
  // Refactor, then continue updating: three more updates fit again.
  const Index v = first_free_var(fx.basis, fx.A.n_cols + fx.A.n_rows);
  const std::vector<double> x = ftran_column(f, fx.A, v);
  const Index p = best_position(x);
  fx.basis[to_size(p)] = v;
  REQUIRE(f.factorize(fx.A, fx.basis) == FactorStatus::Ok);
  CHECK_EQ(f.update_count(), 0);
  expect_matches_fresh(f, fx.A, fx.basis);
  for (int k = 0; k < 3; ++k) {
    REQUIRE(enter_one(f, fx.A, fx.basis) == FactorStatus::Ok);
    expect_matches_fresh(f, fx.A, fx.basis);
  }
}

TEST_CASE(lu_update_declines_a_tiny_pivot_and_stays_valid) {
  // Entering column (1, 1e-13) against the identity: the pivot element at
  // position 1 is 1e-13, far below the update pivot tolerance.
  const SparseMatrix A = small_matrix(2, 3, {{0, 0, 1.0}, {1, 1, 1.0}, {0, 2, 1.0}, {1, 2, 1e-13}});
  BasisFactor f;
  const std::vector<Index> basis{0, 1};
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  const std::vector<double> x = ftran_column(f, A, 2);
  CHECK_NEAR(x[1], 1e-13, 1e-20);
  CHECK(f.update(1) == FactorStatus::NeedRefactor);
  CHECK_EQ(f.update_count(), 0);
  expect_matches_fresh(f, A, basis);  // unchanged: still valid for the old basis
  // Position 0 has a healthy pivot and the update goes through.
  const std::vector<double> x2 = ftran_column(f, A, 2);
  CHECK_NEAR(x2[0], 1.0, 1e-15);
  REQUIRE(f.update(0) == FactorStatus::Ok);
  expect_matches_fresh(f, A, {2, 1});
}

TEST_CASE(lu_update_mismatch_check_triggers_and_leaves_factorization_unchanged) {
  Fixture fx;
  // A negative tolerance makes any rounding difference between the two diagonal
  // computations count as unstable: this forces the stability check to fire.
  FactorParams strict;
  strict.update_mismatch_tol = -1.0;
  BasisFactor f(strict);
  REQUIRE(f.factorize(fx.A, fx.basis) == FactorStatus::Ok);
  const std::vector<double> x = ftran_column(f, fx.A, 6);
  REQUIRE(std::fabs(x[2]) > 1e-6);
  CHECK(f.update(2) == FactorStatus::NeedRefactor);
  CHECK_EQ(f.update_count(), 0);
  expect_matches_fresh(f, fx.A, fx.basis);
  // The same update with the default tolerance is accepted (the saved spike is still there).
  FactorParams normal;
  f.set_params(normal);
  REQUIRE(f.update(2) == FactorStatus::Ok);
  fx.basis[2] = 6;
  expect_matches_fresh(f, fx.A, fx.basis);
}

TEST_CASE(lu_update_growth_limit_triggers_before_the_update_limit) {
  // Identity basis, then dense columns enter: every update adds a column of
  // fill to U, so nnz(L)+nnz(U)+nnz(R) passes 3x the fresh count within a few updates.
  const Index m = 8;
  std::vector<Triplet> t;
  for (Index j = 0; j < m; ++j) t.push_back({j, j, 1.0});
  for (Index j = 0; j < m; ++j)
    for (Index i = 0; i < m; ++i) t.push_back({i, m + j, 1.0 + 0.1 * static_cast<double>((i * 7 + j * 3) % 5) + (i == j ? 2.0 : 0.0)});
  const SparseMatrix A = small_matrix(m, 2 * m, t);
  std::vector<Index> basis(to_size(m));
  for (Index j = 0; j < m; ++j) basis[to_size(j)] = j;
  FactorParams params;  // defaults: max_growth 3, max_updates 100
  BasisFactor f(params);
  REQUIRE(f.factorize(A, basis) == FactorStatus::Ok);
  bool triggered = false;
  int accepted = 0;
  for (int k = 0; k < m && !triggered; ++k) {
    const FactorStatus st = enter_one(f, A, basis, m + k);  // the dense columns, one by one
    if (st == FactorStatus::NeedRefactor) {
      triggered = true;
    } else {
      ++accepted;
    }
    expect_matches_fresh(f, A, basis);
  }
  CHECK(triggered);
  CHECK(accepted >= 1);
  CHECK(f.update_count() < params.max_updates);
  CHECK(static_cast<double>(f.nnz_l() + f.nnz_u() + f.stats().nnz_r) >
        params.max_growth * static_cast<double>(m));  // the fresh factor had nnz(U) = m
}

TEST_CASE(lu_update_requires_a_saved_spike) {
  Fixture fx;
  BasisFactor f;
  REQUIRE(f.factorize(fx.A, fx.basis) == FactorStatus::Ok);
  CHECK(f.update(0) == FactorStatus::NeedRefactor);  // nothing saved yet
  SparseWork w(6);
  load_column(fx.A, 6, w);
  f.ftran(w, false);  // no spike requested
  CHECK(f.update(2) == FactorStatus::NeedRefactor);
  const std::vector<double> x = ftran_column(f, fx.A, 6);
  REQUIRE(std::fabs(x[2]) > 1e-6);
  // Other solves between the saved ftran and the update do not disturb the spike.
  SparseWork other(6);
  other.set(3, 1.0);
  f.ftran(other, false);
  SparseWork other2(6);
  other2.set(1, 1.0);
  f.btran(other2);
  REQUIRE(f.update(2) == FactorStatus::Ok);
  fx.basis[2] = 6;
  expect_matches_fresh(f, fx.A, fx.basis);
  CHECK(f.update(2) == FactorStatus::NeedRefactor);  // the spike was consumed
  bool threw = false;
  try { f.update(6); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}
