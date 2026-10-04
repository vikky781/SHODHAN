// Quadratic objective: QPS reading and writing, hand-computed objective values, the QP KKT check and the
// convexity test (docs/QP.md).

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "shodhan/kkt.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/quadratic.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::testing;

namespace {

// Q = [[2, 1, 0], [1, 4, -1], [0, -1, 6]]; c = (1, -2, 0); the objective row has RHS -5, which is an offset of +5
// (MPS: the RHS of the objective row is minus the constant). One row X1 + X2 + X3 >= 1.
//
// At x = (1, 2, 3):  x^T Q x = 2*1 + 4*4 + 6*9 + 2*(1*1*2) + 2*(-1*2*3) = 2 + 16 + 54 + 4 - 12 = 64,
// so the quadratic part is 32; c^T x = 1 - 4 + 0 = -3; offset 5; total 34.
const char* kHead =
    "NAME qp3\n"
    "ROWS\n N OBJ\n G R1\n"
    "COLUMNS\n"
    "    X1  OBJ  1  R1  1\n"
    "    X2  OBJ  -2  R1  1\n"
    "    X3  R1  1\n"
    "RHS\n    RHS  OBJ  -5  R1  1\n"
    "BOUNDS\n FR BND X1\n FR BND X2\n FR BND X3\n";

std::string with(const std::string& quad) { return std::string(kHead) + quad + "ENDATA\n"; }

LpModel read_ok(const std::string& text) {
  const MpsReadResult r = read_mps_string(text, "t.qps");
  if (!r.ok) std::cerr << r.error << "\n";
  CHECK(r.ok);
  return r.model;
}

std::string read_error(const std::string& text) {
  const MpsReadResult r = read_mps_string(text, "t.qps");
  CHECK(!r.ok);
  return r.error;
}

const std::vector<double> kX = {1.0, 2.0, 3.0};

}  // namespace

TEST_CASE(qps_quadobj_lower_triangle_only) {
  const LpModel m = read_ok(with("QUADOBJ\n    X1  X1  2\n    X2  X1  1\n    X2  X2  4\n    X3  X2  -1\n    X3  X3  6\n"));
  CHECK_EQ(m.quadratic.nnz(), std::size_t{5});
  CHECK(m.validate().empty());
  CHECK_NEAR(quad_form(m.quadratic, kX), 64.0, 1e-12);          // hand computed above
  CHECK_NEAR(model_objective(m, kX), 34.0, 1e-12);              // 5 - 3 + 32
}

TEST_CASE(qps_quadobj_upper_triangle_only_gives_the_same_matrix) {
  const LpModel lo = read_ok(with("QUADOBJ\n    X1  X1  2\n    X2  X1  1\n    X2  X2  4\n    X3  X2  -1\n    X3  X3  6\n"));
  const LpModel up = read_ok(with("QUADOBJ\n    X1  X1  2\n    X1  X2  1\n    X2  X2  4\n    X2  X3  -1\n    X3  X3  6\n"));
  CHECK(lo.quadratic == up.quadratic);
  CHECK_NEAR(model_objective(up, kX), 34.0, 1e-12);
  // Several entries on one line: "col1 col2 v col3 v".
  const LpModel two = read_ok(with("QUADOBJ\n    X1  X1  2  X2  1\n    X2  X2  4  X3  -1\n    X3  X3  6\n"));
  CHECK(lo.quadratic == two.quadratic);
}

TEST_CASE(qps_qmatrix_lists_both_triangles) {
  const LpModel m = read_ok(with("QMATRIX\n    X1  X1  2\n    X1  X2  1\n    X2  X1  1\n    X2  X2  4\n    X2  X3  -1\n    X3  X2  -1\n    X3  X3  6\n"));
  CHECK_EQ(m.quadratic.nnz(), std::size_t{5});
  CHECK_NEAR(model_objective(m, kX), 34.0, 1e-12);
  const LpModel q = read_ok(with("QUADOBJ\n    X1  X1  2\n    X2  X1  1\n    X2  X2  4\n    X3  X2  -1\n    X3  X3  6\n"));
  CHECK(m.quadratic == q.quadratic);
}

TEST_CASE(qps_diagonal_and_off_diagonal_contributions_by_hand) {
  // One diagonal entry q_11 = 6: contributes (1/2)*6*x1^2 = 3 x1^2; one off-diagonal q_21 = 5: contributes
  // 5 x1 x2 once. At x = (2, 3, 0): 3*4 + 5*6 = 42.
  const LpModel m = read_ok(with("QUADOBJ\n    X1  X1  6\n    X2  X1  5\n"));
  const std::vector<double> x = {2.0, 3.0, 0.0};
  CHECK_NEAR(0.5 * quad_form(m.quadratic, x), 42.0, 1e-12);
  std::vector<double> y(3);
  quad_multiply(m.quadratic, x, y);  // Q x = (6*2 + 5*3, 5*2, 0)
  CHECK_NEAR(y[0], 27.0, 1e-12);
  CHECK_NEAR(y[1], 10.0, 1e-12);
  CHECK_NEAR(y[2], 0.0, 1e-12);
}

TEST_CASE(qps_errors_are_clear) {
  const std::string e1 = read_error(with("QUADOBJ\n    X1  X2  1\n    X2  X1  1\n"));
  CHECK(e1.find("duplicate QUADOBJ entry") != std::string::npos);
  const std::string e2 = read_error(with("QMATRIX\n    X1  X2  1\n    X2  X1  2\n"));
  CHECK(e2.find("not symmetric") != std::string::npos);
  const std::string e3 = read_error(with("QMATRIX\n    X1  X2  1\n"));
  CHECK(e3.find("no mirror entry") != std::string::npos);
  const std::string e4 = read_error(with("QSECTION\n    X1  X1  1\n"));
  CHECK(e4.find("quadratic constraints not supported") != std::string::npos);
  const std::string e5 = read_error(with("QCMATRIX R1\n    X1  X1  1\n"));
  CHECK(e5.find("quadratic constraints not supported") != std::string::npos);
  const std::string e6 = read_error(with("QUADOBJ\n    X1  NOPE  1\n"));
  CHECK(e6.find("unknown column name") != std::string::npos);
  const std::string e7 = read_error(with("QUADOBJ\n    X1  X1  1\nQMATRIX\n    X1  X1  1\n"));
  CHECK(e7.find("cannot both be given") != std::string::npos);
}

TEST_CASE(qps_max_sense_negates_in_the_minimization_form) {
  // maximize c^T x + (1/2) x^T Q x with Q = -I (concave): convex after the sign change.
  LpModel m = read_ok(std::string("NAME mx\nOBJSENSE\n    MAX\nROWS\n N OBJ\n L R1\nCOLUMNS\n    X1  OBJ  3  R1  1\n    X2  OBJ  1  R1  1\nRHS\n    RHS  R1  4\nQUADOBJ\n    X1  X1  -1\n    X2  X2  -1\nENDATA\n"));
  CHECK(m.sense == Sense::Maximize);
  CHECK(check_convexity(m).convex);
  m.quadratic.value[0] = 1.0;  // now maximizing a convex function: not convex
  const ConvexityReport r = check_convexity(m);
  CHECK(!r.convex);
  CHECK(r.decided);
}

TEST_CASE(qps_write_read_round_trip) {
  Rng rng(7);
  for (int t = 0; t < 40; ++t) {
    const int n = rng.range(2, 8);
    LpModel m;
    m.name = "rt" + std::to_string(t);
    m.n_cols = n;
    m.n_rows = 2;
    m.sense = rng.chance(0.3) ? Sense::Maximize : Sense::Minimize;
    m.objective_offset = rng.range(-3, 3);
    std::vector<Triplet> a, q;
    for (int j = 0; j < n; ++j) {
      m.col_cost.push_back(rng.range(-5, 5));
      m.col_lower.push_back(rng.chance(0.3) ? -kInf : 0.0);
      m.col_upper.push_back(rng.chance(0.5) ? kInf : 10.0);
      m.col_type.push_back(ColType::Continuous);
      a.push_back({0, j, static_cast<double>(rng.range(1, 3))});
      for (int i = j; i < n; ++i) {
        if (rng.chance(0.4)) q.push_back({i, j, static_cast<double>(rng.range(-4, 4)) + (rng.chance(0.3) ? 0.5 : 0.0)});
      }
    }
    m.row_lower = {1.0, -kInf};
    m.row_upper = {kInf, 5.0};
    a.push_back({1, 0, 1.0});
    std::string err;
    REQUIRE(SparseMatrix::from_triplets(2, n, a, &m.A, &err));
    std::vector<Triplet> qn;
    for (const Triplet& tr : q) {
      if (tr.value != 0.0) qn.push_back(tr);
    }
    if (!qn.empty()) REQUIRE(SparseMatrix::from_triplets(n, n, qn, &m.quadratic, &err));
    REQUIRE(m.validate().empty());
    std::ostringstream os;
    REQUIRE(write_mps(m, os, &err));
    const MpsReadResult back = read_mps_string(os.str(), "rt.qps");
    REQUIRE(back.ok);
    CHECK(back.model.quadratic == m.quadratic);
    CHECK(back.model.A == m.A);
    CHECK(back.model.col_cost == m.col_cost);
    CHECK(back.model.objective_offset == m.objective_offset);
  }
}

namespace {

// One fixed-format data line: fields at columns 5, 15, 25-36 (right aligned), 40, 50-61.
std::string fixed(const std::string& f2, const std::string& f3, const std::string& f4, const std::string& f5 = "",
                  const std::string& f6 = "") {
  auto pad = [](std::string s, std::size_t w) { return s.size() < w ? s + std::string(w - s.size(), ' ') : s; };
  auto right = [](std::string s, std::size_t w) { return s.size() < w ? std::string(w - s.size(), ' ') + s : s; };
  std::string line = "    " + pad(f2, 8) + "  " + pad(f3, 8) + "  " + right(f4, 12);
  if (!f5.empty()) line += "   " + pad(f5, 8) + "  " + right(f6, 12);
  return line + "\n";
}

}  // namespace

TEST_CASE(qps_fixed_format_with_spaces_in_names) {
  const std::string text = std::string("NAME          fx\n") + "ROWS\n" + " N  OBJ\n" + " G  R1\n" + "COLUMNS\n" +
                           fixed("X 1", "OBJ", "1", "R1", "1") + fixed("X 2", "OBJ", "-2", "R1", "1") + "RHS\n" +
                           fixed("RHS", "R1", "1") + "QUADOBJ\n" + fixed("X 1", "X 1", "2") + fixed("X 2", "X 1", "1") +
                           fixed("X 2", "X 2", "4") + "ENDATA\n";
  const MpsReadResult r = read_mps_string(text, "fx.qps");
  if (!r.ok) std::cerr << r.error << "\n";
  CHECK(r.ok);
  CHECK(r.detected_format == MpsFormat::Fixed);
  CHECK_EQ(r.model.quadratic.nnz(), std::size_t{3});
  const std::vector<double> x = {1.0, 1.0};  // x^T Q x = 2 + 4 + 2*1 = 8 -> 4
  CHECK_NEAR(0.5 * quad_form(r.model.quadratic, x), 4.0, 1e-12);
}

// ---------------------------------------------------------------------------------------------------------
// KKT check on QPs with a known optimum.
namespace {

// min 0.5 (x1^2 + x2^2) - x1 - 2 x2 + 3 s.t. x1 + x2 <= 1, x >= 0.
// Unconstrained optimum (1, 2) is infeasible. KKT: x1 - 1 + y' = 0... with the sign rule d = c + Qx - A^T y and the
// row <= (y <= 0): x = (0, 1)?  Solve: Lagrange: x1 - 1 + mu = 0, x2 - 2 + mu = 0, x1 + x2 = 1, mu >= 0 -> mu = 1,
// x = (0, 1). Then x1 = 0 is at its lower bound with reduced cost d1 = c1 + x1 - A^T y = -1 + 0 - (1)(-1) = 0 (y = -1).
// Objective: 0.5*1 - 2 + 3 = 1.5.
LpModel qp_example() {
  LpModel m;
  m.n_rows = 1;
  m.n_cols = 2;
  m.col_cost = {-1.0, -2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInf, kInf};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {-kInf};
  m.row_upper = {1.0};
  m.objective_offset = 3.0;
  std::string err;
  SparseMatrix::from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, &m.A, &err);
  SparseMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}}, &m.quadratic, &err);
  return m;
}

}  // namespace

TEST_CASE(kkt_accepts_the_known_qp_optimum) {
  const LpModel m = qp_example();
  Solution s;
  s.x = {0.0, 1.0};
  s.y = {-1.0};
  s.d = {0.0, 0.0};
  s.objective = 1.5;
  const KktReport r = check_kkt(m, s, 1e-9);
  CHECK(r.ok);
  CHECK_NEAR(r.primal_objective, 1.5, 1e-12);
  CHECK_NEAR(r.dual_objective, 1.5, 1e-12);  // y*b = -1*1, -0.5 x'Qx = -0.5, offset 3 -> 1.5
  CHECK(r.gap_abs < 1e-12);
}

TEST_CASE(kkt_gap_equals_the_complementarity_sum_for_a_feasible_non_optimal_point) {
  // x = (0.5, 0.5) is feasible; take y = -1. d = c + Qx - A^T y = (-1+0.5+1, -2+0.5+1) = (0.5, -0.5).
  // Column 2 has d < 0 with no upper bound: a dual infeasibility, reported as such; the gap identity needs the sign
  // rules, so check it at a point where they hold: y = -1.5: d = (1, 0) (x1 > 0 with d1 > 0 is not complementary).
  const LpModel m = qp_example();
  Solution s;
  s.x = {0.5, 0.5};
  s.y = {-1.5};
  s.objective = 0.25 - 0.5 - 1.0 + 3.0;  // 0.5*(0.25+0.25) - 0.5 - 1 + 3 = 1.75
  const KktReport r = check_kkt(m, s, 1e-9);
  CHECK(!r.ok);
  // d = c + Qx - A^T y = (-1 + 0.5 + 1.5, -2 + 0.5 + 1.5) = (1, 0); activities: row at its bound (x1+x2 = 1).
  // gap = sum d_j (x_j - lb_j) + sum y_i (activity - bound) = 1*0.5 + 0 + (-1.5)*(1-1) = 0.5.
  CHECK_NEAR(r.gap_abs, 0.5, 1e-12);
  CHECK_NEAR(r.complementarity_abs, 0.5, 1e-12);
}

TEST_CASE(kkt_detects_corrupted_qp_solutions) {
  const LpModel m = qp_example();
  Solution good;
  good.x = {0.0, 1.0};
  good.y = {-1.0};
  good.d = {0.0, 0.0};
  good.objective = 1.5;
  Solution s = good;
  s.x[1] = 1.1;  // infeasible row
  CHECK(!check_kkt(m, s, 1e-9).ok);
  s = good;
  s.y[0] = -1.2;  // wrong multiplier: d no longer matches, gap opens
  CHECK(!check_kkt(m, s, 1e-9).ok);
  s = good;
  s.objective = 1.6;  // wrong claimed objective
  CHECK(!check_kkt(m, s, 1e-9).ok);
  s = good;
  s.d = {0.0, 0.3};  // inconsistent supplied d
  CHECK(!check_kkt(m, s, 1e-9).ok);
  s = good;
  s.y[0] = 1.0;  // wrong sign
  CHECK(!check_kkt(m, s, 1e-9).ok);
}

// ---------------------------------------------------------------------------------------------------------
// Convexity.
namespace {

LpModel model_with_q(Index n, const std::vector<Triplet>& lower, Sense sense = Sense::Minimize) {
  LpModel m;
  m.n_cols = n;
  m.n_rows = 0;
  m.sense = sense;
  m.col_cost.assign(to_size(n), 0.0);
  m.col_lower.assign(to_size(n), 0.0);
  m.col_upper.assign(to_size(n), kInf);
  m.col_type.assign(to_size(n), ColType::Continuous);
  m.A = SparseMatrix(0, n);
  std::string err;
  if (!lower.empty()) CHECK(SparseMatrix::from_triplets(n, n, lower, &m.quadratic, &err));
  return m;
}

}  // namespace

TEST_CASE(convexity_positive_definite_semidefinite_and_indefinite) {
  // PD: [[2,1],[1,2]].
  ConvexityReport r = check_convexity(model_with_q(2, {{0, 0, 2.0}, {1, 0, 1.0}, {1, 1, 2.0}}));
  CHECK(r.convex);
  CHECK_EQ(r.rank, Index{2});
  // PSD of rank one: [[1,1],[1,1]].
  r = check_convexity(model_with_q(2, {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}}));
  CHECK(r.convex);
  CHECK_EQ(r.rank, Index{1});
  // PSD with a zero row and column: diag(2, 0, 3).
  r = check_convexity(model_with_q(3, {{0, 0, 2.0}, {2, 2, 3.0}}));
  CHECK(r.convex);
  CHECK_EQ(r.rank, Index{2});
  // Indefinite: [[1,2],[2,1]] has eigenvalues 3 and -1.
  r = check_convexity(model_with_q(2, {{0, 0, 1.0}, {1, 0, 2.0}, {1, 1, 1.0}}));
  CHECK(!r.convex);
  CHECK(r.pivot < 0.0);
  CHECK(r.column >= 0);
  // Negative diagonal entry.
  r = check_convexity(model_with_q(3, {{0, 0, 1.0}, {1, 1, -0.5}, {2, 2, 1.0}}));
  CHECK(!r.convex);
  CHECK_EQ(r.column, Index{1});
  // Zero diagonals with a nonzero off-diagonal: [[0,1],[1,0]] is indefinite.
  r = check_convexity(model_with_q(2, {{1, 0, 1.0}}));
  CHECK(!r.convex);
  // No quadratic term at all.
  CHECK(check_convexity(model_with_q(2, {})).convex);
}

TEST_CASE(convexity_agrees_with_random_gram_and_perturbed_matrices) {
  Rng rng(11);
  int psd = 0, indef = 0;
  for (int t = 0; t < 200; ++t) {
    const int n = rng.range(2, 9);
    const int rank = rng.range(1, n);
    std::vector<std::vector<double>> l(to_size(n), std::vector<double>(to_size(rank)));
    for (auto& row : l) {
      for (double& v : row) v = rng.range(-3, 3);
    }
    std::vector<Triplet> low;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = 0; k < rank; ++k) s += l[to_size(i)][to_size(k)] * l[to_size(j)][to_size(k)];
        if (s != 0.0) low.push_back({i, j, s});
      }
    }
    if (low.empty()) continue;
    CHECK(check_convexity(model_with_q(n, low)).convex);  // a Gram matrix is PSD whatever its rank
    ++psd;
    // Subtract a large multiple of one diagonal entry: the matrix gets a negative eigenvalue.
    std::vector<Triplet> bad = low;
    bool found = false;
    for (Triplet& tr : bad) {
      if (tr.row == tr.col) {
        tr.value -= 1000.0;
        found = true;
        break;
      }
    }
    if (!found) bad.push_back({0, 0, -1000.0});
    CHECK(!check_convexity(model_with_q(n, bad)).convex);
    ++indef;
  }
  std::cout << "  convexity: " << psd << " Gram matrices accepted, " << indef << " perturbed ones rejected\n";
}
