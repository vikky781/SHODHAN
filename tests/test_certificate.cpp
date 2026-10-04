#include <sstream>
#include <string>
#include <vector>

#include "shodhan/certificate.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/rays.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

LpModel named_model() {
  // min x + 2y  s.t.  x + y >= 3  (row named with a space and a quote), x, y >= 0.
  LpModel m;
  m.name = "my model";
  m.n_rows = 1;
  m.n_cols = 2;
  std::string err;
  SparseMatrix::from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}}, &m.A, &err);
  m.col_cost = {1.0, 2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInf, kInf};
  m.col_type = {ColType::Continuous, ColType::Continuous};
  m.row_lower = {3.0};
  m.row_upper = {kInf};
  m.col_names = {"x 1", "y\"q\\"};
  m.row_names = {"row \"A\""};
  return m;
}

CertificateContext context(const LpModel& m) {
  CertificateContext c;
  c.solver_version = "test";
  c.problem_name = m.name;
  c.file_sha256 = std::string(64, 'a');
  return c;
}

std::string text(const LpModel& m, const LpResult& r) {
  std::ostringstream os;
  write_certificate(m, context(m), r, os);
  return os.str();
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

}  // namespace

TEST_CASE(certificate_optimal_uses_names_and_escapes_them) {
  const LpModel m = named_model();
  const LpResult r = LpSolver().solve(m);
  REQUIRE(r.status == Status::Optimal);
  const std::string s = text(m, r);
  CHECK(has(s, "\"format\": \"shodhan-cert\""));
  CHECK(has(s, "\"version\": 1"));
  CHECK(has(s, "\"status\": \"optimal\""));
  CHECK(has(s, "\"name\": \"my model\""));
  CHECK(has(s, "\"sense\": \"min\""));
  CHECK(has(s, "\"claimed_objective\": 3"));
  CHECK(has(s, "\"x 1\": 3"));            // the nonzero column, name with a space
  CHECK(!has(s, "y\\\"q"));               // y = 0 is not listed
  CHECK(has(s, "\"row \\\"A\\\"\": 1"));  // the dual, name with quotes escaped
  CHECK(has(s, "\"configuration\""));
  CHECK(has(s, "\"count\": 1"));
}

TEST_CASE(certificate_escapes_a_backslash_in_a_listed_name) {
  LpModel m = named_model();
  m.col_cost = {2.0, 1.0};  // now y is the cheaper column: y = 3
  const LpResult r = LpSolver().solve(m);
  REQUIRE(r.status == Status::Optimal);
  const std::string s = text(m, r);
  CHECK(has(s, "\"y\\\"q\\\\\": 3"));
}

TEST_CASE(certificate_infeasible_and_unbounded_carry_their_bodies) {
  LpModel inf = named_model();
  inf.col_upper = {1.0, 1.0};  // x + y >= 3 but x, y <= 1
  const LpResult ri = LpSolver().solve(inf);
  REQUIRE(ri.status == Status::Infeasible);
  const std::string si = text(inf, ri);
  CHECK(has(si, "\"status\": \"infeasible\""));
  CHECK(has(si, "\"farkas\""));
  CHECK(!has(si, "claimed_objective"));
  CHECK(!has(si, "\"x\":"));

  LpModel unb = named_model();
  unb.col_cost = {-1.0, -1.0};
  unb.row_lower = {-kInf};
  unb.row_upper = {kInf};  // free row: both columns can grow forever
  const LpResult ru = LpSolver().solve(unb);
  REQUIRE(ru.status == Status::Unbounded);
  CHECK(!ru.unbounded_point.empty());
  CHECK(max_relative_violation(unb, ru.unbounded_point) <= 1e-6);
  const std::string su = text(unb, ru);
  CHECK(has(su, "\"status\": \"unbounded\""));
  CHECK(has(su, "\"point\""));
  CHECK(has(su, "\"ray\""));
}

TEST_CASE(certificate_for_a_result_without_evidence_is_status_other) {
  const LpModel m = named_model();
  for (const Status st : {Status::NumericalError, Status::TimeLimit, Status::IterationLimit, Status::NotImplemented}) {
    LpResult r;
    r.status = st;
    const std::string s = text(m, r);
    CHECK(has(s, "\"status\": \"other\""));
    CHECK(!has(s, "\"x\":") && !has(s, "farkas") && !has(s, "\"ray\"") && !has(s, "claimed_objective"));
  }
  // An Unbounded result without a feasible point is not a proof either.
  LpResult r;
  r.status = Status::Unbounded;
  r.unbounded_ray = {1.0, 1.0};
  CHECK_EQ(certificate_status(r), "other");
}

TEST_CASE(certificate_uses_default_names_for_a_model_without_names) {
  LpModel m = named_model();
  m.col_names.clear();
  m.row_names.clear();
  const LpResult r = LpSolver().solve(m);
  REQUIRE(r.status == Status::Optimal);
  const std::string s = text(m, r);
  CHECK(has(s, "\"C1\": 3"));
  CHECK(has(s, "\"R1\": 1"));
}

namespace {

std::string mip_text(const LpModel& m, const mip::MipResult& r) {
  std::ostringstream os;
  write_mip_certificate(m, context(m), r, os);
  return os.str();
}

}  // namespace

TEST_CASE(mip_certificate_of_an_incumbent_is_feasible_with_unverified_bound_fields) {
  const LpModel m = named_model();
  mip::MipResult r;
  r.status = Status::NodeLimit;  // stopped early: still a verified incumbent
  r.has_solution = true;
  r.solution.x = {3.0, 0.0};
  r.objective = 3.0;
  r.has_bound = true;
  r.best_bound = 2.5;
  r.abs_gap = 0.5;
  r.rel_gap = 0.125;
  r.nodes_processed = 7;
  const std::string s = mip_text(m, r);
  CHECK_EQ(mip_certificate_status(r), "feasible");
  CHECK(has(s, "\"status\": \"feasible\""));
  CHECK(has(s, "\"optimality_certified\": false"));
  CHECK(has(s, "\"claimed_objective\": 3"));
  CHECK(has(s, "\"claimed_best_bound\": 2.5"));
  CHECK(has(s, "\"claimed_gap\": 0.125"));
  CHECK(has(s, "\"nodes\": 7"));
  CHECK(has(s, "\"mip_status\": \"NodeLimit\""));
  CHECK(has(s, "\"x\":"));
  CHECK(!has(s, "\"y\":"));
  CHECK(!has(s, "farkas"));
}

TEST_CASE(mip_certificate_infeasible_has_a_body_only_when_the_lp_relaxation_is_infeasible) {
  const LpModel m = named_model();
  mip::MipResult branch;
  branch.status = Status::Infeasible;  // proved by branching
  std::string s = mip_text(m, branch);
  CHECK_EQ(mip_certificate_status(branch), "infeasible");
  CHECK(has(s, "\"status\": \"infeasible\""));
  CHECK(has(s, "\"certified\": false"));
  CHECK(!has(s, "farkas"));

  mip::MipResult lp;
  lp.status = Status::Infeasible;
  lp.lp_infeasible_certified = true;
  lp.lp_farkas = {1.0};
  s = mip_text(m, lp);
  CHECK(has(s, "\"certified\": true"));
  CHECK(has(s, "farkas"));
  CHECK(has(s, "\"y\":"));
}

TEST_CASE(mip_certificate_without_a_solution_or_an_infeasibility_proof_is_other) {
  const LpModel m = named_model();
  for (const Status st : {Status::TimeLimit, Status::NodeLimit, Status::InfeasibleOrUnbounded, Status::NumericalError}) {
    mip::MipResult r;
    r.status = st;
    CHECK_EQ(mip_certificate_status(r), "other");
    const std::string s = mip_text(m, r);
    CHECK(has(s, "\"status\": \"other\""));
    CHECK(has(s, "\"optimality_certified\": false"));
    CHECK(!has(s, "claimed_objective"));
  }
}
