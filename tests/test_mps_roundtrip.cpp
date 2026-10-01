#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "mps_samples.hpp"
#include "shodhan/mps.hpp"
#include "test_harness.hpp"

using namespace shodhan;

namespace {

// Platform-independent generator (the std distributions are not).
struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed) {}
  std::uint64_t next() {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  int range(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

// Mix of exactly representable values and arbitrary doubles.
double rand_value(Rng& rng) {
  switch (rng.range(0, 3)) {
    case 0:
      return static_cast<double>(rng.range(-20, 20));
    case 1:
      return static_cast<double>(rng.range(-160, 160)) / 8.0;
    case 2:
      return (rng.unit() * 2.0 - 1.0) * std::pow(10.0, rng.range(-6, 6));
    default:
      return (rng.unit() * 2.0 - 1.0) * 1000.0;
  }
}

// Two distinct values, ordered.
void rand_pair(Rng& rng, double* lo, double* hi) {
  double a = rand_value(rng);
  double b = rand_value(rng);
  if (a == b) b = a + 1.0;
  *lo = std::min(a, b);
  *hi = std::max(a, b);
}

LpModel random_model(Rng& rng, int id) {
  LpModel m;
  m.name = "gen" + std::to_string(id);
  m.objective_name = rng.range(0, 1) == 0 ? "OBJ" : "COST";
  m.sense = rng.range(0, 1) == 0 ? Sense::Minimize : Sense::Maximize;
  m.objective_offset = rng.range(0, 2) == 0 ? 0.0 : rand_value(rng);

  const int rows = rng.range(0, 12);
  const int cols = rng.range(1, 12);
  m.n_rows = rows;
  m.n_cols = cols;
  const double density = 0.1 + 0.6 * rng.unit();

  std::vector<Triplet> triplets;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (rng.unit() < density) {
        const double v = rng.range(0, 19) == 0 ? 0.0 : rand_value(rng);  // some explicit zeros
        triplets.push_back({i, j, v});
      }
    }
  }
  std::string err;
  if (!SparseMatrix::from_triplets(rows, cols, triplets, &m.A, &err)) {
    throw std::runtime_error("generator: " + err);
  }

  for (int j = 0; j < cols; ++j) {
    m.col_cost.push_back(rng.range(0, 2) == 0 ? 0.0 : rand_value(rng));
    double lo = 0.0;
    double up = kInf;
    switch (rng.range(0, 7)) {
      case 0:
        break;  // [0, inf)
      case 1:
        up = std::fabs(rand_value(rng)) + 0.5;  // [0, up]
        break;
      case 2:
        lo = rand_value(rng);  // [lo, inf)
        break;
      case 3:
        rand_pair(rng, &lo, &up);  // boxed
        break;
      case 4:
        lo = up = rand_value(rng);  // fixed
        break;
      case 5:
        lo = -kInf;  // free
        break;
      case 6:
        lo = -kInf;  // (-inf, up]
        up = rand_value(rng);
        break;
      default:
        lo = 0.0;  // [0, up] with up possibly 0
        up = static_cast<double>(rng.range(0, 5));
        break;
    }
    ColType type = ColType::Continuous;
    switch (rng.range(0, 4)) {
      case 0:
        type = ColType::Integer;
        break;
      case 1:
        type = ColType::Binary;
        lo = 0.0;
        up = 1.0;
        break;
      default:
        break;
    }
    m.col_lower.push_back(lo);
    m.col_upper.push_back(up);
    m.col_type.push_back(type);
    m.col_names.push_back("c_" + std::to_string(j));
  }

  for (int i = 0; i < rows; ++i) {
    double lo = 0.0;
    double up = 0.0;
    switch (rng.range(0, 4)) {
      case 0:
        lo = -kInf;
        up = rand_value(rng);
        break;
      case 1:
        lo = rand_value(rng);
        up = kInf;
        break;
      case 2:
        lo = up = rand_value(rng);
        break;
      case 3:
        // Ranged rows use multiples of 1/8 so that (rhs, width) encodes the
        // bounds exactly; arbitrary doubles are covered by a tolerance test.
        lo = static_cast<double>(rng.range(-160, 160)) / 8.0;
        up = lo + static_cast<double>(rng.range(1, 80)) / 8.0;
        break;
      default:
        lo = -kInf;
        up = kInf;
        break;
    }
    m.row_lower.push_back(lo);
    m.row_upper.push_back(up);
    m.row_names.push_back("r_" + std::to_string(i));
  }
  return m;
}

std::string vec_diff(const char* what, const std::vector<double>& a, const std::vector<double>& b) {
  if (a == b) return "";
  std::ostringstream os;
  os.precision(17);
  os << what;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i] != b[i]) {
      os << "[" << i << "]: " << a[i] << " vs " << b[i];
      break;
    }
  }
  return os.str();
}

// Description of the first field that differs, or empty when identical.
std::string first_difference(const LpModel& a, const LpModel& b) {
  if (std::string d = vec_diff("row_lower", a.row_lower, b.row_lower); !d.empty()) return d;
  if (std::string d = vec_diff("row_upper", a.row_upper, b.row_upper); !d.empty()) return d;
  if (a.name != b.name) return "name";
  if (a.objective_name != b.objective_name) return "objective_name";
  if (a.sense != b.sense) return "sense";
  if (a.objective_offset != b.objective_offset) return "objective_offset";
  if (a.n_rows != b.n_rows) return "n_rows";
  if (a.n_cols != b.n_cols) return "n_cols";
  if (!(a.A == b.A)) return "A";
  if (a.col_cost != b.col_cost) return "col_cost";
  if (a.col_lower != b.col_lower) return "col_lower";
  if (a.col_upper != b.col_upper) return "col_upper";
  if (a.col_type != b.col_type) return "col_type";
  if (a.row_lower != b.row_lower) return "row_lower";
  if (a.row_upper != b.row_upper) return "row_upper";
  if (a.row_names != b.row_names) return "row_names";
  if (a.col_names != b.col_names) return "col_names";
  if (!(a.quadratic == b.quadratic)) return "quadratic";
  return "";
}

std::string write_to_string(const LpModel& m, bool* ok = nullptr, std::string* err = nullptr) {
  std::ostringstream os;
  const bool wrote = write_mps(m, os, err);
  if (ok != nullptr) *ok = wrote;
  return os.str();
}

}  // namespace

TEST_CASE(mps_round_trip_every_embedded_model) {
  for (const auto& sample : samples::all_valid()) {
    const MpsReadResult first = read_mps_string(sample.second, sample.first);
    REQUIRE(first.ok);
    bool ok = false;
    std::string err;
    const std::string text = write_to_string(first.model, &ok, &err);
    if (!ok) {
      CHECK_EQ(sample.first + ": write failed: " + err, std::string());
      continue;
    }
    const MpsReadResult second = read_mps_string(text, sample.first + " (rewritten)");
    if (!second.ok) {
      CHECK_EQ(sample.first + ": reread failed: " + second.error, std::string());
      continue;
    }
    const std::string diff = first_difference(first.model, second.model);
    if (!diff.empty()) CHECK_EQ(sample.first + ": differs in " + diff, std::string());
    CHECK(first.model == second.model);
    CHECK_EQ(write_to_string(second.model), text);  // writing is idempotent
  }
}

TEST_CASE(mps_round_trip_200_random_models) {
  Rng rng(20260101);
  int integer_cols = 0;
  int ranged_rows = 0;
  for (int id = 0; id < 200; ++id) {
    const LpModel model = random_model(rng, id);
    const std::vector<std::string> problems = model.validate();
    if (!problems.empty()) {
      CHECK_EQ("generator produced invalid model " + std::to_string(id) + ": " + problems.front(),
               std::string());
      continue;
    }
    for (const ColType t : model.col_type) integer_cols += t != ColType::Continuous ? 1 : 0;
    for (Index i = 0; i < model.n_rows; ++i) {
      const double lo = model.row_lower[to_size(i)];
      const double up = model.row_upper[to_size(i)];
      if (!is_inf(lo) && !is_inf(up) && lo != up) ++ranged_rows;
    }

    bool ok = false;
    std::string err;
    const std::string text = write_to_string(model, &ok, &err);
    if (!ok) {
      CHECK_EQ("model " + std::to_string(id) + ": write failed: " + err, std::string());
      continue;
    }
    const MpsReadResult back = read_mps_string(text, "random" + std::to_string(id));
    if (!back.ok) {
      CHECK_EQ("model " + std::to_string(id) + ": " + back.error, std::string());
      continue;
    }
    const std::string diff = first_difference(model, back.model);
    if (!diff.empty()) {
      CHECK_EQ("model " + std::to_string(id) + " differs in " + diff, std::string());
    }
    CHECK(model == back.model);
  }
  // Make sure the generator really exercised these features.
  CHECK(integer_cols > 100);
  CHECK(ranged_rows > 50);
}

TEST_CASE(mps_writer_output_is_deterministic) {
  const MpsReadResult r = read_mps_string(samples::kAllBounds, "x");
  REQUIRE(r.ok);
  CHECK_EQ(write_to_string(r.model), write_to_string(r.model));
}

TEST_CASE(mps_writer_generates_names_when_absent) {
  MpsReadResult r = read_mps_string(samples::kBasicFree, "x");
  REQUIRE(r.ok);
  r.model.row_names.clear();
  r.model.col_names.clear();
  bool ok = false;
  const std::string text = write_to_string(r.model, &ok);
  REQUIRE(ok);
  CHECK_CONTAINS(text, " R1");
  CHECK_CONTAINS(text, "C3 ");
  const MpsReadResult back = read_mps_string(text, "gen");
  REQUIRE(back.ok);
  CHECK((back.model.row_names == std::vector<std::string>{"R1", "R2", "R3"}));
  CHECK(back.model.A == r.model.A);
}

TEST_CASE(mps_writer_rejects_unrepresentable_models_without_output) {
  MpsReadResult r = read_mps_string(samples::kBasicFree, "x");
  REQUIRE(r.ok);

  LpModel spaced = r.model;
  spaced.col_names[0] = "has space";
  bool ok = true;
  std::string err;
  std::string text = write_to_string(spaced, &ok, &err);
  CHECK(!ok);
  CHECK(text.empty());
  CHECK_CONTAINS(err, "has space");

  LpModel dup = r.model;
  dup.row_names[1] = dup.row_names[0];
  text = write_to_string(dup, &ok, &err);
  CHECK(!ok);
  CHECK(text.empty());

  LpModel clash = r.model;
  clash.objective_name = "LIM1";  // collides with a row name
  text = write_to_string(clash, &ok, &err);
  CHECK(!ok);
  CHECK_CONTAINS(err, "duplicate");

  LpModel invalid = r.model;
  invalid.col_cost.pop_back();
  text = write_to_string(invalid, &ok, &err);
  CHECK(!ok);
  CHECK_CONTAINS(err, "invalid");
}

TEST_CASE(mps_writer_uses_shortest_round_trip_numbers) {
  LpModel m;
  m.name = "num";
  m.n_rows = 0;
  m.n_cols = 3;
  m.A = SparseMatrix(0, 3);
  m.col_cost = {0.1, 1.0 / 3.0, 1e-300};
  m.col_lower = {0.0, 0.0, 0.0};
  m.col_upper = {kInf, kInf, kInf};
  m.col_type.assign(3, ColType::Continuous);
  const std::string text = write_to_string(m);
  CHECK_CONTAINS(text, "  0.1\n");  // shortest form, not 0.10000000000000001
  const MpsReadResult back = read_mps_string(text, "num");
  REQUIRE(back.ok);
  CHECK(back.model.col_cost == m.col_cost);
}

TEST_CASE(mps_ranged_rows_with_arbitrary_doubles_stay_within_a_few_ulps) {
  // A range is stored as a width, so (lo, hi) pairs of arbitrary doubles are
  // not always reproduced exactly. Check the error stays tiny.
  Rng rng(777);
  int exact = 0;
  const int total = 2000;
  for (int k = 0; k < total; ++k) {
    LpModel m;
    m.name = "rng";
    m.n_rows = 1;
    m.n_cols = 1;
    std::string err;
    REQUIRE(SparseMatrix::from_triplets(1, 1, {{0, 0, 1.0}}, &m.A, &err));
    m.col_cost = {0.0};
    m.col_lower = {0.0};
    m.col_upper = {kInf};
    m.col_type = {ColType::Continuous};
    double lo = 0.0;
    double hi = 0.0;
    rand_pair(rng, &lo, &hi);
    m.row_lower = {lo};
    m.row_upper = {hi};

    bool ok = false;
    const std::string text = write_to_string(m, &ok);
    REQUIRE(ok);
    const MpsReadResult back = read_mps_string(text, "rng");
    REQUIRE(back.ok);
    const double scale = std::max(std::fabs(lo), std::fabs(hi));
    const double tol = 4.0 * 2.220446049250313e-16 * scale;
    CHECK(std::fabs(back.model.row_lower[0] - lo) <= tol);
    CHECK(std::fabs(back.model.row_upper[0] - hi) <= tol);
    if (back.model.row_lower[0] == lo && back.model.row_upper[0] == hi) ++exact;
  }
  // Informational lower bound only: most pairs are exact.
  CHECK(exact > total / 2);
}

TEST_CASE(mps_writer_falls_back_to_fixed_format_for_names_with_spaces) {
  const MpsReadResult first = read_mps_string(samples::fixed_with_spaces(), "x");
  REQUIRE(first.ok);
  bool ok = false;
  std::string err;
  const std::string text = write_to_string(first.model, &ok, &err);
  REQUIRE(ok);
  CHECK_CONTAINS(text, "COL A");
  const MpsReadResult back = read_mps_string(text, "rewritten");
  REQUIRE(back.ok);
  CHECK(back.detected_format == MpsFormat::Fixed);
  CHECK(back.model == first.model);

  // Names that do not fit the fixed layout are rejected, not truncated.
  LpModel longname = first.model;
  longname.col_names[0] = "COL WITH LONG NAME";
  write_to_string(longname, &ok, &err);
  CHECK(!ok);
  CHECK_CONTAINS(err, "longer than 8");

  // So are numbers that do not fit a 12-character field.
  LpModel wide = first.model;
  wide.col_cost[0] = 0.1 + 0.2;
  write_to_string(wide, &ok, &err);
  CHECK(!ok);
  CHECK_CONTAINS(err, "12-character");
}

TEST_CASE(mps_fixed_format_with_spaces_and_integer_markers) {
  MpsReadResult r = read_mps_string(samples::fixed_with_spaces(), "x");
  REQUIRE(r.ok);
  r.model.col_type = {ColType::Integer, ColType::Binary};
  r.model.col_lower[1] = 0.0;
  r.model.col_upper[1] = 1.0;
  bool ok = false;
  const std::string text = write_to_string(r.model, &ok);
  REQUIRE(ok);
  const MpsReadResult back = read_mps_string(text, "markers");
  REQUIRE(back.ok);
  CHECK(back.detected_format == MpsFormat::Fixed);
  CHECK(back.model == r.model);
}
