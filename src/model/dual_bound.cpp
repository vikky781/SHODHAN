#include "shodhan/dual_bound.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "shodhan/constants.hpp"

namespace shodhan {

namespace {

// Evaluation type: long double (64-bit mantissa on x86, otherwise at least double). The margins use its real
// unit roundoff, so the error bounds stay honest where long double is no wider than double.
using Real = long double;
constexpr Real kUnitRoundoff = std::numeric_limits<Real>::epsilon() / 2;

/// Neumaier compensated sum.
class Sum {
 public:
  void add(Real v) {
    const Real t = s_ + v;
    c_ += std::fabs(s_) >= std::fabs(v) ? (s_ - t) + v : (v - t) + s_;
    s_ = t;
  }
  /// The product is formed in Real: a product of two doubles is not rounded to double first.
  void add_product(Real a, Real b) { add(a * b); }
  Real value() const { return s_ + c_; }

 private:
  Real s_ = 0.0, c_ = 0.0;
};

struct Pass {
  bool finite = false;
  int dropped = 0;
  Real drop_effect = 0.0;
  Real lb_min = 0.0;       // minimization form, before the rounding allowance
  Real allowance = 0.0;
  std::string note;
};

}  // namespace

DualBound compute_dual_bound(const LpModel& m, const std::vector<double>& x, const std::vector<double>& y, double primal_objective,
                             double gap_tol, double zero_tol) {
  DualBound out;
  const std::size_t rows = static_cast<std::size_t>(m.n_rows), cols = static_cast<std::size_t>(m.n_cols);
  if (y.size() != rows || x.size() != cols) {
    out.note = "multiplier or point has the wrong length";
    return out;
  }
  const Real sgn = m.sense == Sense::Maximize ? -1.0L : 1.0L;

  // Quadratic term (convex Q, minimization form Q' = sgn Q): for ANY x~ the first-order underestimate
  // (1/2) x^T Q' x >= x~^T Q' x - (1/2) x~^T Q' x~ gives the bound with d = c + Q' x~ - A^T y and the constant
  // -(1/2) x~^T Q' x~; x~ is the supplied point. The bound is only valid when Q' is positive semidefinite, which the
  // caller establishes (check_convexity, or the exact test in KASAUTI).
  std::vector<Real> qx(cols, 0.0);
  Real quad_half = 0.0;
  if (m.quadratic.nnz() > 0 && m.quadratic.n_cols == m.n_cols) {
    for (std::size_t j = 0; j < cols; ++j) {
      for (Index p = m.quadratic.col_start[j]; p < m.quadratic.col_start[j + 1]; ++p) {
        const std::size_t i = static_cast<std::size_t>(m.quadratic.row_index[to_size(p)]);
        const Real q = sgn * static_cast<Real>(m.quadratic.value[to_size(p)]);
        qx[i] += q * x[j];
        if (i != j) qx[j] += q * x[i];
      }
    }
    for (std::size_t j = 0; j < cols; ++j) quad_half += 0.5L * x[j] * qx[j];
  }

  // d = c + Q x~ - A^T y per column with its scale and rounding margin; row activities for the drop effect.
  std::vector<Real> d(cols), scale(cols), margin(cols);
  for (std::size_t j = 0; j < cols; ++j) {
    Sum s;
    const Real c = sgn * m.col_cost[j];
    s.add(c);
    s.add(qx[j]);
    Real sc = std::fabs(c) + std::fabs(qx[j]);
    for (Index p = m.A.col_start[j]; p < m.A.col_start[j + 1]; ++p) {
      const Real a = m.A.value[to_size(p)], yv = y[to_size(m.A.row_index[to_size(p)])];
      s.add_product(-a, yv);
      sc += std::fabs(a * yv);
    }
    d[j] = s.value();
    scale[j] = sc;
    margin[j] = 4.0L * kUnitRoundoff * sc;  // bounds the rounding error of d_j; a computed zero is never taken as exact
  }
  std::vector<Real> act(rows, 0.0);
  for (std::size_t j = 0; j < cols; ++j) {
    for (Index p = m.A.col_start[j]; p < m.A.col_start[j + 1]; ++p) act[to_size(m.A.row_index[to_size(p)])] += static_cast<Real>(m.A.value[to_size(p)]) * x[j];
  }
  Real ymax = 0.0;
  for (const double v : y) ymax = std::max(ymax, static_cast<Real>(std::fabs(v)));

  auto run = [&](double drop) {
    Pass p;
    Sum lb;
    Real terms_abs = std::fabs(m.objective_offset) + std::fabs(quad_half);
    lb.add(sgn * m.objective_offset);
    lb.add(-quad_half);
    int offenders = 0;
    Real worst_rel = 0.0;
    auto offender = [&](Real mag, Real rel) {
      ++offenders;
      worst_rel = std::max(worst_rel, rel);
      (void)mag;
    };
    for (std::size_t i = 0; i < rows; ++i) {
      const double yi = y[i];
      if (yi == 0.0) continue;
      const double b = yi > 0.0 ? m.row_lower[i] : m.row_upper[i];
      if (is_inf(b)) {
        if (drop > 0.0 && std::fabs(yi) <= drop * ymax) {
          ++p.dropped;
          p.drop_effect += std::fabs(yi * act[i]);
        } else {
          offender(std::fabs(yi), std::fabs(yi) / ymax);
        }
      } else {
        lb.add_product(yi, b);
        terms_abs += std::fabs(yi * b);
      }
    }
    for (std::size_t j = 0; j < cols; ++j) {
      const Real dj = d[j];
      // The sign of d_j is established only beyond the rounding margin; inside it the needed side is
      // whichever bound is missing.
      const bool lo_missing = is_inf(m.col_lower[j]), hi_missing = is_inf(m.col_upper[j]);
      const bool needs_missing = (dj > margin[j] && lo_missing) || (dj < -margin[j] && hi_missing) ||
                                 (std::fabs(dj) <= margin[j] && margin[j] > 0.0 && (lo_missing || hi_missing));
      if (needs_missing) {
        const Real mag = std::fabs(dj);
        if (drop > 0.0 && mag <= drop * scale[j]) {
          ++p.dropped;
          p.drop_effect += mag * std::fabs(x[j]) + margin[j] * std::fabs(x[j]);
        } else if (drop == 0.0 && std::fabs(dj) <= margin[j]) {
          // Strict pass: the sign cannot be established, so the strict bound is not available.
          ++p.dropped;
          p.drop_effect += margin[j] * std::fabs(x[j]);
          p.finite = false;
          p.note = "a reduced cost is within its rounding margin of zero at an infinite bound";
          p.lb_min = 0.0;
          return p;
        } else {
          offender(mag, mag / scale[j]);
        }
        continue;
      }
      if (dj == 0.0) continue;
      const double b = dj > 0.0 ? m.col_lower[j] : m.col_upper[j];
      if (is_inf(b)) continue;  // |dj| within margin with both bounds finite cannot reach here; keeps the sum defined
      lb.add_product(dj, b);
      terms_abs += std::fabs(dj * b);
      p.allowance += margin[j] * std::fabs(b);
    }
    if (offenders > 0) {
      std::ostringstream os;
      os << offenders << " multiplier(s)/reduced cost(s) need an infinite bound (largest relative size " << worst_rel << ")";
      p.note = os.str();
      return p;
    }
    p.finite = true;
    p.lb_min = lb.value();
    p.allowance += 4.0 * kUnitRoundoff * terms_abs;
    return p;
  };

  Pass strict = run(0.0);
  Pass used = strict;
  if (!strict.finite && zero_tol > 0.0) {
    Pass tol = run(zero_tol);
    if (tol.finite) used = tol;
    else used.note = tol.note;
  }
  out.finite = used.finite;
  out.rigorous = used.finite && used.dropped == 0;
  out.dropped = used.finite ? used.dropped : 0;
  out.drop_effect = used.finite ? static_cast<double>(used.drop_effect) : 0.0;
  if (used.finite && used.dropped > 0) out.note = "tolerance-level: dropped multipliers/reduced costs below the relative zero tolerance";
  if (!used.finite) {
    out.note = used.note.empty() ? strict.note : used.note;
    return out;
  }
  out.allowance = static_cast<double>(used.allowance);
  const Real safe_min = used.lb_min - used.allowance;
  out.bound = static_cast<double>(sgn * safe_min);
  // The acceptance gap uses the nominal bound (as the exact verifier does with exact reduced costs); the
  // rounding allowance only makes the reported bound conservative.
  const Real gap = sgn * primal_objective - used.lb_min;
  out.gap_rel = static_cast<double>(std::fabs(gap) / (1.0L + std::fabs(static_cast<Real>(primal_objective))));
  // A primal point that beats the bound (negative gap) is consistent only if its own constraint violations,
  // weighted by the multipliers, explain it: obj(x) >= LB(y) - sum |y_i| viol_i - sum |d_j| viol_j.
  Real explained = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    const Real v = std::max({Real(0), static_cast<Real>(m.row_lower[i]) - act[i], act[i] - static_cast<Real>(m.row_upper[i])});
    if (v > 0.0) explained += std::fabs(y[i]) * v;
  }
  for (std::size_t j = 0; j < cols; ++j) {
    const Real v = std::max({Real(0), static_cast<Real>(m.col_lower[j]) - x[j], x[j] - static_cast<Real>(m.col_upper[j])});
    if (v > 0.0) explained += std::fabs(d[j]) * v;
  }
  out.explained = static_cast<double>(explained);
  const Real obj_scale = 1.0L + std::fabs(static_cast<Real>(primal_objective));
  out.gap_ok = gap <= gap_tol * obj_scale && gap >= -(explained + gap_tol * obj_scale);
  return out;
}

}  // namespace shodhan
