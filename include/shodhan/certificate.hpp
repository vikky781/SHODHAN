#pragma once

#include <iosfwd>
#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/lp_solver.hpp"
#include "shodhan/mip/result.hpp"

namespace shodhan {

/// Facts about the problem and the run that go into a certificate.
struct CertificateContext {
  std::string solver_name = "shodhan";
  std::string solver_version;
  /// Name of the model (the NAME line) and SHA-256 of the exact bytes of the model file.
  std::string problem_name;
  std::string file_sha256;
  LpOptions options;  ///< tolerances recorded in the certificate
  /// MILP runs: the integrality tolerance and the gaps the search was allowed to stop at (recorded only).
  double int_tol = 1e-5;
  double mip_gap = 0.0;
  double mip_abs_gap = 0.0;
};

/// Writes a "shodhan-cert" version 1 JSON certificate (docs/CERTIFICATES.md) for the
/// result of LpSolver::solve on `model`. Rows and columns are identified by their
/// names (model.row_names / col_names; R<i> and C<j> with 1-based i, j are used when the
/// model carries no names). Optimal, Infeasible and Unbounded results get their bodies;
/// anything else (limits, NumericalError, a missing feasible point for Unbounded) is
/// written with status "other" and no body.
void write_certificate(const LpModel& model, const CertificateContext& context, const LpResult& result, std::ostream& out);

/// Same, to a file; returns false and sets *error if it cannot be written.
bool write_certificate_file(const LpModel& model, const CertificateContext& context, const LpResult& result,
                            const std::string& path, std::string* error);

/// The status string a certificate for `result` carries ("optimal", "infeasible", "unbounded" or "other").
std::string certificate_status(const LpResult& result);

/// MILP results. A verified incumbent (also after a time or node limit) is written as status "feasible":
/// `x` with exactly integral integer columns, `claimed_objective`, and, when known, `claimed_best_bound`,
/// `claimed_gap` (relative) and `claimed_gap_abs`, the node count, the solver's own status as `mip_status`, and
/// the explicit field `optimality_certified: false`: no optimality proof is part of a MILP certificate, and a
/// verifier reports the bound as NOT verified. Infeasible is written as status "infeasible": with the Farkas
/// multipliers when the LP relaxation itself is infeasible, otherwise with `certified: false` and no body
/// (the infeasibility was proved by branching or presolve). Everything else is status "other".
void write_mip_certificate(const LpModel& model, const CertificateContext& context, const mip::MipResult& result, std::ostream& out);
bool write_mip_certificate_file(const LpModel& model, const CertificateContext& context, const mip::MipResult& result,
                                const std::string& path, std::string* error);
std::string mip_certificate_status(const mip::MipResult& result);

}  // namespace shodhan
