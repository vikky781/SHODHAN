#pragma once

#include <iosfwd>
#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/lp_solver.hpp"

namespace shodhan {

/// Facts about the problem and the run that go into a certificate.
struct CertificateContext {
  std::string solver_name = "shodhan";
  std::string solver_version;
  /// Name of the model (the NAME line) and SHA-256 of the exact bytes of the model file.
  std::string problem_name;
  std::string file_sha256;
  LpOptions options;  ///< tolerances recorded in the certificate
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

}  // namespace shodhan
