#include "cert_output.hpp"

#include <iostream>

#include "shodhan/certificate.hpp"
#include "shodhan/sha256.hpp"
#include "shodhan/version.hpp"

namespace shodhan::cli {

bool write_certificate_output(const LpModel& model, const std::string& model_path, const LpOptions& options,
                              const LpResult& result, const std::string& cert_path) {
  CertificateContext ctx;
  ctx.solver_version = kVersion;
  ctx.problem_name = model.name;
  ctx.options = options;
  if (!sha256_file_hex(model_path, &ctx.file_sha256)) {
    std::cerr << "error: cannot read '" << model_path << "' to hash it\n";
    return false;
  }
  std::string error;
  if (!write_certificate_file(model, ctx, result, cert_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return false;
  }
  std::cout << "Certificate:   " << certificate_status(result) << " written to " << cert_path << "\n";
  std::cout << "Verify with:   python -m kasauti " << model_path << " " << cert_path << "   (from the verify directory, or with PYTHONPATH=verify)\n";
  return true;
}

bool write_mip_certificate_output(const LpModel& model, const std::string& model_path, const mip::MipOptions& options,
                                  const mip::MipResult& result, const std::string& cert_path) {
  CertificateContext ctx;
  ctx.solver_version = kVersion;
  ctx.problem_name = model.name;
  ctx.options.params = options.params;
  ctx.int_tol = options.params.int_tol;
  ctx.mip_gap = options.mip_gap;
  ctx.mip_abs_gap = options.mip_abs_gap;
  if (!sha256_file_hex(model_path, &ctx.file_sha256)) {
    std::cerr << "error: cannot read '" << model_path << "' to hash it\n";
    return false;
  }
  std::string error;
  if (!write_mip_certificate_file(model, ctx, result, cert_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return false;
  }
  std::cout << "Certificate:   " << mip_certificate_status(result) << " written to " << cert_path;
  if (mip_certificate_status(result) == "feasible") std::cout << " (feasibility, integrality and objective only: optimality is not certified)";
  if (mip_certificate_status(result) == "infeasible" && !result.lp_infeasible_certified) std::cout << " (no certificate: proved by branching or presolve)";
  std::cout << "\n";
  std::cout << "Verify with:   python -m kasauti " << model_path << " " << cert_path << "   (from the verify directory, or with PYTHONPATH=verify)\n";
  return true;
}

}  // namespace shodhan::cli
