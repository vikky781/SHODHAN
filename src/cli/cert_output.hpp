#pragma once

#include <string>

#include "shodhan/lp_model.hpp"
#include "shodhan/lp_solver.hpp"

namespace shodhan::cli {

/// Writes a certificate for `result` to cert_path and tells the user how to verify it.
/// Returns false (after printing an error) if the model file cannot be hashed or the
/// certificate cannot be written.
bool write_certificate_output(const LpModel& model, const std::string& model_path, const LpOptions& options,
                              const LpResult& result, const std::string& cert_path);

}  // namespace shodhan::cli
