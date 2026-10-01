#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan::cli {

/// Prints the model summary shown by `shodhan info` and `shodhan solve`.
void print_model_summary(const LpModel& model, const std::vector<std::string>& warnings,
                         std::ostream& out);

}  // namespace shodhan::cli
