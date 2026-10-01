#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "info.hpp"
#include "shodhan/mps.hpp"
#include "shodhan/status.hpp"
#include "shodhan/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitNotImplemented = 2;

void print_usage(std::ostream& out) {
  out << "SHODHAN " << shodhan::kVersion << " - a from-scratch LP/MILP/QP optimization solver core\n"
      << "\n"
      << "Usage:\n"
      << "  shodhan info <file>     print a summary of the model in an MPS file\n"
      << "  shodhan solve <file>    read the model and solve it (not implemented yet)\n"
      << "  shodhan --help          show this help\n"
      << "  shodhan --version       show the version\n"
      << "\n"
      << "Exit codes: 0 ok, 1 usage or read error, 2 not implemented.\n";
}

// Reads the model; on failure prints the error and returns false.
bool load(const std::string& path, shodhan::MpsReadResult* result) {
  *result = shodhan::read_mps_file(path);
  if (!result->ok) {
    std::cerr << "error: " << result->error << "\n";
    std::cerr << "status: " << shodhan::to_string(shodhan::Status::ReadError) << "\n";
    return false;
  }
  return true;
}

int run(const std::vector<std::string>& args) {
  if (args.empty()) {
    print_usage(std::cerr);
    return kExitUsage;
  }
  const std::string& cmd = args[0];
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    print_usage(std::cout);
    return kExitOk;
  }
  if (cmd == "--version" || cmd == "-V") {
    std::cout << "shodhan " << shodhan::kVersion << "\n";
    return kExitOk;
  }
  if (cmd == "info" || cmd == "solve") {
    if (args.size() != 2) {
      std::cerr << "error: '" << cmd << "' takes exactly one file argument\n\n";
      print_usage(std::cerr);
      return kExitUsage;
    }
    shodhan::MpsReadResult result;
    if (!load(args[1], &result)) return kExitUsage;
    shodhan::cli::print_model_summary(result.model, result.warnings, std::cout);
    if (cmd == "info") return kExitOk;
    std::cout << "\nStatus: " << shodhan::to_string(shodhan::Status::NotImplemented)
              << " (the solver is not implemented yet; no solution is produced)\n";
    return kExitNotImplemented;
  }
  std::cerr << "error: unknown command '" << cmd << "'\n\n";
  print_usage(std::cerr);
  return kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return run(args);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitUsage;
  }
}
