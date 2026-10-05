#include "pool_command.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "shodhan/pooling.hpp"
#include "shodhan/sha256.hpp"

namespace shodhan::cli {

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitFailure = 3;
constexpr int kExitLimit = 4;

int usage_error(const std::string& message) {
  std::cerr << "error: " << message << "\n"
            << "usage: shodhan pool <spec.pool> [--method recursion|slp] [--starts N] [--tol t] [--seed s]\n"
               "                    [--damping a] [--max-iter n] [--no-bound] [--write-sol path] [--write-cert path] [--verbose] [--dump]\n";
  return kExitUsage;
}

bool parse_number(const std::string& s, double* out) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(v)) return false;
  *out = v;
  return true;
}

std::string num(double v) {
  if (v >= 1e29) return "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.10g", v);
  return buf;
}

}  // namespace

int run_pool(const std::vector<std::string>& args) {
  std::string path, sol_path, cert_path;
  pooling::PoolOptions opt;
  bool verbose = false, dump = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    double v = 0.0;
    auto number_next = [&]() {
      if (i + 1 >= args.size() || !parse_number(args[i + 1], &v)) return false;
      ++i;
      return true;
    };
    if (a == "--method") {
      if (i + 1 >= args.size()) return usage_error("--method needs recursion or slp");
      const std::string m = args[++i];
      if (m == "recursion") opt.method = pooling::PoolMethod::Recursion;
      else if (m == "slp") opt.method = pooling::PoolMethod::Slp;
      else return usage_error("--method needs recursion or slp");
    } else if (a == "--starts") {
      if (!number_next() || v < 1 || v != std::floor(v)) return usage_error("--starts needs a positive integer");
      opt.starts = static_cast<int>(v);
    } else if (a == "--tol") {
      if (!number_next() || v <= 0) return usage_error("--tol needs a positive number");
      opt.tol = v;
    } else if (a == "--seed") {
      if (!number_next() || v < 0 || v != std::floor(v)) return usage_error("--seed needs a non-negative integer");
      opt.seed = static_cast<unsigned long long>(v);
    } else if (a == "--damping") {
      if (!number_next() || v <= 0 || v > 1) return usage_error("--damping needs a number in (0, 1]");
      opt.damping = v;
    } else if (a == "--max-iter") {
      if (!number_next() || v < 1 || v != std::floor(v)) return usage_error("--max-iter needs a positive integer");
      opt.max_iterations = static_cast<int>(v);
    } else if (a == "--no-bound") {
      opt.mccormick = false;
    } else if (a == "--verbose") {
      verbose = true;
    } else if (a == "--dump") {
      dump = true;
    } else if (a == "--write-sol") {
      if (i + 1 >= args.size()) return usage_error("--write-sol needs a path");
      sol_path = args[++i];
    } else if (a == "--write-cert") {
      if (i + 1 >= args.size()) return usage_error("--write-cert needs a path");
      cert_path = args[++i];
    } else if (!a.empty() && a[0] == '-') {
      return usage_error("unknown option '" + a + "'");
    } else if (path.empty()) {
      path = a;
    } else {
      return usage_error("more than one file given");
    }
  }
  if (path.empty()) return usage_error("no .pool file given");
  const pooling::PoolParseResult read = pooling::parse_pool_file(path);
  if (!read.ok) {
    std::cerr << "error: " << read.error << "\n";
    return kExitUsage;
  }
  const pooling::PoolProblem& p = read.problem;
  if (dump) {  // canonical text of the parsed problem, to compare the parsers (the verifier has its own)
    auto d = [](double v) {
      if (v >= 1e29) return std::string("inf");
      char b[64];
      std::snprintf(b, sizeof b, "%.17g", v);
      return std::string(b);
    };
    std::cout << "qualities " << p.n_qualities << "\n";
    for (const pooling::Source& s : p.sources) {
      std::cout << "source " << s.name << " " << d(s.cost) << " " << d(s.supply);
      for (const double q : s.quality) std::cout << " " << d(q);
      std::cout << "\n";
    }
    for (const pooling::Pool& q : p.pools) std::cout << "pool " << q.name << " " << d(q.capacity) << "\n";
    for (const pooling::Terminal& t : p.terminals) {
      std::cout << "terminal " << t.name << " " << d(t.price) << " " << d(t.demand);
      for (const double q : t.spec) std::cout << " " << d(q);
      std::cout << "\n";
    }
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) std::cout << "arc " << p.arc_name_sp(a) << "\n";
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) std::cout << "arc " << p.arc_name_pt(a) << "\n";
    for (std::size_t a = 0; a < p.arcs_st.size(); ++a) std::cout << "arc " << p.arc_name_st(a) << "\n";
    return kExitOk;
  }
  if (verbose) opt.log = &std::cout;
  std::cout << "Pooling problem: " << p.name << "\n";
  if (p.synthetic) std::cout << "SYNTHETIC:     structure follows textbook formulations; not plant or MRPL data\n";
  std::cout << "Network:       " << p.sources.size() << " source(s), " << p.pools.size() << " pool(s), " << p.terminals.size() << " terminal(s), " << p.n_qualities
            << " quality(ies), " << p.n_flows() << " flow variable(s)\n";
  std::cout << "Method:        "
            << (opt.method == pooling::PoolMethod::Slp ? std::string("sequential LP with a trust region") : "distributive recursion (damping " + num(opt.damping) + ")") << ", "
            << opt.starts << " start(s), tolerance " << num(opt.tol) << "\n";
  const pooling::PoolResult r = pooling::solve_pool(p, opt);
  std::cout << "Status:        " << pooling::to_string(r.status) << "\n";
  const bool have = !r.point.flow.empty() && r.check.ok(opt.tol);
  if (have) {
    if (r.status == pooling::PoolStatus::Converged) {
      std::cout << "Local value:   " << num(r.objective) << " (revenue minus cost; a LOCAL solution)\n";
    } else {
      std::cout << "Best point:    " << num(r.objective) << " (revenue minus cost; feasible within the tolerance, but NOT a converged solution)\n";
    }
    if (r.has_bound) {
      std::cout << "McCormick bound: " << num(r.mccormick_bound) << " (upper bound for the maximum; gap " << num(r.gap) << ")\n";
      std::cout << (r.gap <= opt.tol ? "Global:        the gap is within the tolerance, so the local value is global up to that tolerance and the accuracy of the bound\n"
                                     : "Global:        NO global-optimality guarantee: the gap is above the tolerance, a better solution may exist\n");
    } else {
      std::cout << "Global:        NO global-optimality guarantee (no McCormick bound)\n";
    }
    std::cout << "Check:         original nonlinear model: worst relative residual " << num(r.check.worst) << " (tolerance " << num(opt.tol) << "); bounds "
              << num(r.check.bounds) << ", supply " << num(r.check.supply) << ", capacity " << num(r.check.capacity) << ", demand " << num(r.check.demand) << ", material "
              << num(r.check.material) << ", quality balance " << num(r.check.quality_balance) << ", terminal quality " << num(r.check.terminal_spec) << "\n";
  } else {
    std::cout << "No verified solution is reported.\n";
  }
  std::cout << "Starts:        " << r.starts_run << " run, " << r.starts_converged << " converged, " << r.starts_cycled << " cycled, " << r.starts_failed << " other\n";
  if (!r.outcomes.empty()) {
    std::cout << "Outcomes:      ";
    for (const auto& o : r.outcomes) std::cout << num(o.first) << " x" << o.second << "  ";
    std::cout << "\n";
  }
  if (!r.message.empty()) std::cout << "Message:       " << r.message << "\n";
  if (have && !sol_path.empty()) {
    std::ofstream f(sol_path);
    if (!f) {
      std::cerr << "error: cannot write '" << sol_path << "'\n";
      return kExitUsage;
    }
    f.precision(17);
    f << "objective " << r.objective << "\n";
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) f << p.arc_name_sp(a) << " " << r.point.flow[p.idx_sp(a)] << "\n";
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) f << p.arc_name_pt(a) << " " << r.point.flow[p.idx_pt(a)] << "\n";
    for (std::size_t a = 0; a < p.arcs_st.size(); ++a) f << p.arc_name_st(a) << " " << r.point.flow[p.idx_st(a)] << "\n";
    for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
      for (int k = 0; k < p.n_qualities; ++k) f << p.pools[pool].name << ":" << p.quality_names[static_cast<std::size_t>(k)] << " " << r.point.q[p.qidx(pool, k)] << "\n";
    }
    std::cout << "Solution:      written to " << sol_path << "\n";
  }
  if (!cert_path.empty()) {
    std::string sha;
    if (!sha256_file_hex(path, &sha)) {
      std::cerr << "error: cannot read '" << path << "' to hash it\n";
      return kExitUsage;
    }
    std::ofstream f(cert_path, std::ios::binary);
    if (!f) {
      std::cerr << "error: cannot write '" << cert_path << "'\n";
      return kExitUsage;
    }
    const std::string st = pooling::write_pool_certificate(p, r, opt, sha, f);
    std::cout << "Certificate:   " << st << " written to " << cert_path << " (feasibility within the stated tolerance only: optimality is not certified)\n";
    std::cout << "Verify with:   python -m kasauti " << path << " " << cert_path << "   (from the verify directory, or with PYTHONPATH=verify)\n";
  }
  if (r.status == pooling::PoolStatus::Converged) return kExitOk;
  // A verified point without convergence (cycling, iteration limit) is a limit outcome; no point at all is a failure.
  return have ? kExitLimit : kExitFailure;
}

}  // namespace shodhan::cli
