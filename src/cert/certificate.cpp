#include "shodhan/certificate.hpp"

#include <fstream>
#include <ostream>

#include "shodhan/json_writer.hpp"

namespace shodhan {

namespace {

std::string row_name(const LpModel& m, Index i) {
  return m.row_names.empty() ? "R" + std::to_string(i + 1) : m.row_names[to_size(i)];
}

std::string col_name(const LpModel& m, Index j) {
  return m.col_names.empty() ? "C" + std::to_string(j + 1) : m.col_names[to_size(j)];
}

template <typename NameFn>
void write_sparse(JsonWriter& w, const std::vector<double>& v, NameFn name) {
  w.begin_object();
  for (std::size_t k = 0; k < v.size(); ++k) {
    if (v[k] != 0.0) {
      w.key(name(static_cast<Index>(k)));
      w.value(v[k]);
    }
  }
  w.end_object();
}

}  // namespace

std::string certificate_status(const LpResult& r) {
  switch (r.status) {
    case Status::Optimal: return "optimal";
    case Status::Infeasible: return r.farkas_ray.empty() ? "other" : "infeasible";
    case Status::Unbounded: return r.unbounded_ray.empty() || r.unbounded_point.empty() ? "other" : "unbounded";
    default: return "other";
  }
}

// The members every certificate starts with, up to and including "status". The object stays open.
static void write_head(JsonWriter& w, const LpModel& model, const CertificateContext& ctx, const std::string& status) {
  std::size_t n_integer = 0;
  for (const ColType t : model.col_type) n_integer += t != ColType::Continuous ? 1 : 0;
  w.begin_object();
  w.key("format");
  w.value("shodhan-cert");
  w.key("version");
  w.value(1);
  w.key("solver");
  w.begin_object();
  w.key("name");
  w.value(ctx.solver_name);
  w.key("version");
  w.value(ctx.solver_version);
  w.end_object();
  w.key("problem");
  w.begin_object();
  w.key("name");
  w.value(ctx.problem_name);
  w.key("file_sha256");
  w.value(ctx.file_sha256);
  w.key("rows");
  w.value(static_cast<long long>(model.n_rows));
  w.key("cols");
  w.value(static_cast<long long>(model.n_cols));
  w.key("nnz");
  w.value(static_cast<unsigned long long>(model.A.nnz()));
  w.key("sense");
  w.value(model.sense == Sense::Maximize ? "max" : "min");
  w.key("n_integer");
  w.value(static_cast<unsigned long long>(n_integer));
  w.key("quadratic");
  w.value(model.quadratic.nnz() > 0);
  w.key("q_nnz");
  w.value(static_cast<unsigned long long>(model.quadratic.nnz()));
  w.end_object();
  w.key("status");
  w.value(status);
}

void write_certificate(const LpModel& model, const CertificateContext& ctx, const LpResult& r, std::ostream& out) {
  const std::string status = certificate_status(r);
  JsonWriter w(out);
  write_head(w, model, ctx, status);
  if (status == "optimal") {
    w.key("claimed_objective");
    w.value(r.solution.objective);
    if (r.quadratic) {
      // How the solver established convexity of Q: a claim, not evidence (KASAUTI tests it exactly).
      w.key("convexity");
      w.value(r.convexity);
    }
    // The pipeline's own weak-duality bound of the multipliers: rigorous only if no tiny multiplier had to be
    // treated as zero. A claim, not evidence: the verifier recomputes everything.
    w.key("dual_bound");
    w.begin_object();
    w.key("rigorous");
    w.value(r.rigorous);
    w.key("available");
    w.value(r.dual_bound.finite);
    if (r.dual_bound.finite) {
      w.key("value");
      w.value(r.dual_bound.bound);
      w.key("dropped");
      w.value(static_cast<long long>(r.dual_bound.dropped));
      w.key("gap_rel");
      w.value(r.dual_bound.gap_rel);
    }
    w.end_object();
  }
  w.key("tolerances");
  w.begin_object();
  w.key("primal_tol");
  w.value(ctx.options.params.primal_tol);
  w.key("dual_tol");
  w.value(ctx.options.params.dual_tol);
  w.key("kkt_tol");
  w.value(ctx.options.kkt_tol);
  w.end_object();
  w.key("attempts");
  w.begin_object();
  w.key("count");
  w.value(r.attempts);
  w.key("configuration");
  w.value(r.configuration);
  w.end_object();
  if (status == "optimal") {
    w.key("x");
    write_sparse(w, r.solution.x, [&](Index j) { return col_name(model, j); });
    w.key("y");
    write_sparse(w, r.solution.y, [&](Index i) { return row_name(model, i); });
  } else if (status == "infeasible") {
    w.key("farkas");
    w.begin_object();
    w.key("y");
    write_sparse(w, r.farkas_ray, [&](Index i) { return row_name(model, i); });
    w.end_object();
  } else if (status == "unbounded") {
    w.key("point");
    write_sparse(w, r.unbounded_point, [&](Index j) { return col_name(model, j); });
    w.key("ray");
    write_sparse(w, r.unbounded_ray, [&](Index j) { return col_name(model, j); });
  }
  w.end_object();
  out << '\n';
}

bool write_certificate_file(const LpModel& model, const CertificateContext& ctx, const LpResult& r, const std::string& path,
                            std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    if (error != nullptr) *error = "cannot open '" + path + "' for writing";
    return false;
  }
  write_certificate(model, ctx, r, out);
  out.close();
  if (!out) {
    if (error != nullptr) *error = "error while writing '" + path + "'";
    return false;
  }
  return true;
}

std::string mip_certificate_status(const mip::MipResult& r) {
  if (r.has_solution && !r.solution.x.empty()) return "feasible";
  if (r.status == Status::Infeasible) return "infeasible";
  return "other";
}

void write_mip_certificate(const LpModel& model, const CertificateContext& ctx, const mip::MipResult& r, std::ostream& out) {
  const std::string status = mip_certificate_status(r);
  JsonWriter w(out);
  write_head(w, model, ctx, status);
  w.key("tolerances");
  w.begin_object();
  w.key("primal_tol");
  w.value(ctx.options.params.primal_tol);
  w.key("int_tol");
  w.value(ctx.int_tol);
  w.key("mip_gap");
  w.value(ctx.mip_gap);
  w.key("mip_abs_gap");
  w.value(ctx.mip_abs_gap);
  w.end_object();
  w.key("attempts");
  w.begin_object();
  w.key("count");
  w.value(1);
  w.key("configuration");
  w.value("branch-and-bound");
  w.end_object();
  w.key("mip_status");
  w.value(std::string(to_string(r.status)));
  w.key("nodes");
  w.value(r.nodes_processed);
  w.key("optimality_certified");
  w.value(false);
  if (status == "feasible") {
    w.key("claimed_objective");
    w.value(r.objective);
    if (r.has_bound) {
      w.key("claimed_best_bound");
      w.value(r.best_bound);
      w.key("claimed_gap");
      w.value(r.rel_gap);
      w.key("claimed_gap_abs");
      w.value(r.abs_gap);
    }
    w.key("x");
    write_sparse(w, r.solution.x, [&](Index j) { return col_name(model, j); });
  } else if (status == "infeasible") {
    if (r.lp_infeasible_certified && !r.lp_farkas.empty()) {
      w.key("certified");
      w.value(true);
      w.key("farkas");
      w.begin_object();
      w.key("y");
      write_sparse(w, r.lp_farkas, [&](Index i) { return row_name(model, i); });
      w.end_object();
    } else {
      w.key("certified");
      w.value(false);
      w.key("note");
      w.value("infeasibility was proved by branching or presolve: there is no certificate");
    }
  }
  w.end_object();
  out << '\n';
}

bool write_mip_certificate_file(const LpModel& model, const CertificateContext& ctx, const mip::MipResult& r, const std::string& path,
                                std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    if (error != nullptr) *error = "cannot open '" + path + "' for writing";
    return false;
  }
  write_mip_certificate(model, ctx, r, out);
  out.close();
  if (!out) {
    if (error != nullptr) *error = "error while writing '" + path + "'";
    return false;
  }
  return true;
}

}  // namespace shodhan
