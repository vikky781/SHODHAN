#include <ostream>

#include "shodhan/json_writer.hpp"
#include "shodhan/pooling.hpp"
#include "shodhan/version.hpp"

namespace shodhan::pooling {

std::string write_pool_certificate(const PoolProblem& p, const PoolResult& r, const PoolOptions& options, const std::string& file_sha256,
                                   std::ostream& out) {
  const bool have = !r.point.flow.empty() && r.check.ok(options.tol);
  const std::string status = have ? "feasible" : "other";
  JsonWriter w(out);
  w.begin_object();
  w.key("format");
  w.value("shodhan-cert");
  w.key("version");
  w.value(1);
  w.key("solver");
  w.begin_object();
  w.key("name");
  w.value("shodhan");
  w.key("version");
  w.value(kVersion);
  w.end_object();
  w.key("problem");
  w.begin_object();
  w.key("name");
  w.value(p.name);
  w.key("file_sha256");
  w.value(file_sha256);
  w.key("kind");
  w.value("pooling");
  w.key("sources");
  w.value(static_cast<long long>(p.sources.size()));
  w.key("pools");
  w.value(static_cast<long long>(p.pools.size()));
  w.key("terminals");
  w.value(static_cast<long long>(p.terminals.size()));
  w.key("qualities");
  w.value(p.n_qualities);
  w.key("synthetic");
  w.value(p.synthetic);
  w.end_object();
  w.key("status");
  w.value(status);
  if (have) {
    w.key("claimed_objective");
    w.value(r.objective);
    if (r.has_bound) {
      w.key("claimed_upper_bound");
      w.value(r.mccormick_bound);
    }
    w.key("converged");
    w.value(r.status == PoolStatus::Converged);
    w.key("optimality_certified");
    w.value(false);
    w.key("nonconvex");
    w.value(true);
    w.key("flows");
    w.begin_object();
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
      w.key(p.arc_name_sp(a));
      w.value(r.point.flow[p.idx_sp(a)]);
    }
    for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
      w.key(p.arc_name_pt(a));
      w.value(r.point.flow[p.idx_pt(a)]);
    }
    for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
      w.key(p.arc_name_st(a));
      w.value(r.point.flow[p.idx_st(a)]);
    }
    w.end_object();
    w.key("q");
    w.begin_object();
    for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
      for (int k = 0; k < p.n_qualities; ++k) {
        w.key(p.pools[pool].name + ":" + p.quality_names[static_cast<std::size_t>(k)]);
        w.value(r.point.q[p.qidx(pool, k)]);
      }
    }
    w.end_object();
  }
  w.key("tolerances");
  w.begin_object();
  w.key("pool_tol");
  w.value(options.tol);
  w.end_object();
  w.key("attempts");
  w.begin_object();
  w.key("count");
  w.value(r.starts_run);
  w.key("configuration");
  w.value(options.method == PoolMethod::Slp ? "slp" : "recursion");
  w.end_object();
  w.end_object();
  out << "\n";
  return status;
}

}  // namespace shodhan::pooling
