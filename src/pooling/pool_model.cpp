#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "shodhan/pooling.hpp"

namespace shodhan::pooling {

std::string PoolProblem::arc_name_sp(std::size_t a) const { return sources[to_size(arcs_sp[a].first)].name + ">" + pools[to_size(arcs_sp[a].second)].name; }
std::string PoolProblem::arc_name_pt(std::size_t a) const { return pools[to_size(arcs_pt[a].first)].name + ">" + terminals[to_size(arcs_pt[a].second)].name; }
std::string PoolProblem::arc_name_st(std::size_t a) const { return sources[to_size(arcs_st[a].first)].name + ">" + terminals[to_size(arcs_st[a].second)].name; }

const char* to_string(PoolStatus s) noexcept {
  switch (s) {
    case PoolStatus::Converged: return "Converged";
    case PoolStatus::NoSolution: return "NoSolution";
    case PoolStatus::Cycling: return "Cycling";
    case PoolStatus::IterationLimit: return "IterationLimit";
    case PoolStatus::NumericalError: return "NumericalError";
  }
  return "?";
}

namespace {

std::vector<std::string> tokens_of(const std::string& line) {
  std::vector<std::string> t;
  std::istringstream is(line);
  std::string w;
  while (is >> w) {
    if (w[0] == '#') break;  // a comment runs to the end of the line
    t.push_back(w);
  }
  return t;
}

bool parse_number(const std::string& s, double* out) {
  if (s == "inf" || s == "+inf") {
    *out = kInf;
    return true;
  }
  if (s.empty()) return false;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end != s.c_str() + s.size() || !std::isfinite(v)) return false;
  *out = v;
  return true;
}

}  // namespace

PoolParseResult parse_pool_string(const std::string& text, const std::string& source_name) {
  PoolParseResult res;
  PoolProblem& p = res.problem;
  std::unordered_map<std::string, std::pair<char, int>> names;  // name -> (kind 's' 'p' 't', index)
  bool have_q = false;
  auto fail = [&](int line, const std::string& msg) {
    res.ok = false;
    res.error = source_name + ":" + std::to_string(line) + ": " + msg;
    return res;
  };
  std::istringstream in(text);
  std::string raw;
  int line = 0;
  while (std::getline(in, raw)) {
    ++line;
    const std::vector<std::string> t = tokens_of(raw);
    if (t.empty()) continue;
    const std::string& kw = t[0];
    if (kw == "name") {
      if (t.size() != 2) return fail(line, "'name' takes one token");
      p.name = t[1];
    } else if (kw == "synthetic") {
      if (t.size() != 2 || (t[1] != "yes" && t[1] != "no")) return fail(line, "'synthetic' takes yes or no");
      p.synthetic = t[1] == "yes";
    } else if (kw == "qualities") {
      if (have_q) return fail(line, "'qualities' given twice");
      double n = 0;
      if (t.size() < 2 || !parse_number(t[1], &n) || n < 1 || n != std::floor(n) || n > 16) return fail(line, "'qualities' needs a count from 1 to 16");
      p.n_qualities = static_cast<int>(n);
      if (t.size() > 2 && t.size() != 2 + to_size(p.n_qualities)) return fail(line, "'qualities' lists " + std::to_string(p.n_qualities) + " names or none");
      for (int k = 0; k < p.n_qualities; ++k) p.quality_names.push_back(t.size() > 2 ? t[2 + to_size(k)] : "Q" + std::to_string(k + 1));
      have_q = true;
    } else if (kw == "source" || kw == "pool" || kw == "terminal") {
      if (!have_q) return fail(line, "'qualities' must come before '" + kw + "'");
      const std::size_t K = to_size(p.n_qualities);
      const std::size_t need = kw == "pool" ? 3 : 4 + K;
      if (t.size() != need) {
        return fail(line, kw == "pool" ? "'pool' needs: pool NAME CAPACITY"
                          : kw == "source" ? "'source' needs: source NAME COST SUPPLY q1..qK"
                                           : "'terminal' needs: terminal NAME PRICE DEMAND Q1..QK");
      }
      if (names.count(t[1]) != 0) return fail(line, "duplicate name '" + t[1] + "'");
      if (t[1].find_first_of(">:\"") != std::string::npos) return fail(line, "a name must not contain '>', ':' or a quote: '" + t[1] + "'");
      std::vector<double> v(t.size() - 2);
      for (std::size_t i = 2; i < t.size(); ++i) {
        if (!parse_number(t[i], &v[i - 2])) return fail(line, "not a number: '" + t[i] + "'");
      }
      if (kw == "source") {
        Source s;
        s.name = t[1];
        s.cost = v[0];
        s.supply = v[1];
        if (is_inf(s.cost)) return fail(line, "a cost must be finite");
        if (s.supply < 0) return fail(line, "a supply must not be negative");
        for (std::size_t k = 0; k < K; ++k) {
          if (is_inf(v[2 + k])) return fail(line, "a source quality must be finite");
          s.quality.push_back(v[2 + k]);
        }
        names[t[1]] = {'s', static_cast<int>(p.sources.size())};
        p.sources.push_back(std::move(s));
      } else if (kw == "pool") {
        Pool q;
        q.name = t[1];
        q.capacity = v[0];
        if (q.capacity < 0) return fail(line, "a capacity must not be negative");
        names[t[1]] = {'p', static_cast<int>(p.pools.size())};
        p.pools.push_back(std::move(q));
      } else {
        Terminal m;
        m.name = t[1];
        m.price = v[0];
        m.demand = v[1];
        if (is_inf(m.price)) return fail(line, "a price must be finite");
        if (m.demand < 0) return fail(line, "a demand must not be negative");
        for (std::size_t k = 0; k < K; ++k) m.spec.push_back(v[2 + k]);
        names[t[1]] = {'t', static_cast<int>(p.terminals.size())};
        p.terminals.push_back(std::move(m));
      }
    } else if (kw == "arc") {
      if (t.size() != 3) return fail(line, "'arc' needs: arc FROM TO");
      const auto a = names.find(t[1]);
      const auto b = names.find(t[2]);
      if (a == names.end()) return fail(line, "unknown name '" + t[1] + "'");
      if (b == names.end()) return fail(line, "unknown name '" + t[2] + "'");
      const char ka = a->second.first, kb = b->second.first;
      std::vector<std::pair<int, int>>* list = nullptr;
      if (ka == 's' && kb == 'p') list = &p.arcs_sp;
      else if (ka == 'p' && kb == 't') list = &p.arcs_pt;
      else if (ka == 's' && kb == 't') list = &p.arcs_st;
      else return fail(line, "an arc goes from a source to a pool or a terminal, or from a pool to a terminal");
      const std::pair<int, int> arc{a->second.second, b->second.second};
      if (std::find(list->begin(), list->end(), arc) != list->end()) return fail(line, "duplicate arc " + t[1] + " " + t[2]);
      list->push_back(arc);
    } else {
      return fail(line, "unknown keyword '" + kw + "'");
    }
  }
  if (!have_q) return fail(line, "no 'qualities' line");
  if (p.sources.empty() || p.terminals.empty()) return fail(line, "at least one source and one terminal are required");
  if (p.arcs_sp.empty() && p.arcs_st.empty()) return fail(line, "no arc leaves a source");
  // Every pool must be connected on both sides, otherwise it is dead weight and its quality is undefined.
  for (std::size_t q = 0; q < p.pools.size(); ++q) {
    bool has_in = false, has_out = false;
    for (const auto& a : p.arcs_sp) has_in = has_in || to_size(a.second) == q;
    for (const auto& a : p.arcs_pt) has_out = has_out || to_size(a.first) == q;
    if (!has_in || !has_out) return fail(line, "pool '" + p.pools[q].name + "' needs an incoming and an outgoing arc");
  }
  if (p.name.empty()) p.name = source_name;
  res.ok = true;
  return res;
}

PoolParseResult parse_pool_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    PoolParseResult r;
    r.error = path + ": cannot open the file";
    return r;
  }
  std::ostringstream os;
  os << f.rdbuf();
  return parse_pool_string(os.str(), path);
}

double pool_objective(const PoolProblem& p, const std::vector<double>& flow) {
  double v = 0.0;
  for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) v -= p.sources[to_size(p.arcs_sp[a].first)].cost * flow[p.idx_sp(a)];
  for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) v += p.terminals[to_size(p.arcs_pt[a].second)].price * flow[p.idx_pt(a)];
  for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
    v += (p.terminals[to_size(p.arcs_st[a].second)].price - p.sources[to_size(p.arcs_st[a].first)].cost) * flow[p.idx_st(a)];
  }
  return v;
}

std::pair<double, double> pool_quality_range(const PoolProblem& p, std::size_t pool, int k) {
  double lo = kInf, hi = -kInf;
  for (const auto& a : p.arcs_sp) {
    if (to_size(a.second) != pool) continue;
    const double q = p.sources[to_size(a.first)].quality[to_size(k)];
    lo = std::min(lo, q);
    hi = std::max(hi, q);
  }
  if (lo > hi) {
    for (const Source& s : p.sources) {
      lo = std::min(lo, s.quality[to_size(k)]);
      hi = std::max(hi, s.quality[to_size(k)]);
    }
  }
  return {lo, hi};
}

std::vector<double> implied_pool_qualities(const PoolProblem& p, const std::vector<double>& flow, const std::vector<double>& previous) {
  std::vector<double> q = previous;
  for (std::size_t pool = 0; pool < p.pools.size(); ++pool) {
    double total = 0.0;
    for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
      if (to_size(p.arcs_sp[a].second) == pool) total += flow[p.idx_sp(a)];
    }
    if (total <= 1e-12) continue;
    for (int k = 0; k < p.n_qualities; ++k) {
      double num = 0.0;
      for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
        if (to_size(p.arcs_sp[a].second) == pool) num += p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)] * flow[p.idx_sp(a)];
      }
      q[p.qidx(pool, k)] = num / total;
    }
  }
  return q;
}

PoolCheck check_pool_point(const PoolProblem& p, const PoolPoint& pt) {
  PoolCheck c;
  const std::size_t nS = p.sources.size(), nP = p.pools.size(), nT = p.terminals.size();
  const std::vector<double>& f = pt.flow;
  auto rel = [](double residual, double scale) { return std::max(0.0, residual) / (1.0 + scale); };
  for (const double v : f) c.bounds = std::max(c.bounds, rel(-v, 0.0));
  for (std::size_t pool = 0; pool < nP; ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const auto r = pool_quality_range(p, pool, k);
      const double q = pt.q[p.qidx(pool, k)];
      c.bounds = std::max(c.bounds, rel(std::max(r.first - q, q - r.second), std::fabs(q)));
    }
  }
  std::vector<double> out_s(nS, 0.0), in_p(nP, 0.0), out_p(nP, 0.0), in_t(nT, 0.0);
  for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
    out_s[to_size(p.arcs_sp[a].first)] += f[p.idx_sp(a)];
    in_p[to_size(p.arcs_sp[a].second)] += f[p.idx_sp(a)];
  }
  for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
    out_p[to_size(p.arcs_pt[a].first)] += f[p.idx_pt(a)];
    in_t[to_size(p.arcs_pt[a].second)] += f[p.idx_pt(a)];
  }
  for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
    out_s[to_size(p.arcs_st[a].first)] += f[p.idx_st(a)];
    in_t[to_size(p.arcs_st[a].second)] += f[p.idx_st(a)];
  }
  for (std::size_t s = 0; s < nS; ++s) {
    if (!is_inf(p.sources[s].supply)) c.supply = std::max(c.supply, rel(out_s[s] - p.sources[s].supply, std::fabs(out_s[s]) + p.sources[s].supply));
  }
  for (std::size_t pool = 0; pool < nP; ++pool) {
    if (!is_inf(p.pools[pool].capacity)) c.capacity = std::max(c.capacity, rel(in_p[pool] - p.pools[pool].capacity, std::fabs(in_p[pool]) + p.pools[pool].capacity));
    c.material = std::max(c.material, rel(std::fabs(in_p[pool] - out_p[pool]), std::fabs(in_p[pool]) + std::fabs(out_p[pool])));
  }
  for (std::size_t t = 0; t < nT; ++t) {
    if (!is_inf(p.terminals[t].demand)) c.demand = std::max(c.demand, rel(in_t[t] - p.terminals[t].demand, std::fabs(in_t[t]) + p.terminals[t].demand));
  }
  for (std::size_t pool = 0; pool < nP; ++pool) {
    for (int k = 0; k < p.n_qualities; ++k) {
      double lhs = 0.0, scale = 0.0;
      for (std::size_t a = 0; a < p.arcs_sp.size(); ++a) {
        if (to_size(p.arcs_sp[a].second) != pool) continue;
        const double v = p.sources[to_size(p.arcs_sp[a].first)].quality[to_size(k)] * f[p.idx_sp(a)];
        lhs += v;
        scale += std::fabs(v);
      }
      const double rhs = pt.q[p.qidx(pool, k)] * out_p[pool];
      scale += std::fabs(rhs);
      c.quality_balance = std::max(c.quality_balance, rel(std::fabs(lhs - rhs), scale));
    }
  }
  for (std::size_t t = 0; t < nT; ++t) {
    for (int k = 0; k < p.n_qualities; ++k) {
      const double spec = p.terminals[t].spec[to_size(k)];
      if (is_inf(spec)) continue;
      double lhs = 0.0, scale = std::fabs(spec) * std::fabs(in_t[t]);
      for (std::size_t a = 0; a < p.arcs_pt.size(); ++a) {
        if (to_size(p.arcs_pt[a].second) != t) continue;
        const double v = pt.q[p.qidx(to_size(p.arcs_pt[a].first), k)] * f[p.idx_pt(a)];
        lhs += v;
        scale += std::fabs(v);
      }
      for (std::size_t a = 0; a < p.arcs_st.size(); ++a) {
        if (to_size(p.arcs_st[a].second) != t) continue;
        const double v = p.sources[to_size(p.arcs_st[a].first)].quality[to_size(k)] * f[p.idx_st(a)];
        lhs += v;
        scale += std::fabs(v);
      }
      c.terminal_spec = std::max(c.terminal_spec, rel(lhs - spec * in_t[t], scale));
    }
  }
  c.worst = std::max({c.bounds, c.supply, c.capacity, c.demand, c.material, c.quality_balance, c.terminal_spec});
  return c;
}

}  // namespace shodhan::pooling
