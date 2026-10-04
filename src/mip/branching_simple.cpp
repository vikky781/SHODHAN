// Branching rules that need no history: most fractional and lowest index.

#include <cmath>

#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

double fractionality(double v) {
  const double f = v - std::floor(v);
  return std::min(f, 1.0 - f);
}

class MostFractionalRule final : public BranchingRule {
 public:
  const char* name() const override { return "mostfrac"; }
  BranchDecision select(SearchState& s) override {
    BranchDecision d;
    double best = -1.0;
    for (const Index j : s.lp.fractional) {  // ascending index: ties go to the lowest index
      const double f = fractionality(s.lp.x[to_size(j)]);
      if (f > best + 1e-12) {
        best = f;
        d.col = j;
      }
    }
    return d;
  }
};

class FirstIndexRule final : public BranchingRule {
 public:
  const char* name() const override { return "first"; }
  BranchDecision select(SearchState& s) override {
    BranchDecision d;
    d.col = s.lp.fractional.front();  // fractional is in ascending index order
    return d;
  }
};

}  // namespace

void register_branching_simple(PluginRegistry& reg) {
  reg.branching.add("mostfrac", [] { return std::make_unique<MostFractionalRule>(); });
  reg.branching.add("first", [] { return std::make_unique<FirstIndexRule>(); });
}

}  // namespace shodhan::mip
