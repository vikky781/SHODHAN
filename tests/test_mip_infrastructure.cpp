// MIP infrastructure: node tree, open set, incumbent manager, plugin registry.

#include <cmath>
#include <string>
#include <vector>

#include "shodhan/mip/incumbent.hpp"
#include "shodhan/mip/node_tree.hpp"
#include "shodhan/mip/plugins.hpp"
#include "shodhan/presolve.hpp"
#include "test_harness.hpp"

using namespace shodhan;
using namespace shodhan::mip;

namespace {

// max 5a + 4b + 3c + y   s.t.  2a + 3b + c + y <= 5,  a + b + c <= 2,  a,b,c binary, 0 <= y <= 1.5 continuous.
LpModel small_mip() {
  LpModel m;
  m.name = "small";
  m.sense = Sense::Maximize;
  m.n_rows = 2;
  m.n_cols = 4;
  std::string err;
  SparseMatrix::from_triplets(2, 4, {{0, 0, 2.0}, {0, 1, 3.0}, {0, 2, 1.0}, {0, 3, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}}, &m.A, &err);
  m.col_cost = {5.0, 4.0, 3.0, 1.0};
  m.col_lower = {0.0, 0.0, 0.0, 0.0};
  m.col_upper = {1.0, 1.0, 1.0, 1.5};
  m.col_type = {ColType::Binary, ColType::Binary, ColType::Binary, ColType::Continuous};
  m.row_lower = {-kInf, -kInf};
  m.row_upper = {5.0, 2.0};
  m.col_names = {"a", "b", "c", "y"};
  m.row_names = {"r1", "r2"};
  return m;
}

}  // namespace

TEST_CASE(node_tree_reconstructs_bounds_along_the_path_and_ids_are_deterministic) {
  NodeTree tree(-10.0);
  CHECK_EQ(tree.size(), 1u);
  const NodeId a = tree.add_child(0, 2, -1, 0.5, -9.0, -8.0, BoundChange{2, 0.0, 0.0}, nullptr);
  const NodeId b = tree.add_child(0, 2, +1, 0.5, -9.0, -8.5, BoundChange{2, 1.0, 1.0}, nullptr);
  const NodeId c = tree.add_child(a, 1, +1, 0.3, -8.0, -7.0, BoundChange{1, 1.0, 5.0}, nullptr);
  CHECK_EQ(a, 1);
  CHECK_EQ(b, 2);
  CHECK_EQ(c, 3);
  CHECK_EQ(tree.at(c).depth, 2);
  CHECK_EQ(tree.max_depth(), 2);
  std::vector<double> lo = {0, 0, 0}, hi = {9, 9, 9};
  CHECK(tree.path_bounds(c, lo, hi));
  CHECK_EQ(lo[2], 0.0);
  CHECK_EQ(hi[2], 0.0);  // x2 <= 0 from the ancestor
  CHECK_EQ(lo[1], 1.0);
  CHECK_EQ(hi[1], 5.0);
  lo = {0, 0, 0};
  hi = {9, 9, 9};
  CHECK(tree.path_bounds(b, lo, hi));
  CHECK_EQ(lo[2], 1.0);
  CHECK_EQ(hi[2], 1.0);
  // A node that crosses an ancestor's bounds is infeasible.
  const NodeId d = tree.add_child(c, 2, +1, 0.0, -7.0, -7.0, BoundChange{2, 1.0, 1.0}, nullptr);
  lo = {0, 0, 0};
  hi = {9, 9, 9};
  CHECK(!tree.path_bounds(d, lo, hi));
}

TEST_CASE(open_set_gives_the_smallest_bound) {
  OpenSet open;
  CHECK(open.empty());
  CHECK_EQ(open.min_bound(), kInf);
  open.insert(3, 5.0);
  open.insert(1, 2.0);
  open.insert(2, 2.0);
  CHECK_EQ(open.size(), 3u);
  CHECK_EQ(open.min_bound(), 2.0);
  open.erase(1, 2.0);
  open.erase(2, 2.0);
  CHECK_EQ(open.min_bound(), 5.0);
}

TEST_CASE(incumbent_manager_verifies_snaps_completes_and_counts_rejections) {
  const LpModel m = small_mip();
  MipOptions opt;
  IncumbentManager inc(m, nullptr, opt);
  CHECK(!inc.has_incumbent());

  // a = 1, b = 0, c = 1 (snapped from 0.999999), y fractional garbage: the LP completion fixes y = 1.5 - ...
  // row 1: 2 + 1 + y <= 5 -> y <= 2, y <= 1.5; objective 5 + 3 + 1.5 = 9.5.
  CHECK(inc.submit({0.999999, 0.0, 1.000001, 0.0}, "test") == SubmitOutcome::Accepted);
  CHECK(inc.has_incumbent());
  CHECK(std::fabs(inc.objective() - 9.5) < 1e-7);
  CHECK_EQ(inc.x()[0], 1.0);  // exactly integral after snapping
  CHECK_EQ(inc.x()[2], 1.0);
  CHECK(std::fabs(inc.x()[3] - 1.5) < 1e-7);

  // Not strictly better.
  CHECK(inc.submit({1.0, 0.0, 1.0, 0.0}, "test") == SubmitOutcome::NotBetter);
  // Infeasible: a + b + c = 3 > 2.
  CHECK(inc.submit({1.0, 1.0, 1.0, 0.0}, "test") == SubmitOutcome::Rejected);
  CHECK(inc.rejections().lp_resolve_failed + inc.rejections().row_violated >= 1);
  // Integer column outside its bounds after snapping.
  CHECK(inc.submit({2.0, 0.0, 0.0, 0.0}, "test") == SubmitOutcome::Rejected);
  CHECK(inc.rejections().not_integral >= 1);
  // Wrong size.
  CHECK(inc.submit({1.0}, "test") == SubmitOutcome::Rejected);
  CHECK_EQ(inc.rejections().wrong_size, 1);
  CHECK_EQ(inc.found(), 1);
  CHECK_EQ(inc.submitted(), 5);

  // A better point replaces it (b + a: 5 + 4 = 9 + y: 2 + 3 + y <= 5 -> y = 0 -> 9 < 9.5: not better).
  CHECK(inc.submit({1.0, 1.0, 0.0, 0.0}, "test") == SubmitOutcome::NotBetter);
}

TEST_CASE(incumbent_manager_accepts_a_pure_integer_point_without_an_lp) {
  LpModel m = small_mip();
  m.col_type[3] = ColType::Integer;
  m.col_upper[3] = 1.0;
  MipOptions opt;
  IncumbentManager inc(m, nullptr, opt);
  CHECK(inc.submit({1.0, 0.0, 1.0, 1.0}, "x") == SubmitOutcome::Accepted);
  CHECK_EQ(inc.objective(), 9.0);
  // A row violation is caught on the ORIGINAL model.
  CHECK(inc.submit({1.0, 1.0, 1.0, 0.0}, "x") == SubmitOutcome::Rejected);
  CHECK_EQ(inc.rejections().row_violated, 1);
}

TEST_CASE(incumbent_manager_maps_presolved_points_back_and_reports_the_original_objective) {
  const LpModel m = small_mip();
  PresolveOptions po;
  po.is_mip = true;
  po.need_duals = false;
  const PresolveResult pre = presolve(m, po);
  MipOptions opt;
  CHECK(pre.status == PresolveStatus::Reduced);
  if (pre.status != PresolveStatus::Reduced) return;
  IncumbentManager inc(m, &pre.stack, opt);
  // The point a = 1, b = 0, c = 1 expressed in the reduced columns (col_map: reduced -> original).
  std::vector<double> xr(pre.stack.col_map.size(), 0.0);
  const std::vector<double> full = {1.0, 0.0, 1.0, 0.0};
  for (std::size_t k = 0; k < xr.size(); ++k) xr[k] = full[static_cast<std::size_t>(pre.stack.col_map[k])];
  CHECK(inc.submit(xr, "presolved") == SubmitOutcome::Accepted);
  CHECK(std::fabs(inc.objective() - 9.5) < 1e-7);
}

namespace {
class DummySelector final : public NodeSelector {
 public:
  const char* name() const override { return "dummy"; }
  void push(const Node& n) override { ids_.push_back(n.id); }
  NodeId pop() override {
    if (ids_.empty()) return kNoNode;
    const NodeId id = ids_.back();
    ids_.pop_back();
    return id;
  }
  bool empty() const override { return ids_.empty(); }
  std::size_t size() const override { return ids_.size(); }

 private:
  std::vector<NodeId> ids_;
};
}  // namespace

TEST_CASE(plugin_registry_creates_plugins_by_name) {
  PluginRegistry reg;
  reg.selectors.add("dummy", [] { return std::make_unique<DummySelector>(); });
  CHECK(reg.selectors.has("dummy"));
  CHECK(!reg.selectors.has("other"));
  auto sel = reg.selectors.make("dummy");
  CHECK(sel != nullptr);
  CHECK_EQ(std::string(sel->name()), "dummy");
  Node n;
  n.id = 7;
  sel->push(n);
  CHECK_EQ(sel->size(), 1u);
  CHECK_EQ(sel->pop(), 7);
  CHECK_EQ(sel->pop(), kNoNode);
  CHECK(reg.selectors.make("other") == nullptr);
}
