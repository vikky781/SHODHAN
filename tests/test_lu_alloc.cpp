// No dynamic allocation inside ftran / btran / update after warm-up.
//
// The test binary replaces the global operator new with a counting version. The
// count only runs inside the calls under test (a guard object around each call),
// so allocations by the test itself and by factorize() are not counted.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <vector>

#include "shodhan/basis_factor.hpp"
#include "support/lu_testing.hpp"
#include "test_harness.hpp"

// GCC reports a mismatch between the replaced operator new (malloc) and operator
// delete (free) when they are inlined at a use site; the pair is consistent.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

namespace {

bool g_counting = false;
long g_allocations = 0;

struct CountGuard {
  CountGuard() { g_counting = true; }
  ~CountGuard() { g_counting = false; }
};

}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_allocations;
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
  if (g_counting) ++g_allocations;
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace shodhan;
using namespace shodhan::testing;

namespace {

// Runs `cycles` factorize-then-update cycles; counts allocations made inside the
// solves and updates only. Returns that count.
long run_cycles(BasisFactor& f, const TestBasis& tb, std::vector<Index>& basis, int cycles, int steps, Rng& rng) {
  const Index m = tb.m();
  const Index total = tb.A.n_cols + m;
  SparseWork w(m), y(m);
  long counted = 0;
  for (int c = 0; c < cycles; ++c) {
    if (f.factorize(tb.A, basis) != FactorStatus::Ok) throw std::runtime_error("test basis lost its rank");
    for (int s = 0; s < steps; ++s) {
      const Index q = rng.range(0, total - 1);
      if (std::find(basis.begin(), basis.end(), q) != basis.end()) continue;
      w.clear();
      if (q < tb.A.n_cols) {
        for (Index t = tb.A.col_start[to_size(q)]; t < tb.A.col_start[to_size(q) + 1]; ++t) w.set(tb.A.row_index[to_size(t)], tb.A.value[to_size(t)]);
      } else {
        w.set(q - tb.A.n_cols, -1.0);
      }
      const long before = g_allocations;
      Index p = -1;
      {
        CountGuard guard;
        f.ftran(w, true);
        double best = 0.0;
        for (const Index i : w.indices()) {
          if (std::fabs(w[i]) > best) { best = std::fabs(w[i]); p = i; }
        }
      }
      if (p < 0 || std::fabs(w[p]) < 1e-6) { counted += g_allocations - before; continue; }
      y.clear();
      y.set(p, 1.0);
      FactorStatus st;
      {
        CountGuard guard;
        f.btran(y);
        st = f.update(p);
      }
      counted += g_allocations - before;
      basis[to_size(p)] = q;
      if (st != FactorStatus::Ok) break;  // refactor (counted separately: not at all)
    }
  }
  return counted;
}

}  // namespace

TEST_CASE(lu_no_allocation_in_solves_and_updates_after_warm_up) {
  {
    // Control: the counter does see an allocation made inside a guard.
    const long before = g_allocations;
    {
      CountGuard guard;
      std::vector<double>* v = new std::vector<double>(100, 1.0);
      CHECK(v->size() == 100);
      delete v;
    }
    CHECK(g_allocations - before >= 2);
  }
  TestBasis tb = make_family_basis(4 /* arrowhead */, 60, 2024);
  std::vector<Index> basis = tb.basis;
  FactorParams params;
  params.max_updates = 40;
  BasisFactor f(params);
  Rng warm(1), rng(2);
  // Warm-up: the work buffers and arenas grow to their working size.
  run_cycles(f, tb, basis, 12, 60, warm);
  const long allocations = run_cycles(f, tb, basis, 6, 60, rng);
  std::cout << "    allocations inside ftran/btran/update over 6 cycles after warm-up: " << allocations << "\n";
  CHECK_EQ(allocations, 0L);
  // Sparse (hypersparse path) and dense right-hand sides do not allocate either.
  SparseWork w(60);
  REQUIRE(f.factorize(tb.A, basis) == FactorStatus::Ok);
  w.set(3, 1.0);
  f.ftran(w);  // first use of each path may size internal buffers
  w.clear();
  w.set(5, 1.0);
  f.btran(w);
  const long before = g_allocations;
  {
    CountGuard guard;
    for (int k = 0; k < 20; ++k) {
      w.clear();
      w.set(k, 1.0);
      f.ftran(w);
      w.clear();
      w.set(59 - k, 1.0);
      f.btran(w);
    }
  }
  CHECK_EQ(g_allocations - before, 0L);
}
