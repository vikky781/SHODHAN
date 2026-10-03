#include <vector>

#include "shodhan/sparse_work.hpp"
#include "support/work_check.hpp"
#include "test_harness.hpp"

using namespace shodhan;

TEST_CASE(sparse_work_add_set_and_clear) {
  SparseWork w(10);
  CHECK_EQ(w.size(), 10);
  CHECK_EQ(w.count(), 0);
  w.add(3, 1.5);
  w.add(7, -2.0);
  w.add(3, 0.5);
  CHECK_EQ(w.count(), 2);
  CHECK_NEAR(w[3], 2.0, 0.0);
  CHECK_NEAR(w[7], -2.0, 0.0);
  w.set(7, 4.0);
  CHECK_EQ(w.count(), 2);
  CHECK_NEAR(w[7], 4.0, 0.0);
  CHECK(w.is_listed(3) && w.is_listed(7) && !w.is_listed(0));
  CHECK_NEAR(w.density(), 0.2, 1e-15);
  CHECK(testing::work_invariant_holds(w));
  w.clear();
  CHECK_EQ(w.count(), 0);
  CHECK_NEAR(w[3], 0.0, 0.0);
  CHECK(!w.is_listed(3));
  CHECK(testing::work_invariant_holds(w));
}

TEST_CASE(sparse_work_cancellation_does_not_duplicate_the_index) {
  SparseWork w(5);
  w.add(2, 1.0);
  w.add(2, -1.0);  // cancels to exactly zero but stays listed
  w.add(2, 3.0);
  CHECK_EQ(w.count(), 1);
  CHECK_NEAR(w[2], 3.0, 0.0);
  CHECK(testing::work_invariant_holds(w));
}

TEST_CASE(sparse_work_drop_small_is_absolute) {
  SparseWork w(6);
  w.set(0, 1e-15);
  w.set(1, -1e-13);
  w.set(2, 0.0);
  w.set(3, 5.0);
  w.set(4, -1e-14);  // exactly at the tolerance: dropped (<=)
  const Index dropped = w.drop_small();  // default 1e-14
  CHECK_EQ(dropped, 3);
  CHECK_EQ(w.count(), 2);
  CHECK_NEAR(w[0], 0.0, 0.0);
  CHECK_NEAR(w[1], -1e-13, 0.0);
  CHECK_NEAR(w[3], 5.0, 0.0);
  CHECK(!w.is_listed(0) && w.is_listed(1));
  CHECK(testing::work_invariant_holds(w));
  CHECK_EQ(w.drop_small(0.0), 0);
  CHECK_EQ(w.drop_small(1.0), 1);  // removes the -1e-13 entry
  CHECK_EQ(w.count(), 1);
}

TEST_CASE(sparse_work_dense_and_sparse_conversions) {
  SparseWork w;
  w.load_dense(std::vector<double>{0.0, 2.0, 0.0, -3.0, 1e-20});
  CHECK_EQ(w.size(), 5);
  CHECK_EQ(w.count(), 3);
  std::vector<double> out(5, 99.0);
  w.to_dense(out);
  CHECK(out == (std::vector<double>{0.0, 2.0, 0.0, -3.0, 1e-20}));

  const std::vector<Index> idx{4, 1, 4};
  const std::vector<double> val{1.0, 2.0, 0.5};
  w.load_sparse(idx, val);  // repeated index adds
  CHECK_EQ(w.count(), 2);
  CHECK_NEAR(w[4], 1.5, 0.0);
  CHECK_NEAR(w[1], 2.0, 0.0);
  CHECK_NEAR(w[3], 0.0, 0.0);  // old content was cleared

  SparseWork copy;
  copy.assign(w);
  CHECK_EQ(copy.size(), 5);
  CHECK_EQ(copy.count(), 2);
  CHECK_NEAR(copy[4], 1.5, 0.0);
  w.clear();
  CHECK_NEAR(copy[4], 1.5, 0.0);  // independent
  CHECK(testing::work_invariant_holds(copy));
}

TEST_CASE(sparse_work_reindex_after_raw_writes) {
  SparseWork w(8);
  w.set(1, 1.0);
  w.set(5, 2.0);
  double* raw = w.raw();
  raw[5] = 0.0;
  raw[6] = 7.0;
  raw[0] = -1.0;
  w.reindex();
  CHECK_EQ(w.count(), 3);
  CHECK(w.is_listed(0) && w.is_listed(1) && w.is_listed(6) && !w.is_listed(5));
  CHECK(testing::work_invariant_holds(w));
  CHECK_NEAR(w.norm_inf(), 7.0, 0.0);
}

TEST_CASE(sparse_work_swap) {
  SparseWork a(4), b(4);
  a.set(1, 1.0);
  b.set(2, 2.0);
  b.set(3, 3.0);
  a.swap(b);
  CHECK_EQ(a.count(), 2);
  CHECK_EQ(b.count(), 1);
  CHECK_NEAR(a[3], 3.0, 0.0);
  CHECK_NEAR(b[1], 1.0, 0.0);
  CHECK(testing::work_invariant_holds(a) && testing::work_invariant_holds(b));
}
