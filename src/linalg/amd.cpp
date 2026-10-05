// Approximate minimum degree ordering on the quotient graph, written from the description in
//   P. R. Amestoy, T. A. Davis, I. S. Duff, "An approximate minimum degree ordering algorithm",
//   SIAM J. Matrix Anal. Appl. 17 (1996) 886-905.
//
// Notation. After eliminating pivots the graph is kept as a quotient graph: every remaining node is a variable (a
// supervariable of weight nv) with a list A of variable neighbours and a list E of adjacent elements; every element e
// stands for an eliminated pivot and has the list L of the variables it connects. Eliminating a pivot p forms the new
// element Lp = (A_p + union of L_e over e in E_p) \ {p}, absorbs the elements of E_p, updates the lists of the variables
// in Lp, and approximates their external degrees by
//   d_i = min( n - k - nv_i,  d_i_old + |Lp \ i|,  |A_i \ i| + |Lp \ i| + sum_{e in E_i \ p} |L_e \ Lp| ),
// all sizes weighted by nv. Variables with identical lists are merged (indistinguishable), and variables whose only
// neighbour is the new element are eliminated with the pivot (mass elimination).

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "shodhan/sparse_ldl.hpp"

namespace shodhan {

std::vector<Index> natural_order(Index n) {
  std::vector<Index> p(to_size(n));
  std::iota(p.begin(), p.end(), Index{0});
  return p;
}

std::vector<Index> amd_order(const SparseMatrix& lower, AmdStats* stats_out) {
  const Index n = lower.n_cols;
  AmdStats stats;
  const std::size_t sn = to_size(n);
  // ---- symmetric adjacency without the diagonal ----
  std::vector<std::vector<Index>> A(sn);
  for (Index j = 0; j < n; ++j) {
    for (Index p = lower.col_start[to_size(j)]; p < lower.col_start[to_size(j) + 1]; ++p) {
      const Index i = lower.row_index[to_size(p)];
      if (i == j) continue;
      A[to_size(i)].push_back(j);
      A[to_size(j)].push_back(i);
    }
  }
  for (std::vector<Index>& a : A) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  }
  std::vector<std::vector<Index>> E(sn), L(sn);
  std::vector<Index> nv(sn, 1), deg(sn), len(sn, 0);
  std::vector<char> is_elem(sn, 0), dead(sn, 0);
  for (std::size_t i = 0; i < sn; ++i) deg[i] = static_cast<Index>(A[i].size());

  // Degree buckets (doubly linked lists).
  std::vector<Index> head(sn + 1, -1), next(sn, -1), prev(sn, -1);
  Index mindeg = 0;
  auto insert = [&](Index i) {
    const Index d = std::min<Index>(std::max<Index>(deg[to_size(i)], 0), n);
    deg[to_size(i)] = d;
    next[to_size(i)] = head[to_size(d)];
    prev[to_size(i)] = -1;
    if (head[to_size(d)] != -1) prev[to_size(head[to_size(d)])] = i;
    head[to_size(d)] = i;
    mindeg = std::min(mindeg, d);
  };
  auto remove = [&](Index i) {
    const Index d = deg[to_size(i)];
    if (prev[to_size(i)] != -1) next[to_size(prev[to_size(i)])] = next[to_size(i)];
    else head[to_size(d)] = next[to_size(i)];
    if (next[to_size(i)] != -1) prev[to_size(next[to_size(i)])] = prev[to_size(i)];
  };
  for (Index i = 0; i < n; ++i) insert(i);

  // Chains of the variables merged into a supervariable (head variable first).
  std::vector<Index> mnext(sn, -1), mtail(sn);
  std::iota(mtail.begin(), mtail.end(), Index{0});

  std::vector<Index> mark(sn, -1), wmark(sn, -1), w(sn, 0), cmark(sn, -1);
  Index stamp = 0, wstamp = 0, cstamp = 0;
  std::vector<Index> order;
  order.reserve(sn);
  auto is_var = [&](Index j) { return !dead[to_size(j)] && !is_elem[to_size(j)]; };
  auto output_chain = [&](Index v) {
    for (Index c = v; c != -1; c = mnext[to_size(c)]) order.push_back(c);
  };

  Index nel = 0;
  std::vector<Index> Lp;
  std::vector<std::pair<Index, Index>> hashed;  // (hash, variable)
  while (nel < n) {
    while (mindeg <= n && head[to_size(mindeg)] == -1) ++mindeg;
    const Index p = head[to_size(mindeg)];
    remove(p);
    ++stats.pivots;
    ++stamp;
    mark[to_size(p)] = stamp;
    Lp.clear();
    Index degp = 0;
    for (const Index j : A[to_size(p)]) {
      if (is_var(j) && mark[to_size(j)] != stamp) {
        mark[to_size(j)] = stamp;
        Lp.push_back(j);
        degp += nv[to_size(j)];
      }
    }
    for (const Index e : E[to_size(p)]) {
      if (dead[to_size(e)]) continue;
      for (const Index j : L[to_size(e)]) {
        if (is_var(j) && mark[to_size(j)] != stamp) {
          mark[to_size(j)] = stamp;
          Lp.push_back(j);
          degp += nv[to_size(j)];
        }
      }
      dead[to_size(e)] = 1;  // absorbed into the new element p
      ++stats.absorbed_elements;
      std::vector<Index>().swap(L[to_size(e)]);
    }
    std::vector<Index>().swap(A[to_size(p)]);
    std::vector<Index>().swap(E[to_size(p)]);
    is_elem[to_size(p)] = 1;
    const Index nvp = nv[to_size(p)];
    nel += nvp;
    for (const Index i : Lp) remove(i);

    // |L_e \ Lp| for every element adjacent to a variable of Lp.
    ++wstamp;
    for (const Index i : Lp) {
      for (const Index e : E[to_size(i)]) {
        if (dead[to_size(e)] || e == p) continue;
        if (wmark[to_size(e)] != wstamp) {
          wmark[to_size(e)] = wstamp;
          w[to_size(e)] = len[to_size(e)];
        }
        w[to_size(e)] -= nv[to_size(i)];
      }
    }
    // Prune the lists of the variables of Lp; remember the sums for the degree update.
    std::vector<Index> sum_e(Lp.size(), 0), sum_a(Lp.size(), 0);
    for (std::size_t q = 0; q < Lp.size(); ++q) {
      const Index i = Lp[q];
      std::vector<Index>& Ei = E[to_size(i)];
      std::size_t keep = 0;
      for (const Index e : Ei) {
        if (dead[to_size(e)] || e == p) continue;
        if (w[to_size(e)] == 0) {  // aggressive absorption: e lies entirely inside Lp
          if (!dead[to_size(e)]) {
            dead[to_size(e)] = 1;
            ++stats.absorbed_elements;
            std::vector<Index>().swap(L[to_size(e)]);
          }
          continue;
        }
        Ei[keep++] = e;
        sum_e[q] += w[to_size(e)];
      }
      Ei.resize(keep);
      Ei.push_back(p);
      std::vector<Index>& Ai = A[to_size(i)];
      keep = 0;
      for (const Index j : Ai) {
        if (j == p || !is_var(j) || mark[to_size(j)] == stamp) continue;
        Ai[keep++] = j;
        sum_a[q] += nv[to_size(j)];
      }
      Ai.resize(keep);
    }
    // Mass elimination: a variable whose only neighbour is the new element is indistinguishable from the pivot.
    std::vector<Index> keep_lp;
    std::vector<Index> keep_sum_e, keep_sum_a;
    output_chain(p);
    Index mass = 0;
    for (std::size_t q = 0; q < Lp.size(); ++q) {
      const Index i = Lp[q];
      if (A[to_size(i)].empty() && E[to_size(i)].size() == 1) {
        nel += nv[to_size(i)];
        degp -= nv[to_size(i)];
        output_chain(i);
        dead[to_size(i)] = 1;
        ++stats.mass_eliminations;
        ++mass;
        continue;
      }
      keep_lp.push_back(i);
      keep_sum_e.push_back(sum_e[q]);
      keep_sum_a.push_back(sum_a[q]);
    }
    Lp.swap(keep_lp);
    sum_e.swap(keep_sum_e);
    sum_a.swap(keep_sum_a);
    (void)mass;

    // Approximate external degrees and hashes for the indistinguishability test.
    hashed.clear();
    for (std::size_t q = 0; q < Lp.size(); ++q) {
      const Index i = Lp[q];
      const Index lp_minus_i = degp - nv[to_size(i)];
      const Index d1 = n - nel - nv[to_size(i)];
      const Index d2 = deg[to_size(i)] + lp_minus_i;
      const Index d3 = sum_a[q] + lp_minus_i + sum_e[q];
      deg[to_size(i)] = std::max<Index>(0, std::min(d1, std::min(d2, d3)));
      long long h = 0;
      for (const Index j : A[to_size(i)]) h += j;
      for (const Index e : E[to_size(i)]) h += e;
      hashed.emplace_back(static_cast<Index>(h % std::max<long long>(n, 1)), i);
    }
    // Merge indistinguishable variables (same A and E lists).
    std::sort(hashed.begin(), hashed.end());
    for (std::size_t a = 0; a < hashed.size();) {
      std::size_t b = a;
      while (b < hashed.size() && hashed[b].first == hashed[a].first) ++b;
      for (std::size_t x = a; x < b; ++x) {
        const Index i = hashed[x].second;
        if (dead[to_size(i)]) continue;
        ++cstamp;
        for (const Index j : A[to_size(i)]) cmark[to_size(j)] = cstamp;
        for (const Index e : E[to_size(i)]) cmark[to_size(e)] = cstamp;
        for (std::size_t y = x + 1; y < b; ++y) {
          const Index j = hashed[y].second;
          if (dead[to_size(j)]) continue;
          if (A[to_size(j)].size() != A[to_size(i)].size() || E[to_size(j)].size() != E[to_size(i)].size()) continue;
          bool same = true;
          for (const Index t : A[to_size(j)]) same = same && cmark[to_size(t)] == cstamp;
          for (const Index t : E[to_size(j)]) same = same && cmark[to_size(t)] == cstamp;
          if (!same) continue;
          // j is indistinguishable from i: merge it into i.
          nv[to_size(i)] += nv[to_size(j)];
          deg[to_size(i)] = std::max<Index>(0, deg[to_size(i)] - nv[to_size(j)]);
          nv[to_size(j)] = 0;
          dead[to_size(j)] = 1;
          mnext[to_size(mtail[to_size(i)])] = j;
          mtail[to_size(i)] = mtail[to_size(j)];
          ++stats.supervariable_merges;
        }
      }
      a = b;
    }
    // The element p keeps the surviving variables; reinsert them with their new degrees.
    std::vector<Index> survivors;
    for (const Index i : Lp) {
      if (!dead[to_size(i)]) survivors.push_back(i);
    }
    Index degp_now = 0;
    for (const Index i : survivors) degp_now += nv[to_size(i)];
    L[to_size(p)] = survivors;
    len[to_size(p)] = degp_now;
    for (const Index i : survivors) insert(i);
  }
  if (stats_out != nullptr) *stats_out = stats;
  return order;
}

}  // namespace shodhan
