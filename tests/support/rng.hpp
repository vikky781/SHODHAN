#pragma once

// Platform-independent pseudo-random numbers for tests (the standard
// distributions give different sequences on different standard libraries).

#include <cstdint>
#include <utility>
#include <vector>

namespace shodhan::testing {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  /// Uniform in [0, 1).
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }

  double uniform(double lo, double hi) { return lo + (hi - lo) * unit(); }

  /// Uniform integer in [lo, hi].
  int range(int lo, int hi) {
    return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }

  bool chance(double p) { return unit() < p; }

  template <typename T>
  void shuffle(std::vector<T>& v) {
    for (std::size_t i = v.size(); i > 1; --i) {
      const std::size_t j = static_cast<std::size_t>(next() % i);
      std::swap(v[i - 1], v[j]);
    }
  }

 private:
  std::uint64_t state_;
};

}  // namespace shodhan::testing
