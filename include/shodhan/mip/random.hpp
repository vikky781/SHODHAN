#pragma once

#include <cstdint>

namespace shodhan::mip {

/// SplitMix64: a small deterministic generator, identical on every platform (the standard distributions are
/// not). Used for seeded tie-breaking in heuristics.
class Random {
 public:
  explicit Random(std::uint64_t seed = 0) : state_(seed) {}
  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  /// Uniform in [0, 1).
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
  /// Uniform integer in [0, n) for n > 0.
  std::uint64_t below(std::uint64_t n) { return next() % n; }

 private:
  std::uint64_t state_;
};

}  // namespace shodhan::mip
