#pragma once

namespace shodhan {

/// Value used to store an infinite bound. Any bound with |value| >= kInf is
/// treated as infinite; readers clamp such values to exactly +/-kInf.
inline constexpr double kInf = 1e30;

constexpr bool is_inf(double x) noexcept { return x >= kInf || x <= -kInf; }
constexpr bool is_pos_inf(double x) noexcept { return x >= kInf; }
constexpr bool is_neg_inf(double x) noexcept { return x <= -kInf; }

}  // namespace shodhan
