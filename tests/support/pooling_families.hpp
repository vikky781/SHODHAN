#pragma once

// Test-only generators and an independent grid reference for the pooling driver (SYNTHETIC instances: invented
// numbers, not from any publication).

#include <cstdint>
#include <vector>

#include "shodhan/pooling.hpp"

namespace shodhan::testing {

/// One pool, one quality, 2-4 sources, 1-2 terminals (the shape of the independent reference below).
pooling::PoolProblem make_single_pool(std::uint64_t seed);

/// 2 pools, 2 qualities, 3-4 sources, 2 terminals.
pooling::PoolProblem make_two_pool(std::uint64_t seed);

/// Near-global reference for a problem with ONE pool and ONE quality: the pool quality q is scanned over `grid_points`
/// values of its range; for each q the LP with the pool quality equality ENFORCED (a linear constraint once q is fixed)
/// is solved with LpSolver; the best interval is then refined by a ternary search. Every value found is the objective of a
/// point that is feasible for the nonlinear model, so the reference never exceeds the global maximum.
double single_pool_reference(const pooling::PoolProblem& p, int grid_points, int refine_steps = 60);

/// The same for ANY pooling problem on a regular grid with `points_per_axis` values for each pool quality (so
/// points_per_axis ^ (pools * qualities) LPs): a coarse reference for the multi-pool tests.
double grid_reference(const pooling::PoolProblem& p, int points_per_axis);

/// The value of the LP with all pool qualities fixed to `q` AND the pool quality balances enforced (-inf if infeasible).
double fixed_q_enforced_lp(const pooling::PoolProblem& p, const std::vector<double>& q);

}  // namespace shodhan::testing
