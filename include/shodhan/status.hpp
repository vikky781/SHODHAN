#pragma once

namespace shodhan {

/// Outcome of an operation on a model (reading, solving, ...).
enum class Status {
  Optimal,
  Infeasible,
  Unbounded,
  InfeasibleOrUnbounded,
  TimeLimit,
  IterationLimit,
  NodeLimit,  ///< branch and bound stopped at its node limit (see docs/MIP.md)
  NumericalError,
  Interrupted,
  NotImplemented,
  NonConvex,  ///< the quadratic term is not positive semidefinite (docs/QP.md)
  ReadError,
};

/// Stable, human-readable name of a status value (never null).
const char* to_string(Status status) noexcept;

}  // namespace shodhan
