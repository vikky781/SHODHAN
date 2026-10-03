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
  NumericalError,
  Interrupted,
  NotImplemented,
  ReadError,
};

/// Stable, human-readable name of a status value (never null).
const char* to_string(Status status) noexcept;

}  // namespace shodhan
