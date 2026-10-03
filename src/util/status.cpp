#include "shodhan/status.hpp"

namespace shodhan {

const char* to_string(Status status) noexcept {
  switch (status) {
    case Status::Optimal:
      return "Optimal";
    case Status::Infeasible:
      return "Infeasible";
    case Status::Unbounded:
      return "Unbounded";
    case Status::InfeasibleOrUnbounded:
      return "InfeasibleOrUnbounded";
    case Status::TimeLimit:
      return "TimeLimit";
    case Status::IterationLimit:
      return "IterationLimit";
    case Status::NumericalError:
      return "NumericalError";
    case Status::Interrupted:
      return "Interrupted";
    case Status::NotImplemented:
      return "NotImplemented";
    case Status::ReadError:
      return "ReadError";
  }
  return "Unknown";
}

}  // namespace shodhan
