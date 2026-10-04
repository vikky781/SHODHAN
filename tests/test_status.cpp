#include <cstring>

#include "shodhan/constants.hpp"
#include "shodhan/status.hpp"
#include "test_harness.hpp"

using namespace shodhan;

TEST_CASE(status_names_are_stable) {
  CHECK_EQ(std::string(to_string(Status::Optimal)), "Optimal");
  CHECK_EQ(std::string(to_string(Status::Infeasible)), "Infeasible");
  CHECK_EQ(std::string(to_string(Status::Unbounded)), "Unbounded");
  CHECK_EQ(std::string(to_string(Status::InfeasibleOrUnbounded)), "InfeasibleOrUnbounded");
  CHECK_EQ(std::string(to_string(Status::TimeLimit)), "TimeLimit");
  CHECK_EQ(std::string(to_string(Status::IterationLimit)), "IterationLimit");
  CHECK_EQ(std::string(to_string(Status::NumericalError)), "NumericalError");
  CHECK_EQ(std::string(to_string(Status::NodeLimit)), "NodeLimit");
  CHECK_EQ(std::string(to_string(Status::Interrupted)), "Interrupted");
  CHECK_EQ(std::string(to_string(Status::NotImplemented)), "NotImplemented");
  CHECK_EQ(std::string(to_string(Status::ReadError)), "ReadError");
}

TEST_CASE(inf_helpers) {
  CHECK(is_inf(kInf));
  CHECK(is_inf(-kInf));
  CHECK(is_inf(1e31));
  CHECK(!is_inf(1e29));
  CHECK(is_pos_inf(kInf));
  CHECK(!is_pos_inf(-kInf));
  CHECK(is_neg_inf(-kInf));
  CHECK(!is_inf(0.0));
}
