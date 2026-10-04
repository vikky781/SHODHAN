#include "shodhan/mip/options.hpp"

namespace shodhan::mip {

const char* to_string(BranchingKind kind) noexcept {
  switch (kind) {
    case BranchingKind::Reliability: return "reliability";
    case BranchingKind::Pseudocost: return "pseudocost";
    case BranchingKind::MostFractional: return "mostfrac";
    case BranchingKind::FirstIndex: return "first";
  }
  return "unknown";
}

const char* to_string(NodeSelectKind kind) noexcept {
  switch (kind) {
    case NodeSelectKind::BestBound: return "bestbound";
    case NodeSelectKind::DepthFirst: return "depth";
    case NodeSelectKind::BestEstimate: return "bestestimate";
  }
  return "unknown";
}

bool parse_branching(const std::string& name, BranchingKind* out) {
  for (const BranchingKind k : {BranchingKind::Reliability, BranchingKind::Pseudocost, BranchingKind::MostFractional, BranchingKind::FirstIndex}) {
    if (name == to_string(k)) {
      *out = k;
      return true;
    }
  }
  return false;
}

bool parse_node_select(const std::string& name, NodeSelectKind* out) {
  for (const NodeSelectKind k : {NodeSelectKind::BestBound, NodeSelectKind::DepthFirst, NodeSelectKind::BestEstimate}) {
    if (name == to_string(k)) {
      *out = k;
      return true;
    }
  }
  return false;
}

}  // namespace shodhan::mip
