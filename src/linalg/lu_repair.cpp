#include <stdexcept>

#include "shodhan/basis_factor.hpp"

namespace shodhan {

std::vector<BasisSubstitution> BasisFactor::repair(const SparseMatrix& A, std::vector<Index>& basis_vars) {
  std::vector<BasisSubstitution> changes;
  if (status_ != FactorStatus::RankDeficient) return changes;
  if (basis_vars.size() != to_size(m_)) {
    throw std::invalid_argument("BasisFactor::repair: basis size changed since factorize");
  }
  const Index n = A.n_cols;
  const std::vector<Index> original = basis_vars;

  // One round pairs the deficient positions (ascending) with the unpivoted
  // rows (ascending) and refactorizes. For an exactly singular basis one round
  // is enough. For a numerically singular one the refactorization may pick
  // other pivots and find new deficiencies, so a few more rounds are allowed;
  // the limit only guards against a loop that cannot end.
  constexpr int kMaxRounds = 8;
  for (int round = 0; round < kMaxRounds && status_ == FactorStatus::RankDeficient; ++round) {
    const std::vector<Index> positions = deficient_positions_;
    const std::vector<Index> rows = unpivoted_rows_;
    for (std::size_t k = 0; k < positions.size(); ++k) basis_vars[to_size(positions[k])] = n + rows[k];
    factorize(A, basis_vars);
  }

  // Net substitutions relative to the basis that was passed in, by position.
  for (std::size_t p = 0; p < basis_vars.size(); ++p) {
    if (basis_vars[p] != original[p]) {
      changes.push_back({static_cast<Index>(p), original[p], basis_vars[p]});
    }
  }
  return changes;
}

}  // namespace shodhan
