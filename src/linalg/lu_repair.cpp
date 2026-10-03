#include <stdexcept>

#include "shodhan/basis_factor.hpp"

namespace shodhan {

std::vector<BasisSubstitution> BasisFactor::repair(const SparseMatrix& A, std::vector<Index>& basis_vars) {
  std::vector<BasisSubstitution> changes;
  if (status_ != FactorStatus::RankDeficient) return changes;
  if (basis_vars.size() != to_size(m_)) throw std::invalid_argument("BasisFactor::repair: basis size changed since factorize");
  const Index n = A.n_cols;
  const std::vector<Index> positions = deficient_positions_;
  const std::vector<Index> rows = unpivoted_rows_;
  changes.reserve(positions.size());
  for (std::size_t k = 0; k < positions.size(); ++k) {
    BasisSubstitution s;
    s.position = positions[k];
    s.old_var = basis_vars[to_size(positions[k])];
    s.new_var = n + rows[k];
    basis_vars[to_size(positions[k])] = s.new_var;
    changes.push_back(s);
  }
  factorize(A, basis_vars);
  return changes;
}

}  // namespace shodhan
