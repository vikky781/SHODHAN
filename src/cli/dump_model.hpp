#pragma once

#include <iosfwd>

#include "shodhan/lp_model.hpp"

namespace shodhan::cli {

/// Prints a canonical text form of a parsed model, one record per line, fields separated by a tab:
///
///   name <TAB> NAME
///   sense <TAB> min|max
///   offset <TAB> number
///   rows <TAB> m          cols <TAB> n
///   row <TAB> name <TAB> lower <TAB> upper
///   col <TAB> name <TAB> cost <TAB> lower <TAB> upper <TAB> C|I      (I: integer or binary)
///   entry <TAB> row name <TAB> col name <TAB> value                  (column by column, rows ascending)
///
/// Numbers are the shortest decimal that reads back to the same double; bounds of magnitude 1e30 or
/// more are printed as inf / -inf. Used by the differential test against KASAUTI's parser.
void print_model_dump(const LpModel& model, std::ostream& out);

}  // namespace shodhan::cli
