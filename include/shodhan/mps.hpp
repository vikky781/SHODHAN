#pragma once

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include "shodhan/lp_model.hpp"

namespace shodhan {

enum class MpsFormat {
  Auto,   // detect (see docs/MPS_FORMAT.md)
  Fixed,  // column-based fields; names may contain spaces
  Free,   // whitespace-separated fields
};

struct MpsReadOptions {
  MpsFormat format = MpsFormat::Auto;
};

struct MpsReadResult {
  /// True when parsing succeeded; `model` is only meaningful then.
  bool ok = false;
  /// On failure: "<source>:<line>: <message>: '<offending text>'" (the line
  /// part is omitted for errors that are not tied to a line).
  std::string error;
  /// Non-fatal diagnostics, each prefixed with "<source>:<line>: ".
  std::vector<std::string> warnings;
  MpsFormat detected_format = MpsFormat::Free;
  LpModel model;
};

/// Reads an MPS file. A path ending in ".gz" is decompressed when the library
/// was built with SHODHAN_ENABLE_ZLIB, and rejected with a clear error
/// otherwise. Never throws on malformed input.
MpsReadResult read_mps_file(const std::string& path, const MpsReadOptions& options = {});

/// Reads MPS text from a stream; `source_name` is used in diagnostics.
MpsReadResult read_mps_stream(std::istream& in, const std::string& source_name,
                              const MpsReadOptions& options = {});

/// Reads MPS text held in memory; `source_name` is used in diagnostics.
MpsReadResult read_mps_string(std::string_view text, const std::string& source_name,
                              const MpsReadOptions& options = {});

/// Writes the model in free-format MPS using shortest round-trip number
/// formatting. Row/column names are generated ("R<i>", "C<j>") when the model
/// has none. Returns false and sets *error (if given) when the model cannot
/// be represented, e.g. a name that is empty or contains whitespace, or
/// duplicate names. Nothing is written in that case.
bool write_mps(const LpModel& model, std::ostream& out, std::string* error = nullptr);

}  // namespace shodhan
