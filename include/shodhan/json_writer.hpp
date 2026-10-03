#pragma once

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace shodhan {

/// Streaming JSON writer (no third-party code). Objects and arrays are written
/// in order; a misuse (value without key inside an object, unbalanced end) throws
/// std::logic_error.
///
/// Strings are written as UTF-8. Control characters and the quote and backslash are
/// escaped; a byte that is not part of a valid UTF-8 sequence is written as the
/// escape \udcXX (XX = the byte), which a reader that decodes the original text
/// with Python's "surrogateescape" handler maps back to the same string.
/// Doubles use the shortest decimal that reads back to the identical double
/// (std::to_chars); a non-finite double throws std::invalid_argument.
class JsonWriter {
 public:
  explicit JsonWriter(std::ostream& out, bool pretty = true) : out_(out), pretty_(pretty) {}

  void begin_object();
  void end_object();
  void begin_array();
  void end_array();
  void key(std::string_view name);

  void value(std::string_view s);
  void value(const char* s) { value(std::string_view(s)); }
  void value(const std::string& s) { value(std::string_view(s)); }
  void value(double v);
  void value(long long v);
  void value(int v) { value(static_cast<long long>(v)); }
  void value(unsigned long long v);
  void value(bool v);
  void null();

  /// Shortest round-trip decimal text of v (also used by tests).
  static std::string format_double(double v);
  /// The JSON string literal (with quotes) for s.
  static std::string quote(std::string_view s);

 private:
  struct Frame {
    bool is_object;
    bool has_items = false;
    bool expecting_value = false;  // object: a key was written
  };
  void before_value();
  void newline_indent();

  std::ostream& out_;
  bool pretty_;
  std::vector<Frame> stack_;
};

}  // namespace shodhan
