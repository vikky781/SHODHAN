#include "shodhan/json_writer.hpp"

#include <charconv>
#include <cmath>
#include <ostream>
#include <stdexcept>

namespace shodhan {

namespace {

// Length of the valid UTF-8 sequence starting at s[i], or 0 if there is none.
std::size_t utf8_length(std::string_view s, std::size_t i) {
  const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  auto cont = [&](std::size_t k) { return k < s.size() && (b(k) & 0xC0) == 0x80; };
  if (c < 0x80) return 1;
  if (c >= 0xC2 && c <= 0xDF) return cont(i + 1) ? 2 : 0;
  if (c >= 0xE0 && c <= 0xEF) {
    if (!cont(i + 1) || !cont(i + 2)) return 0;
    const unsigned char c1 = b(i + 1);
    if (c == 0xE0 && c1 < 0xA0) return 0;   // overlong
    if (c == 0xED && c1 >= 0xA0) return 0;  // surrogates
    return 3;
  }
  if (c >= 0xF0 && c <= 0xF4) {
    if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return 0;
    const unsigned char c1 = b(i + 1);
    if (c == 0xF0 && c1 < 0x90) return 0;
    if (c == 0xF4 && c1 >= 0x90) return 0;
    return 4;
  }
  return 0;
}

}  // namespace

std::string JsonWriter::quote(std::string_view s) {
  static const char* const hex = "0123456789abcdef";
  std::string out = "\"";
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '"') {
      out += "\\\"";
      ++i;
    } else if (c == '\\') {
      out += "\\\\";
      ++i;
    } else if (c < 0x20) {
      switch (c) {
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
          out += "\\u00";
          out += hex[c >> 4];
          out += hex[c & 15];
      }
      ++i;
    } else if (c < 0x80) {
      out += static_cast<char>(c);
      ++i;
    } else {
      const std::size_t len = utf8_length(s, i);
      if (len == 0) {  // not valid UTF-8: escape the byte as a lone low surrogate
        out += "\\udc";
        out += hex[c >> 4];
        out += hex[c & 15];
        ++i;
      } else {
        out.append(s.substr(i, len));
        i += len;
      }
    }
  }
  out += '"';
  return out;
}

std::string JsonWriter::format_double(double v) {
  if (!std::isfinite(v)) throw std::invalid_argument("JsonWriter: non-finite number");
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), v);
  if (res.ec != std::errc()) throw std::runtime_error("JsonWriter: to_chars failed");
  return std::string(buf, res.ptr);
}

void JsonWriter::newline_indent() {
  if (!pretty_) return;
  out_ << '\n';
  for (std::size_t i = 0; i < stack_.size(); ++i) out_ << "  ";
}

void JsonWriter::before_value() {
  if (stack_.empty()) return;
  Frame& f = stack_.back();
  if (f.is_object) {
    if (!f.expecting_value) throw std::logic_error("JsonWriter: value without a key inside an object");
    f.expecting_value = false;
  } else {
    if (f.has_items) out_ << ',';
    newline_indent();
  }
  f.has_items = true;
}

void JsonWriter::begin_object() {
  before_value();
  out_ << '{';
  stack_.push_back({true});
}

void JsonWriter::end_object() {
  if (stack_.empty() || !stack_.back().is_object || stack_.back().expecting_value) throw std::logic_error("JsonWriter: unbalanced end_object");
  const bool had = stack_.back().has_items;
  stack_.pop_back();
  if (had) newline_indent();
  out_ << '}';
}

void JsonWriter::begin_array() {
  before_value();
  out_ << '[';
  stack_.push_back({false});
}

void JsonWriter::end_array() {
  if (stack_.empty() || stack_.back().is_object) throw std::logic_error("JsonWriter: unbalanced end_array");
  const bool had = stack_.back().has_items;
  stack_.pop_back();
  if (had) newline_indent();
  out_ << ']';
}

void JsonWriter::key(std::string_view name) {
  if (stack_.empty() || !stack_.back().is_object || stack_.back().expecting_value) throw std::logic_error("JsonWriter: key outside an object");
  Frame& f = stack_.back();
  if (f.has_items) out_ << ',';
  newline_indent();
  out_ << quote(name) << (pretty_ ? ": " : ":");
  f.expecting_value = true;
}

void JsonWriter::value(std::string_view s) {
  before_value();
  out_ << quote(s);
}

void JsonWriter::value(double v) {
  const std::string text = format_double(v);
  before_value();
  out_ << text;
}

void JsonWriter::value(long long v) {
  before_value();
  out_ << v;
}

void JsonWriter::value(unsigned long long v) {
  before_value();
  out_ << v;
}

void JsonWriter::value(bool v) {
  before_value();
  out_ << (v ? "true" : "false");
}

void JsonWriter::null() {
  before_value();
  out_ << "null";
}

}  // namespace shodhan
