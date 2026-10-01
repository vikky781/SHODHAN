#pragma once

// Tiny header-only test harness: registration, CHECK macros, summary and a
// non-zero exit code on failure. No third-party dependency.

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace shtest {

struct TestCase {
  const char* name;
  void (*fn)();
};

struct State {
  int checks = 0;
  int failed_checks = 0;
  bool current_failed = false;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline State& state() {
  static State s;
  return s;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

template <typename T>
std::string stringify(const T& v) {
  if constexpr (std::is_enum_v<T>) {
    return std::to_string(static_cast<long long>(v));
  } else if constexpr (requires(std::ostream& o, const T& t) { o << t; }) {
    std::ostringstream os;
    os.precision(17);
    os << v;
    return os.str();
  } else {
    return "<unprintable>";
  }
}

inline void fail(const char* file, int line, const std::string& what) {
  std::cerr << file << ":" << line << ": CHECK FAILED: " << what << "\n";
  ++state().failed_checks;
  state().current_failed = true;
}

inline bool check(bool ok, const char* file, int line, const char* expr) {
  ++state().checks;
  if (!ok) fail(file, line, expr);
  return ok;
}

template <typename A, typename B>
bool check_eq(const A& a, const B& b, const char* file, int line, const char* ea, const char* eb) {
  ++state().checks;
  if (a == b) return true;
  fail(file, line, std::string(ea) + " == " + eb + "  (" + stringify(a) + " vs " + stringify(b) + ")");
  return false;
}

inline bool check_near(double a, double b, double tol, const char* file, int line, const char* ea,
                       const char* eb) {
  ++state().checks;
  if (std::fabs(a - b) <= tol) return true;
  fail(file, line,
       std::string(ea) + " ~= " + eb + "  (" + stringify(a) + " vs " + stringify(b) +
           ", tol " + stringify(tol) + ")");
  return false;
}

inline bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

/// Runs all registered tests. An optional first argument filters by substring.
inline int run_all(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  int passed = 0;
  for (const TestCase& t : registry()) {
    if (filter && std::strstr(t.name, filter) == nullptr) continue;
    ++ran;
    state().current_failed = false;
    try {
      t.fn();
    } catch (const std::exception& e) {
      fail(t.name, 0, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      fail(t.name, 0, "unexpected non-standard exception");
    }
    if (state().current_failed) {
      std::cout << "[FAIL] " << t.name << "\n";
    } else {
      ++passed;
      std::cout << "[ ok ] " << t.name << "\n";
    }
  }
  std::cout << "\n"
            << ran << " tests run, " << passed << " passed, " << (ran - passed) << " failed; "
            << state().checks << " checks, " << state().failed_checks << " failed\n";
  return (ran - passed) == 0 && ran > 0 ? 0 : 1;
}

}  // namespace shtest

#define SH_CONCAT_INNER(a, b) a##b
#define SH_CONCAT(a, b) SH_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                    \
  static void name();                                                      \
  static ::shtest::Registrar SH_CONCAT(sh_registrar_, name)(#name, &name); \
  static void name()

#define CHECK(expr) ::shtest::check(static_cast<bool>(expr), __FILE__, __LINE__, #expr)
#define CHECK_EQ(a, b) ::shtest::check_eq((a), (b), __FILE__, __LINE__, #a, #b)
#define CHECK_NEAR(a, b, tol) ::shtest::check_near((a), (b), (tol), __FILE__, __LINE__, #a, #b)
#define CHECK_CONTAINS(haystack, needle) \
  ::shtest::check(::shtest::contains((haystack), (needle)), __FILE__, __LINE__, \
                  #haystack " contains " #needle)
// Aborts the current test case when the condition is false.
#define REQUIRE(expr)                                                      \
  do {                                                                     \
    if (!::shtest::check(static_cast<bool>(expr), __FILE__, __LINE__, #expr)) return; \
  } while (false)
