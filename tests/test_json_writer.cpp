#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "shodhan/json_writer.hpp"
#include "support/rng.hpp"
#include "test_harness.hpp"

using namespace shodhan;

TEST_CASE(json_quote_escapes_special_characters) {
  CHECK_EQ(JsonWriter::quote("plain"), "\"plain\"");
  CHECK_EQ(JsonWriter::quote("a\"b"), "\"a\\\"b\"");
  CHECK_EQ(JsonWriter::quote("back\\slash"), "\"back\\\\slash\"");
  CHECK_EQ(JsonWriter::quote("tab\there\nnew\rline"), "\"tab\\there\\nnew\\rline\"");
  CHECK_EQ(JsonWriter::quote(std::string("nul\0x", 5)), "\"nul\\u0000x\"");
  CHECK_EQ(JsonWriter::quote("ctl\x01\x1f"), "\"ctl\\u0001\\u001f\"");
  CHECK_EQ(JsonWriter::quote("with space and /slash"), "\"with space and /slash\"");
  // Valid UTF-8 passes through unchanged (e-acute, a snowman, a 4-byte emoji).
  CHECK_EQ(JsonWriter::quote("caf\xc3\xa9 \xe2\x98\x83 \xf0\x9f\x98\x80"), "\"caf\xc3\xa9 \xe2\x98\x83 \xf0\x9f\x98\x80\"");
  // A lone 0xe9 byte (Latin-1) is not UTF-8: it is written as a low surrogate escape.
  CHECK_EQ(JsonWriter::quote("caf\xe9"), "\"caf\\udce9\"");
  CHECK_EQ(JsonWriter::quote("\xc3"), "\"\\udcc3\"");              // truncated sequence
  CHECK_EQ(JsonWriter::quote("\xc0\xaf"), "\"\\udcc0\\udcaf\"");   // overlong
}

TEST_CASE(json_doubles_round_trip_exactly) {
  CHECK_EQ(JsonWriter::format_double(0.1), "0.1");
  CHECK_EQ(JsonWriter::format_double(100.0), "100");
  CHECK_EQ(JsonWriter::format_double(-0.0), "-0");
  CHECK_EQ(JsonWriter::format_double(1e30), "1e+30");
  CHECK_EQ(JsonWriter::format_double(1.0 / 3.0), "0.3333333333333333");
  shodhan::testing::Rng rng(12345);
  for (int i = 0; i < 20000; ++i) {
    double v;
    switch (i % 4) {
      case 0: v = rng.uniform(-1e6, 1e6); break;
      case 1: v = std::ldexp(rng.uniform(1.0, 2.0), rng.range(-1070, 1000)); break;  // includes denormals
      case 2: v = static_cast<double>(rng.range(-1000000, 1000000)) / 1024.0; break;
      default: v = rng.uniform(-1.0, 1.0) * 1e-300; break;
    }
    const std::string text = JsonWriter::format_double(v);
    const double back = std::strtod(text.c_str(), nullptr);
    CHECK(back == v);
  }
  bool threw = false;
  try {
    JsonWriter::format_double(std::numeric_limits<double>::infinity());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    JsonWriter::format_double(std::nan(""));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE(json_writer_structure_compact_and_pretty) {
  std::ostringstream compact;
  {
    JsonWriter w(compact, false);
    w.begin_object();
    w.key("a");
    w.value(1);
    w.key("b");
    w.begin_array();
    w.value("x");
    w.value(2.5);
    w.value(true);
    w.null();
    w.end_array();
    w.key("c");
    w.begin_object();
    w.end_object();
    w.key("d");
    w.begin_array();
    w.end_array();
    w.end_object();
  }
  CHECK_EQ(compact.str(), "{\"a\":1,\"b\":[\"x\",2.5,true,null],\"c\":{},\"d\":[]}");
  std::ostringstream pretty;
  {
    JsonWriter w(pretty);
    w.begin_object();
    w.key("k y");
    w.value("v");
    w.key("n");
    w.begin_object();
    w.key("z");
    w.value(3);
    w.end_object();
    w.end_object();
  }
  CHECK_EQ(pretty.str(), "{\n  \"k y\": \"v\",\n  \"n\": {\n    \"z\": 3\n  }\n}");
}

TEST_CASE(json_writer_rejects_misuse) {
  std::ostringstream os;
  JsonWriter w(os);
  bool threw = false;
  try {
    w.key("outside");
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  w.begin_object();
  threw = false;
  try {
    w.value(1);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);  // value without a key
  w.key("a");
  threw = false;
  try {
    w.end_object();
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);  // key still waiting for its value
  w.value(1);
  w.end_object();
  threw = false;
  try {
    w.end_array();
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}
