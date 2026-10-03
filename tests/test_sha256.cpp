#include <cstdio>
#include <fstream>
#include <string>

#include "shodhan/sha256.hpp"
#include "test_harness.hpp"

using namespace shodhan;

TEST_CASE(sha256_standard_vectors) {
  CHECK_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // 448-bit message from FIPS 180-4: padding spills into a second block.
  CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // One million 'a' (many blocks), fed in uneven pieces to exercise the buffering.
  Sha256 h;
  std::string piece(1000, 'a');
  for (int i = 0; i < 1000; ++i) h.update(piece.data(), i % 2 == 0 ? piece.size() : piece.size() / 2);
  for (int i = 0; i < 500; ++i) h.update(piece.data(), piece.size() / 2);
  CHECK_EQ(h.hex_digest(), sha256_hex(std::string(500 * 1000 + 500 * 500 + 500 * 500, 'a')));
  Sha256 million;
  for (int i = 0; i < 1000; ++i) million.update(piece);
  CHECK_EQ(million.hex_digest(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE(sha256_block_boundaries) {
  // Lengths around the 55/56/63/64 byte boundaries must agree between one-shot and byte-wise hashing.
  for (std::size_t n = 0; n < 200; ++n) {
    const std::string s(n, 'x');
    Sha256 h;
    for (const char c : s) h.update(&c, 1);
    CHECK_EQ(h.hex_digest(), sha256_hex(s));
  }
  CHECK(sha256_hex(std::string(55, 'a')) != sha256_hex(std::string(56, 'a')));
}

TEST_CASE(sha256_of_a_file_matches_the_string_hash) {
  std::string hex;
  CHECK(!sha256_file_hex("this_file_does_not_exist.bin", &hex));
  CHECK(hex.empty());
  const std::string path = "sha256_test_file.tmp";
  const std::string data = std::string("line one") + char(13) + char(10) + std::string(1, char(0)) + "binary" + std::string(100000, 'q');
  {
    std::ofstream out(path, std::ios::binary);
    out << data;
  }
  CHECK(sha256_file_hex(path, &hex));
  CHECK_EQ(hex, sha256_hex(data));  // exact bytes, no newline translation
  std::remove(path.c_str());
}
