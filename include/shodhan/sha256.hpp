#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace shodhan {

/// Streaming SHA-256 (FIPS 180-4), standard library only.
class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t size);
  void update(std::string_view s) { update(s.data(), s.size()); }
  /// Finishes the hash; the object must not be updated afterwards.
  std::array<std::uint8_t, 32> digest();
  /// Lower-case hex of the digest.
  std::string hex_digest();

 private:
  void process_block(const std::uint8_t* block);

  std::array<std::uint32_t, 8> h_;
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_ = 0;
  std::uint64_t total_bytes_ = 0;
};

/// SHA-256 of a string, as lower-case hex.
std::string sha256_hex(std::string_view data);

/// SHA-256 of the exact bytes of a file; returns false (and leaves *hex empty)
/// if the file cannot be read.
bool sha256_file_hex(const std::string& path, std::string* hex);

}  // namespace shodhan
