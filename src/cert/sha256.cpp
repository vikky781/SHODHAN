#include "shodhan/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace shodhan {

namespace {

// The round constants are the first 32 bits of the fractional parts of the cube roots of the first 64
// primes, the initial hash values those of the square roots of the first 8 primes (FIPS 180-4). They are
// computed here instead of typed in; the unit tests check the standard vectors, which would fail on any
// wrong constant.
struct Constants {
  std::array<std::uint32_t, 64> k{};
  std::array<std::uint32_t, 8> h0{};
  Constants() {
    std::vector<int> primes;
    for (int n = 2; primes.size() < 64; ++n) {
      bool is_prime = true;
      for (const int p : primes) {
        if (p * p > n) break;
        if (n % p == 0) {
          is_prime = false;
          break;
        }
      }
      if (is_prime) primes.push_back(n);
    }
    auto frac32 = [](double v) {
      double ip;
      const double f = std::modf(v, &ip);
      return static_cast<std::uint32_t>(std::floor(f * 4294967296.0));
    };
    for (std::size_t i = 0; i < 64; ++i) k[i] = frac32(std::cbrt(static_cast<double>(primes[i])));
    for (std::size_t i = 0; i < 8; ++i) h0[i] = frac32(std::sqrt(static_cast<double>(primes[i])));
  }
};

const Constants& constants() {
  static const Constants c;
  return c;
}

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256() : h_(constants().h0) {}

void Sha256::process_block(const std::uint8_t* p) {
  const auto& k = constants().k;
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
           (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) | static_cast<std::uint32_t>(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t big_s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t t1 = h + big_s1 + ch + k[static_cast<std::size_t>(i)] + w[i];
    const std::uint32_t big_s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = big_s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  total_bytes_ += size;
  while (size > 0) {
    const std::size_t take = std::min<std::size_t>(size, 64 - buffered_);
    for (std::size_t i = 0; i < take; ++i) buffer_[buffered_ + i] = p[i];
    buffered_ += take;
    p += take;
    size -= take;
    if (buffered_ == 64) {
      process_block(buffer_.data());
      buffered_ = 0;
    }
  }
}

std::array<std::uint8_t, 32> Sha256::digest() {
  const std::uint64_t bit_length = total_bytes_ * 8;
  const std::uint8_t one = 0x80;
  update(&one, 1);
  const std::uint8_t zero = 0;
  while (buffered_ != 56) update(&zero, 1);
  std::uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bit_length >> (56 - 8 * i));
  update(len, 8);
  std::array<std::uint8_t, 32> out{};
  for (std::size_t i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
    out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
    out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
    out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
  }
  return out;
}

std::string Sha256::hex_digest() {
  static const char* const digits = "0123456789abcdef";
  const auto d = digest();
  std::string s;
  s.reserve(64);
  for (const std::uint8_t b : d) {
    s.push_back(digits[b >> 4]);
    s.push_back(digits[b & 15]);
  }
  return s;
}

std::string sha256_hex(std::string_view data) {
  Sha256 h;
  h.update(data);
  return h.hex_digest();
}

bool sha256_file_hex(const std::string& path, std::string* hex) {
  hex->clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  Sha256 h;
  std::vector<char> buf(1 << 16);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    h.update(buf.data(), static_cast<std::size_t>(in.gcount()));
  }
  if (in.bad()) return false;
  *hex = h.hex_digest();
  return true;
}

}  // namespace shodhan
