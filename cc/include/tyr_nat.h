#pragma once

#include <lean/lean.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tyr {

// A Lean `Nat` argument, checked against `T`, the native type it is narrowed
// to. Nats above LEAN_MAX_SMALL_NAT are heap objects and never fit an
// integer type of at most 64 bits.
template <typename T>
uint64_t natTo(b_lean_obj_arg value, const char* name) {
  static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(uint64_t));
  constexpr auto max = static_cast<uint64_t>(std::numeric_limits<T>::max());
  if (lean_is_scalar(value)) {
    const auto n = static_cast<uint64_t>(lean_unbox(value));
    if (n <= max) return n;
  }
  throw std::out_of_range(std::string(name) + " exceeds " + std::to_string(max));
}

// Releases an owned Lean argument on every path out of its scope. Lean passes a
// parameter with a default value owned even when it is declared `@&`.
struct OwnedArg {
  lean_object* object;
  explicit OwnedArg(lean_obj_arg value) : object(value) {}
  OwnedArg(const OwnedArg&) = delete;
  OwnedArg& operator=(const OwnedArg&) = delete;
  ~OwnedArg() { lean_dec(object); }
};

// `a * b` when it does not exceed the maximum of `T`.
template <typename T>
uint64_t checkedProduct(uint64_t a, uint64_t b, const char* name) {
  constexpr auto max = static_cast<uint64_t>(std::numeric_limits<T>::max());
  uint64_t product = 0;
  if (__builtin_mul_overflow(a, b, &product) || product > max) {
    throw std::out_of_range(std::string(name) + " exceeds " + std::to_string(max));
  }
  return product;
}

}  // namespace tyr
