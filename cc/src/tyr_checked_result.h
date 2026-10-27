#pragma once
#include <lean/lean.h>
#include <c10/util/Exception.h>
#include <exception>
#include <utility>

namespace tyr {

// Build errors with the Lean allocator, avoiding a second C++ allocation while
// handling an exception. Lean's allocator retains its ordinary fatal OOM policy.
inline lean_object* checkedError(const char* context, const char* message) noexcept {
  auto* separator = lean_mk_string(": ");
  auto* prefix = lean_string_append(lean_mk_string(context), separator);
  lean_dec(separator);
  auto* detail = lean_mk_string(message);
  auto* text = lean_string_append(prefix, detail);
  lean_dec(detail);
  auto* result = lean_alloc_ctor(0, 1, 0);
  lean_ctor_set(result, 0, text);
  return result;
}

// The callable returns an owned Lean payload. All native work (including
// ownership guards) executes inside this firewall, never across a Lean frame.
// The noexcept boundary also covers exceptions not derived from std::exception.
template <typename F>
lean_object* checkedCall(const char* context, F&& operation) noexcept {
  try {
    auto* payload = std::forward<F>(operation)();
    auto* result = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(result, 0, payload);
    return result;
  } catch (const c10::Error& error) {
    return checkedError(context, error.what_without_backtrace());
  } catch (const std::exception& error) {
    return checkedError(context, error.what());
  } catch (...) {
    return checkedError(context, "unknown native exception");
  }
}

inline lean_object* checkedIoError(const char* context, const char* message) noexcept {
  auto* separator = lean_mk_string(": ");
  auto* prefix = lean_string_append(lean_mk_string(context), separator);
  lean_dec(separator);
  auto* detail = lean_mk_string(message);
  auto* text = lean_string_append(prefix, detail);
  lean_dec(detail);
  return lean_io_result_mk_error(lean_mk_io_user_error(text));
}

// The callable already returns IO.Result, including ordinary validation errors.
// Keep every C++ allocation and native operation inside the recovery boundary.
template <typename F>
lean_object* checkedIoCall(const char* context, F&& operation) noexcept {
  try {
    return std::forward<F>(operation)();
  } catch (const c10::Error& error) {
    return checkedIoError(context, error.what_without_backtrace());
  } catch (const std::exception& error) {
    return checkedIoError(context, error.what());
  } catch (...) {
    return checkedIoError(context, "unknown native exception");
  }
}

} // namespace tyr
