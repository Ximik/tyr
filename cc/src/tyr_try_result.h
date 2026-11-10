#pragma once
#include <lean/lean.h>
#include <c10/util/Exception.h>
#include <exception>
#include <stdexcept>
#include <utility>

namespace tyr {

// Build errors with the Lean allocator, avoiding a second C++ allocation while
// handling an exception. Lean's allocator retains its ordinary fatal OOM policy.
inline lean_object* errorResult(const char* context, const char* message) noexcept {
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

// A failure that depends on the values of the inputs (no real logarithm, a
// factorization that does not converge), as opposed to a contract violation or
// a missing kernel. Only `tryCall` turns it into an error.
struct DataError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define TYR_DATA_CHECK(cond, ...) \
  do { if (!(cond)) throw ::tyr::DataError(::c10::str(__VA_ARGS__)); } while (0)

// The export behind a `foo?`: data failures, including libtorch's own
// non-convergence (`c10::LinAlgError`), become `Except.error`. Every other
// exception propagates to Tyr's terminate handler and aborts.
template <typename F>
lean_object* tryCall(const char* context, F&& operation) {
  try {
    auto* payload = std::forward<F>(operation)();
    auto* result = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(result, 0, payload);
    return result;
  } catch (const DataError& error) {
    return errorResult(context, error.what());
  } catch (const c10::LinAlgError& error) {
    return errorResult(context, error.what_without_backtrace());
  }
}

inline lean_object* ioErrorResult(const char* context, const char* message) noexcept {
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
lean_object* ioCall(const char* context, F&& operation) noexcept {
  try {
    return std::forward<F>(operation)();
  } catch (const c10::Error& error) {
    return ioErrorResult(context, error.what_without_backtrace());
  } catch (const std::exception& error) {
    return ioErrorResult(context, error.what());
  } catch (...) {
    return ioErrorResult(context, "unknown native exception");
  }
}

} // namespace tyr
