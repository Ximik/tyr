#include <lean/lean.h>
#include "generated/tyr_ffi.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

// ============================================================================
// FloatBuffer: contiguous unboxed double array, analogous to
// Haskell's Data.Vector.Storable Double or a Rust Vec<f64>.
//
// Lean sees it as an opaque external object managed via lean_external_class.
// Mutating operations are exposed through IO.
// ============================================================================

struct FloatBuf {
  double* data;
  size_t len;
  size_t cap;
};

static void float_buf_finalize(void* p) {
  auto* b = static_cast<FloatBuf*>(p);
  std::free(b->data);
  std::free(b);
}

static void float_buf_foreach(void* /*p*/, b_lean_obj_arg /*fn*/) {
  // No nested Lean objects to traverse.
}

static lean_external_class* g_fb_class = nullptr;

static lean_external_class* fb_class() {
  if (!g_fb_class)
    g_fb_class = lean_register_external_class(float_buf_finalize, float_buf_foreach);
  return g_fb_class;
}

static inline FloatBuf* fb_of(lean_object* o) {
  return static_cast<FloatBuf*>(lean_get_external_data(o));
}

static inline lean_object* fb_wrap(FloatBuf* b) {
  return lean_alloc_external(fb_class(), b);
}

static FloatBuf* fb_alloc(size_t cap) {
  if (cap > SIZE_MAX / sizeof(double)) return nullptr;
  auto* b = static_cast<FloatBuf*>(std::malloc(sizeof(FloatBuf)));
  if (!b) return nullptr;
  b->cap = cap;
  b->data = cap > 0 ? static_cast<double*>(std::malloc(sizeof(double) * cap)) : nullptr;
  if (cap > 0 && !b->data) {
    std::free(b);
    return nullptr;
  }
  b->len = 0;
  return b;
}

static bool fb_ensure(FloatBuf* b, size_t extra) {
  constexpr size_t max_cap = SIZE_MAX / sizeof(double);
  if (extra > max_cap - b->len) return false;
  size_t need = b->len + extra;
  if (need <= b->cap) return true;
  size_t new_cap = b->cap == 0 ? 64 : b->cap;
  while (new_cap < need)
    new_cap = new_cap > max_cap / 2 ? max_cap : new_cap * 2;
  auto* data = static_cast<double*>(std::realloc(b->data, sizeof(double) * new_cap));
  if (!data) return false;
  b->data = data;
  b->cap = new_cap;
  return true;
}

static lean_object* fb_error(const char* message) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(message)));
}

static lean_object* fb_growth_error(lean_object* owned_buffer) {
  lean_dec(owned_buffer);
  return fb_error("FloatBuffer capacity exceeded or allocation failed");
}

extern "C" {

// -- Construction -----------------------------------------------------------

lean_object* lean_float_buffer_mk_empty(b_lean_obj_arg capacity) {
  // Every addressable double buffer has a capacity below Lean's small-Nat
  // limit. Reject larger Nats before unboxing; never truncate a big Nat.
  if (!lean_is_scalar(capacity) || lean_unbox(capacity) > SIZE_MAX / sizeof(double))
    return fb_error("FloatBuffer capacity exceeds the addressable element count");
  auto* buffer = fb_alloc(lean_unbox(capacity));
  if (!buffer) return fb_error("FloatBuffer allocation failed");
  return lean_io_result_mk_ok(fb_wrap(buffer));
}

// -- Query ------------------------------------------------------------------

lean_object* lean_float_buffer_size(b_lean_obj_arg buf) {
  return lean_usize_to_nat(fb_of(buf)->len);
}

lean_object* lean_float_buffer_rms(b_lean_obj_arg buf) {
  auto* b = fb_of(buf);
  if (b->len == 0)
    return lean_io_result_mk_ok(lean_box_float(0.0));
  double sum_sq = 0.0;
  for (size_t i = 0; i < b->len; ++i) {
    double x = b->data[i];
    sum_sq += x * x;
  }
  return lean_io_result_mk_ok(
      lean_box_float(std::sqrt(sum_sq / static_cast<double>(b->len))));
}

double lean_float_buffer_get(b_lean_obj_arg buf, b_lean_obj_arg index) {
  auto* buffer = fb_of(buf);
  if (!lean_is_scalar(index) || lean_unbox(index) >= buffer->len)
    lean_internal_panic("FloatBuffer.uget: index out of bounds");
  return buffer->data[lean_unbox(index)];
}

// -- Mutation (takes owned buf, returns it) ---------------------------------

lean_object* lean_float_buffer_push(lean_object* buf_obj, double x) {
  auto* b = fb_of(buf_obj);
  if (!fb_ensure(b, 1)) return fb_growth_error(buf_obj);
  b->data[b->len++] = x;
  return lean_io_result_mk_ok(buf_obj);
}

lean_object* lean_float_buffer_append(lean_object* dst_obj, b_lean_obj_arg src_obj) {
  auto* dst = fb_of(dst_obj);
  auto* src = fb_of(src_obj);
  if (src->len > 0) {
    if (!fb_ensure(dst, src->len)) return fb_growth_error(dst_obj);
    std::memcpy(dst->data + dst->len, src->data, sizeof(double) * src->len);
    dst->len += src->len;
  }
  return lean_io_result_mk_ok(dst_obj);
}

lean_object* lean_float_buffer_append_array(lean_object* dst_obj, b_lean_obj_arg arr) {
  auto* dst = fb_of(dst_obj);
  size_t n = lean_array_size(arr);
  if (n > 0) {
    if (!fb_ensure(dst, n)) return fb_growth_error(dst_obj);
    for (size_t i = 0; i < n; ++i) {
      dst->data[dst->len + i] = lean_unbox_float(lean_array_uget(arr, i));
    }
    dst->len += n;
  }
  return lean_io_result_mk_ok(dst_obj);
}

lean_object* lean_float_buffer_clear(lean_object* buf_obj) {
  fb_of(buf_obj)->len = 0;
  return lean_io_result_mk_ok(buf_obj);
}

// -- Conversion -------------------------------------------------------------

lean_object* lean_float_buffer_to_array(b_lean_obj_arg buf) {
  auto* b = fb_of(buf);
  lean_object* arr = lean_alloc_array(b->len, b->len);
  for (size_t i = 0; i < b->len; ++i) {
    lean_array_set_core(arr, i, lean_box_float(b->data[i]));
  }
  return lean_io_result_mk_ok(arr);
}

// -- Truncation (for pre-roll sliding window) -------------------------------

lean_object* lean_float_buffer_keep_last(lean_object* buf_obj, b_lean_obj_arg count) {
  auto* b = fb_of(buf_obj);
  // A heap-allocated Nat exceeds the maximum possible buffer length.
  if (!lean_is_scalar(count)) return lean_io_result_mk_ok(buf_obj);
  size_t n = lean_unbox(count);
  if (n >= b->len) return lean_io_result_mk_ok(buf_obj);
  size_t drop = b->len - n;
  std::memmove(b->data, b->data + drop, sizeof(double) * n);
  b->len = n;
  return lean_io_result_mk_ok(buf_obj);
}

// -- Bulk construction from raw float FIFO (used by audio input) ------------
// This is exposed so audio_input_read_buffer can call it.

lean_object* tyr_float_buffer_from_raw(const float* src, size_t n) {
  auto* b = fb_alloc(n);
  if (!b) lean_internal_panic_out_of_memory();
  b->len = n;
  for (size_t i = 0; i < n; ++i)
    b->data[i] = static_cast<double>(src[i]);
  return fb_wrap(b);
}

} // extern "C"
