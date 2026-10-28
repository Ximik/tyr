#include "tyr_nat.h"
#include "generated/tyr_ffi.h"
/*
 * tyr_media_stub.cpp
 *
 * Cross-platform stubs for Apple-only media preprocessing FFI entrypoints.
 * Linux/other non-Apple builds must still provide these symbols because Lean
 * references them unconditionally from extern declarations.
 */

#include <lean/lean.h>

#ifndef __APPLE__
extern "C" {

lean_object* lean_torch_media_load_image_patchified(
    b_lean_obj_arg /*path_obj*/,
    b_lean_obj_arg /*in_channels*/,
    b_lean_obj_arg /*patch_size*/,
    b_lean_obj_arg /*temporal_patch_size*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Apple media path requires macOS build")));
}

lean_object* lean_torch_media_load_video_patchified(
    b_lean_obj_arg /*path_obj*/,
    b_lean_obj_arg /*in_channels*/,
    b_lean_obj_arg /*patch_size*/,
    b_lean_obj_arg /*temporal_patch_size*/,
    b_lean_obj_arg /*max_frames*/,
    b_lean_obj_arg /*frame_stride*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Apple media path requires macOS build")));
}

LEAN_EXPORT lean_object* lean_torch_media_load_image_patch_grid_gemma4(
    b_lean_obj_arg /*path_obj*/,
    b_lean_obj_arg /*patch_size*/,
    b_lean_obj_arg /*pooling_kernel_size*/,
    b_lean_obj_arg /*max_soft_tokens*/,
    double /*rescale_factor*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Apple media path requires macOS build")));
}

} // extern "C"
#endif
