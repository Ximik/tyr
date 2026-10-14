#pragma once
#include <lean/lean.h>

// Keep these declarations identical to Lean's generated C. Nat arguments are
// boxed objects, pure Float results are unboxed doubles, and IO has no world
// parameter at this native boundary.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_float_buffer_mk_empty(lean_object*);
lean_object* lean_float_buffer_size(lean_object*);
double lean_float_buffer_get(lean_object*, lean_object*);
lean_object* lean_float_buffer_rms(lean_object*);
lean_object* lean_float_buffer_push(lean_object*, double);
lean_object* lean_float_buffer_append(lean_object*, lean_object*);
lean_object* lean_float_buffer_append_array(lean_object*, lean_object*);
lean_object* lean_float_buffer_clear(lean_object*);
lean_object* lean_float_buffer_keep_last(lean_object*, lean_object*);
lean_object* lean_float_buffer_to_array(lean_object*);
#ifdef __cplusplus
}
#endif
