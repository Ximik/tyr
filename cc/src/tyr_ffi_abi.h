#pragma once

#include <lean/lean.h>

// Audited tensor creation, shape/index, and linear3d boundaries. Keep these declarations
// identical to Lean's generated C prototypes; scripts/check_ffi_abi.py checks
// the Lean side, and including this header checks the C++ definitions.
// Lean Int/Nat are objects; Int64/UInt64 are unboxed uint64_t values.
#ifdef __cplusplus
extern "C" {
#endif

lean_object* lean_torch_arange(uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_eye(uint64_t, uint8_t);
lean_object* lean_torch_linspace(double, double, uint64_t, uint8_t);
lean_object* lean_torch_logspace(double, double, uint64_t, double, uint8_t);
lean_object* lean_torch_linear3d(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_unbind(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_reshape_exact(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_load_tensor_exact(lean_object*, lean_object*);

#ifdef __cplusplus
}
#endif
