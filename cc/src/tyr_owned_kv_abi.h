#pragma once
#include <lean/lean.h>

// Compare with Tyr/RawTensor/OwnedKV.lean generated C. Lean 4.29 erases the
// IO world argument. Shapes are owned; clone borrows its source. Slice copy
// consumes its destination, borrows its source, and returns IO RawTensor.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_clone_inference_io(lean_object*, lean_object*);
lean_object* lean_torch_copy_slice_io(lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, lean_object*);
#ifdef __cplusplus
}
#endif
