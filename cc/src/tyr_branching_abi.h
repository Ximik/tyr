#pragma once
#include <lean/lean.h>

// Audited against Tyr/RawTensor/BranchingFlows.c.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_copy_inplace(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_touch(lean_object*, lean_object*);
lean_object* lean_torch_mul_tensor_scalar(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_tensor_softplus(lean_object*, lean_object*);
lean_object* lean_torch_sdpa_4d_bias(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t);
lean_object* lean_torch_binary_cross_entropy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
