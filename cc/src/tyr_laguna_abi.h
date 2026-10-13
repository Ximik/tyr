#pragma once
#include <lean/lean.h>

// Audited against Tyr/RawTensor/Laguna.c.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_laguna_moe_fp4_forward(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, uint64_t);
uint8_t lean_torch_has_grad_fn(lean_object*, lean_object*);
lean_object* lean_torch_bmm(uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*);
lean_object* lean_torch_scatter_add(lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
