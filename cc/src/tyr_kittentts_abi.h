#pragma once
#include <lean/lean.h>

// Audited against Tyr/RawTensor/KittenTTS.c.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_leaky_relu(lean_object*, lean_object*, double);
lean_object* lean_torch_tensor_floor(lean_object*, lean_object*);
lean_object* lean_torch_atan2(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_cumsum(lean_object*, lean_object*, uint64_t);
lean_object* lean_torch_interpolate(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_istft_1d(lean_object*, uint64_t, uint64_t, uint64_t, lean_object*, uint8_t, uint8_t, uint64_t);
#ifdef __cplusplus
}
#endif
