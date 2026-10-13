#pragma once
#include <lean/lean.h>
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_timestep_embedding(uint64_t, lean_object*, uint64_t, double, double);
lean_object* lean_torch_save_ppm(lean_object*, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
