#pragma once
#include <lean/lean.h>
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_safetensors_open(lean_object*);
lean_object* lean_torch_safetensors_load_from_handle(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_safetensors_load(lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_safetensors_load_sharded(lean_object*, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
