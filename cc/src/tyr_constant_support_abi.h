#pragma once
#include <lean/lean.h>

// Both metadata arguments are borrowed. The erased proof payload is Unit.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_acquire_constant_support(lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
