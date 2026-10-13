#pragma once
#include <lean/lean.h>

// Audited against Tyr/RawTensor/Gemma4.c.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_gemma4_text_experts_forward(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_sdpa_gqa_mask_qkv(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, double, uint8_t);
lean_object* lean_torch_softcap(lean_object*, lean_object*, double);
lean_object* lean_torch_media_load_image_patch_grid_gemma4(lean_object*, uint64_t, uint64_t, uint64_t, double);
lean_object* lean_torch_avg_pool2d(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
#ifdef __cplusplus
}
#endif
