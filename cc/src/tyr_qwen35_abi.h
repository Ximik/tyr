#pragma once
#include <lean/lean.h>

// Audited against Tyr/RawTensor/Qwen35.c.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_qwen35_chunk_gated_delta_rule(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_qwen35_recurrent_gated_delta_rule(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_einsum2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_torch_topk_2d(uint64_t, uint64_t, lean_object*, uint64_t, uint64_t);
lean_object* lean_torch_tensor_softplus(lean_object*, lean_object*);
lean_object* lean_torch_media_load_image_patchified(lean_object*, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_media_load_video_patchified(lean_object*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
#ifdef __cplusplus
}
#endif
