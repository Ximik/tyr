#pragma once
#include <lean/lean.h>

// Complete runtime-service ABI, audited against generated Tyr/Torch.c.
// IO's world argument is erased. Heap inputs are borrowed, never consumed.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_cuda_is_available();
lean_object* lean_torch_cuda_current_stream();
lean_object* lean_torch_cuda_synchronize();
lean_object* lean_torch_cuda_event_create();
lean_object* lean_torch_cuda_event_record(uint64_t, uint64_t);
lean_object* lean_torch_cuda_event_synchronize(uint64_t);
lean_object* lean_torch_cuda_event_elapsed_ms(uint64_t, uint64_t);
lean_object* lean_torch_cuda_event_destroy(uint64_t);
lean_object* lean_torch_cuda_graph_capture_begin();
lean_object* lean_torch_cuda_graph_capture_end();
lean_object* lean_torch_cuda_graph_replay();
lean_object* lean_torch_cuda_graph_reset();
lean_object* lean_torch_mps_is_available();
lean_object* lean_torch_bin_file_token_count(lean_object*);
lean_object* lean_torch_resample_soxr_hq(lean_object*, uint64_t, uint64_t);
lean_object* lean_torch_wav_begin(lean_object*, uint64_t);
lean_object* lean_torch_wav_finalize(lean_object*);
lean_object* lean_torch_file_exists(lean_object*);
lean_object* lean_torch_get_live_tensors();
lean_object* lean_torch_manual_seed(uint64_t);
#ifdef __cplusplus
}
#endif
