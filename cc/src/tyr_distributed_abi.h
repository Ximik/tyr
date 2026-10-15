#pragma once
#include <lean/lean.h>

// Complete ABI for Tyr/Distributed.lean, including its polar kernels.
// IO world arguments are erased; every heap input below is borrowed.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_dist_wait(uint64_t);
lean_object* lean_torch_dist_init_process_group(lean_object*, lean_object*, uint64_t, uint64_t, uint64_t);
lean_object* lean_torch_dist_set_cuda_device(uint64_t);
lean_object* lean_torch_dist_get_rank();
lean_object* lean_torch_dist_get_world_size();
lean_object* lean_torch_dist_is_initialized();
lean_object* lean_torch_dist_destroy_process_group();
lean_object* lean_torch_dist_barrier();
lean_object* lean_torch_dist_all_reduce(lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_dist_broadcast(lean_object*, uint64_t);
lean_object* lean_torch_dist_reduce_scatter_tensor(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_torch_dist_reduce_scatter_list(lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_dist_all_gather_into_tensor(lean_object*, lean_object*, uint8_t);
lean_object* lean_torch_dist_all_gather_list(lean_object*, lean_object*);
lean_object* lean_torch_dist_is_completed(uint64_t);
lean_object* lean_torch_xxt(lean_object*);
lean_object* lean_torch_ba_plus_cAA(lean_object*, double, double);
lean_object* lean_torch_newton_schulz_step(lean_object*, double, double, double);
lean_object* lean_torch_polar_express(lean_object*, uint64_t);
lean_object* lean_torch_muon_orthogonalize(lean_object*, uint64_t);
lean_object* lean_torch_cautious_update(lean_object*, lean_object*, double, double);
#ifdef __cplusplus
}
#endif
