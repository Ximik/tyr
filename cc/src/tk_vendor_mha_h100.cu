#include <torch/csrc/utils/pybind.h>

#define TORCH_COMPILE
#define TORCH_EXTENSION_NAME tyr_tk_vendor_unused

struct DummyPybindModule {
  template <typename... Args>
  void def(Args&&...) const {}
};

#ifdef PYBIND11_MODULE
#undef PYBIND11_MODULE
#endif
#define PYBIND11_MODULE(name, variable) static void ignored_pybind_module(DummyPybindModule& variable)

#include "../../external/git/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu"

#undef PYBIND11_MODULE
#undef TORCH_EXTENSION_NAME
#undef TORCH_COMPILE

std::vector<at::Tensor>
tyr_tk_attention_forward_nosync(at::Tensor q, at::Tensor k, at::Tensor v, bool causal) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto is_causal = causal;
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(q.size(0) == batch && k.size(0) == batch && v.size(0) == batch,
        "TK vendored forward: Q/K/V batch mismatch");
    TORCH_CHECK(q.size(2) == seq_len && k.size(2) == seq_len && v.size(2) == seq_len,
        "TK vendored forward: Q/K/V sequence mismatch");
    TORCH_CHECK(q.size(3) == head_dim && k.size(3) == head_dim && v.size(3) == head_dim,
        "TK vendored forward: Q/K/V head-dim mismatch");
    TORCH_CHECK(qo_heads >= kv_heads && qo_heads % kv_heads == 0,
        "TK vendored forward: invalid Q/KV head ratio");
    TORCH_CHECK(head_dim == 64, "TK vendored forward nosync currently supports head_dim=64 only");
    TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4) == 0,
        "TK vendored forward nosync requires seq_len divisible by 192");

    auto hr = qo_heads / kv_heads;

    bf16* d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());

    at::Tensor o = at::empty(
        {static_cast<long>(batch), static_cast<long>(qo_heads),
         static_cast<long>(seq_len), static_cast<long>(head_dim)},
        v.options());
    at::Tensor l_vec = at::empty(
        {static_cast<long>(batch), static_cast<long>(qo_heads),
         static_cast<long>(seq_len), 1L},
        q.options().dtype(at::kFloat));

    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float* d_l = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using globals = fwd_globals<64>;
    using q_global = typename globals::q_gl;
    using k_global = typename globals::k_gl;
    using v_global = typename globals::v_gl;
    using l_global = typename globals::l_gl;
    using o_global = typename globals::o_gl;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                    1U, static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};

    globals g{
        qg_arg,
        kg_arg,
        vg_arg,
        lg_arg,
        og_arg,
        static_cast<int>(seq_len),
        static_cast<int>(hr)
    };

    auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;
    dim3 grid(seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4), qo_heads, batch);

    if (is_causal) {
        cudaFuncSetAttribute(fwd_attend_ker<64, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        fwd_attend_ker<64, true><<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    } else {
        cudaFuncSetAttribute(fwd_attend_ker<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        fwd_attend_ker<64, false><<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {o, l_vec};
}

std::vector<at::Tensor>
tyr_tk_attention_backward_nosync(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor l_vec,
    at::Tensor og,
    bool causal) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto is_causal = causal;

    TORCH_CHECK(q.size(0) == batch && k.size(0) == batch && v.size(0) == batch &&
                l_vec.size(0) == batch && o.size(0) == batch && og.size(0) == batch,
        "TK vendored backward: batch mismatch");
    TORCH_CHECK(q.size(2) == seq_len && k.size(2) == seq_len && v.size(2) == seq_len &&
                l_vec.size(2) == seq_len && o.size(2) == seq_len && og.size(2) == seq_len,
        "TK vendored backward: sequence mismatch");
    TORCH_CHECK(q.size(3) == head_dim && k.size(3) == head_dim && v.size(3) == head_dim &&
                o.size(3) == head_dim && og.size(3) == head_dim,
        "TK vendored backward: head-dim mismatch");
    TORCH_CHECK(qo_heads >= kv_heads && qo_heads % kv_heads == 0,
        "TK vendored backward: invalid Q/KV head ratio");
    TORCH_CHECK(head_dim == 64, "TK vendored backward nosync currently supports head_dim=64 only");
    TORCH_CHECK(seq_len % (4 * BWD_CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16>) == 0,
        "TK vendored backward nosync requires seq_len divisible by 128");

    auto hr = qo_heads / kv_heads;

    bf16* d_q  = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k  = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_o  = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16* d_og = reinterpret_cast<bf16*>(og.data_ptr<c10::BFloat16>());
    float* d_l = reinterpret_cast<float*>(l_vec.data_ptr<float>());

    at::Tensor qg = at::zeros(
        {static_cast<long>(batch), static_cast<long>(qo_heads),
         static_cast<long>(seq_len), static_cast<long>(head_dim)},
        l_vec.options());
    at::Tensor kg = at::zeros(
        {static_cast<long>(batch), static_cast<long>(kv_heads),
         static_cast<long>(seq_len), static_cast<long>(head_dim)},
        l_vec.options());
    at::Tensor vg = at::zeros(
        {static_cast<long>(batch), static_cast<long>(kv_heads),
         static_cast<long>(seq_len), static_cast<long>(head_dim)},
        l_vec.options());
    at::Tensor d_vec = at::empty(
        {static_cast<long>(batch), static_cast<long>(qo_heads),
         static_cast<long>(seq_len), 1L},
        l_vec.options());

    float* d_qg = reinterpret_cast<float*>(qg.data_ptr<float>());
    float* d_kg = reinterpret_cast<float*>(kg.data_ptr<float>());
    float* d_vg = reinterpret_cast<float*>(vg.data_ptr<float>());
    float* d_d  = reinterpret_cast<float*>(d_vec.data_ptr<float>());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto threads = 4 * kittens::WARP_THREADS;
    dim3 grid_bwd(seq_len / (4 * kittens::TILE_ROW_DIM<bf16> * 4), qo_heads, batch);

    using og_tile = st_bf<4 * 16, 64>;
    using o_tile  = st_bf<4 * 16, 64>;
    using d_tile  = col_vec<st_fl<4 * 16, 64>>;
    using og_global = gl<bf16, -1, -1, -1, -1, og_tile>;
    using o_global  = gl<bf16, -1, -1, -1, -1, o_tile>;
    using d_global  = gl<float, -1, -1, -1, -1, d_tile>;
    using bwd_prep_global_args = bwd_prep_globals<64>;

    og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                          static_cast<unsigned int>(seq_len), 64U};
    o_global prep_o_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(seq_len), 64U};
    d_global prep_d_arg{d_d, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                        1U, static_cast<unsigned int>(seq_len)};
    bwd_prep_global_args bwd_prep_g{prep_og_arg, prep_o_arg, prep_d_arg};

    using prep_og_tile = st_bf<64, 64>;
    using prep_o_tile  = st_bf<64, 64>;
    using prep_d_tile  = col_vec<st_fl<64, 64>>;
    size_t prep_mem_size =
        sizeof(prep_og_tile) * 4 + sizeof(prep_o_tile) * 4 + sizeof(prep_d_tile) * 4 + 4096;

    cudaFuncSetAttribute(
        bwd_attend_prep_ker<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        prep_mem_size);
    bwd_attend_prep_ker<64><<<grid_bwd, threads, prep_mem_size, stream>>>(bwd_prep_g);

    using bwd_q_tile    = st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_k_tile    = st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_v_tile    = st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_og_tile   = st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_qg_tile   = st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_kg_tile   = st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_vg_tile   = st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;
    using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;
    using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
    using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
    using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;
    using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;
    using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
    using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
    using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;
    using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
    using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;
    using bwd_global_args = bwd_globals<64>;

    bwd_q_global  bwd_q_arg{d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                            static_cast<unsigned int>(seq_len), 64U};
    bwd_k_global  bwd_k_arg{d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                            static_cast<unsigned int>(seq_len), 64U};
    bwd_v_global  bwd_v_arg{d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                            static_cast<unsigned int>(seq_len), 64U};
    bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_l_global  bwd_l_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                            1U, static_cast<unsigned int>(seq_len)};
    bwd_d_global  bwd_d_arg{d_d, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads),
                            1U, static_cast<unsigned int>(seq_len)};

    bwd_global_args bwd_g{
        bwd_q_arg,
        bwd_k_arg,
        bwd_v_arg,
        bwd_og_arg,
        bwd_qg_arg,
        bwd_kg_arg,
        bwd_vg_arg,
        bwd_l_arg,
        bwd_d_arg,
        static_cast<int>(seq_len),
        static_cast<int>(hr)
    };

    dim3 grid_bwd_2(seq_len / (4 * BWD_CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16>), qo_heads, batch);
    threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

    if (is_causal) {
        cudaFuncSetAttribute(bwd_attend_ker<64, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 117760);
        bwd_attend_ker<64, true><<<grid_bwd_2, threads, 117760, stream>>>(bwd_g);
    } else {
        cudaFuncSetAttribute(bwd_attend_ker<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 117760);
        bwd_attend_ker<64, false><<<grid_bwd_2, threads, 117760, stream>>>(bwd_g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {qg, kg, vg};
}
