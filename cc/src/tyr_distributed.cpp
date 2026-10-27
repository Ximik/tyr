/**
 * tyr_distributed.cpp - Distributed training primitives for Tyr
 *
 * Provides Lean FFI bindings for PyTorch distributed communication:
 * - Process group initialization and management
 * - Collective operations (reduce_scatter, all_gather, all_reduce, broadcast)
 * - Async operation handling with futures
 */

#include <lean/lean.h>
#include "generated/tyr_ffi.h"
#include "tyr_checked_result.h"
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#if defined(USE_C10D_NCCL)
#include <c10/cuda/CUDAFunctions.h>
#endif

#if defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#define TYR_HAS_NCCL 1
#endif

// ProcessGroupGloo.hpp declares nothing unless USE_C10D_GLOO is defined.
#define USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#ifndef TYR_HAS_NCCL
#define TYR_HAS_NCCL 0
#endif

#include <memory>
#include <string>
#include <mutex>
#include <limits>
#include <unordered_map>
#include <stdexcept>

// Forward declarations from tyr.cpp
extern lean_object* fromTorchTensor(torch::Tensor t);
extern torch::Tensor borrowTensor(b_lean_obj_arg o);

namespace {

// Global process group state
c10::intrusive_ptr<c10d::Backend> g_process_group;
c10::intrusive_ptr<c10d::Store> g_store;
int g_rank = 0;
int g_world_size = 1;
bool g_initialized = false;
std::mutex g_dist_mutex;

// Future storage for async operations
std::unordered_map<uint64_t, c10::intrusive_ptr<c10d::Work>> g_pending_work;
std::atomic<uint64_t> g_next_work_id{1};

// Work::wait throws on failure but returns false on abortion. Both outcomes
// must prevent exposing a collective result as successfully completed.
void waitForCompletion(const c10::intrusive_ptr<c10d::Work>& work) {
    if (!work->wait())
        throw std::runtime_error("Distributed operation was aborted");
}

enum class BackendKind { Gloo, Nccl };

// Validate before creating a TCPStore or narrowing UInt64 fields to libtorch's
// native index types. In particular, unavailable backends must not rendezvous.
BackendKind validateInit(const std::string& backend, uint64_t port,
                         uint64_t rank, uint64_t world_size) {
    if (port > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("master_port must be in [0, 65535]");
    if (world_size == 0 || world_size > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("world_size must be in [1, INT_MAX]");
    if (rank >= world_size)
        throw std::invalid_argument("rank must be less than world_size");
    if (backend == "gloo") {
        return BackendKind::Gloo;
    }
    if (backend == "nccl") {
#if TYR_HAS_NCCL
        return BackendKind::Nccl;
#else
        throw std::invalid_argument("NCCL backend requested but ProcessGroupNCCL is unavailable in this build");
#endif
    }
    throw std::invalid_argument("Unsupported backend: " + backend);
}

static_assert(tyr::lean::ReduceOpCount == 5, "reduceOpFromLean must map every Lean ReduceOp");

c10d::ReduceOp::RedOpType reduceOpFromLean(uint8_t op) {
    switch (static_cast<tyr::lean::ReduceOp>(op)) {
        case tyr::lean::ReduceOp::sum: return c10d::ReduceOp::SUM;
        case tyr::lean::ReduceOp::avg: return c10d::ReduceOp::AVG;
        case tyr::lean::ReduceOp::product: return c10d::ReduceOp::PRODUCT;
        case tyr::lean::ReduceOp::min: return c10d::ReduceOp::MIN;
        case tyr::lean::ReduceOp::max: return c10d::ReduceOp::MAX;
    }
    throw std::invalid_argument("Invalid Lean ReduceOp tag: " + std::to_string(op));
}

void maybeSetCurrentCudaDeviceFromTensor(const torch::Tensor& t) {
#if TYR_HAS_NCCL
    if (t.is_cuda()) {
        c10::cuda::set_device(static_cast<c10::DeviceIndex>(t.get_device()));
    }
#else
    (void)t;
#endif
}

} // anonymous namespace

extern "C" {

/**
 * Set the current CUDA device for this process.
 *
 * This should be called with LOCAL_RANK before NCCL collectives to avoid
 * duplicate-device communicator initialization.
 */
lean_object* lean_torch_dist_set_cuda_device(uint64_t device) {
  return tyr::checkedIoCall("lean_torch_dist_set_cuda_device", [&] {
      try {
          if (device > static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max()))
              throw std::invalid_argument("CUDA device index exceeds the native DeviceIndex range");
  #if TYR_HAS_NCCL
          if (!torch::cuda::is_available()) {
              return lean_io_result_mk_ok(lean_box(0));
          }
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
  #else
          (void)device;
  #endif
          return lean_io_result_mk_ok(lean_box(0));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Failed to set CUDA device: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Initialize the distributed process group.
 *
 * @param backend_obj - String: "nccl" or "gloo"
 * @param master_addr_obj - String: Master address (e.g., "localhost")
 * @param master_port - Port number for rendezvous
 * @param rank - This process's rank
 * @param world_size - Total number of processes
 */
lean_object* lean_torch_dist_init_process_group(
    b_lean_obj_arg backend_obj,
    b_lean_obj_arg master_addr_obj,
    uint64_t master_port,
    uint64_t rank,
    uint64_t world_size
) {
  return tyr::checkedIoCall("lean_torch_dist_init_process_group", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group already initialized")));
      }

      try {
          std::string backend(lean_string_cstr(backend_obj));
          std::string master_addr(lean_string_cstr(master_addr_obj));

          const auto kind = validateInit(backend, master_port, rank, world_size);
          const auto native_rank = static_cast<int>(rank);
          const auto native_world_size = static_cast<int>(world_size);

          // Create TCP store for rendezvous
          bool is_master = (rank == 0);
          auto store_opts = c10d::TCPStoreOptions();
          store_opts.port = static_cast<uint16_t>(master_port);
          store_opts.isServer = is_master;
          store_opts.numWorkers = native_world_size;
          store_opts.waitWorkers = true;

          auto store = c10::make_intrusive<c10d::TCPStore>(master_addr, store_opts);
          c10::intrusive_ptr<c10d::Backend> process_group;

          if (kind == BackendKind::Nccl) {
  #if TYR_HAS_NCCL
              auto options = c10d::ProcessGroupNCCL::Options::create();
              options->is_high_priority_stream = true;
              process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
                  store, native_rank, native_world_size, options);
  #endif
          } else {
              auto options = c10d::ProcessGroupGloo::Options::create();
              options->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());
              process_group = c10::make_intrusive<c10d::ProcessGroupGloo>(
                  store, native_rank, native_world_size, options);
          }

          // Local RAII owns both resources until construction has fully succeeded.
          g_store = std::move(store);
          g_process_group = std::move(process_group);
          g_rank = native_rank;
          g_world_size = native_world_size;
          g_initialized = true;
          return lean_io_result_mk_ok(lean_box(0));

      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Failed to init process group: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Get the rank of this process.
 */
lean_object* lean_torch_dist_get_rank() {
  return tyr::checkedIoCall("lean_torch_dist_get_rank", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);
      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }
      return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_rank)));
  });
}

/**
 * Get the world size (total number of processes).
 */
lean_object* lean_torch_dist_get_world_size() {
  return tyr::checkedIoCall("lean_torch_dist_get_world_size", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);
      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }
      return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_world_size)));
  });
}

/**
 * Check if distributed is initialized.
 */
lean_object* lean_torch_dist_is_initialized() {
  return tyr::checkedIoCall("lean_torch_dist_is_initialized", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);
      return lean_io_result_mk_ok(lean_box(g_initialized ? 1 : 0));
  });
}

/**
 * Destroy the process group and clean up resources.
 */
lean_object* lean_torch_dist_destroy_process_group() {
  return tyr::checkedIoCall("lean_torch_dist_destroy_process_group", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);
      // Attempt every wait, including after an earlier failure. Always release
      // residual state, even when initialization never reached its commit point.
      std::exception_ptr first_error;
      for (auto& [id, work] : g_pending_work) {
          try {
              if (work) waitForCompletion(work);
          } catch (...) {
              if (!first_error) first_error = std::current_exception();
          }
      }
      g_pending_work.clear();
      g_process_group.reset();
      g_store.reset();
      g_initialized = false;
      g_rank = 0;
      g_world_size = 1;
      if (first_error) std::rethrow_exception(first_error);
      return lean_io_result_mk_ok(lean_box(0));
  });
}

/**
 * Barrier - synchronize all processes.
 */
lean_object* lean_torch_dist_barrier() {
  return tyr::checkedIoCall("lean_torch_dist_barrier", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          c10d::BarrierOptions opts;
          auto work = g_process_group->barrier(opts);
          waitForCompletion(work);
          return lean_io_result_mk_ok(lean_box(0));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Barrier failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * All-reduce operation: sum tensors across all processes.
 *
 * @param tensor - Input/output tensor (in-place reduction)
 * @param op - Reduce operation (0=SUM, 1=AVG, 2=PRODUCT, 3=MIN, 4=MAX)
 * @param async_op - If true, returns work ID for async wait
 */
lean_object* lean_torch_dist_all_reduce(
    b_lean_obj_arg tensor,
    uint8_t op,
    uint8_t async_op) {
  return tyr::checkedIoCall("lean_torch_dist_all_reduce", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          auto t = borrowTensor(tensor);
          maybeSetCurrentCudaDeviceFromTensor(t);
          std::vector<torch::Tensor> tensors = {t};

          c10d::AllreduceOptions opts;
          opts.reduceOp = c10d::ReduceOp(reduceOpFromLean(op));

          auto work = g_process_group->allreduce(tensors, opts);

          if (async_op) {
              uint64_t work_id = g_next_work_id++;
              g_pending_work[work_id] = work;
              return lean_io_result_mk_ok(lean_box_uint64(work_id));
          } else {
              waitForCompletion(work);
              return lean_io_result_mk_ok(lean_box_uint64(0));
          }
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("All-reduce failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Broadcast tensor from source rank to all other processes.
 *
 * @param tensor - Input/output tensor
 * @param src_rank - Source rank to broadcast from
 */
lean_object* lean_torch_dist_broadcast(
    b_lean_obj_arg tensor,
    uint64_t src_rank) {
  return tyr::checkedIoCall("lean_torch_dist_broadcast", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          if (src_rank >= static_cast<uint64_t>(g_world_size))
              throw std::invalid_argument("broadcast source rank must be less than world_size");
          auto t = borrowTensor(tensor);
          maybeSetCurrentCudaDeviceFromTensor(t);
          std::vector<torch::Tensor> tensors = {t};

          c10d::BroadcastOptions opts;
          opts.rootRank = static_cast<int>(src_rank);

          auto work = g_process_group->broadcast(tensors, opts);
          waitForCompletion(work);

          return lean_io_result_mk_ok(lean_box(0));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Broadcast failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Reduce-scatter: reduce and distribute results to all processes.
 * Each process gets a different portion of the reduced tensor.
 *
 * @param output - Output tensor (portion for this rank)
 * @param input - Input tensor to reduce
 * @param op - Reduce operation
 * @param async_op - If true, returns work ID
 */
lean_object* lean_torch_dist_reduce_scatter_tensor(
    b_lean_obj_arg output,
    b_lean_obj_arg input,
    uint8_t op,
    uint8_t async_op) {
  return tyr::checkedIoCall("lean_torch_dist_reduce_scatter_tensor", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          auto out_tensor = borrowTensor(output);
          auto in_tensor = borrowTensor(input);
          maybeSetCurrentCudaDeviceFromTensor(out_tensor);

          c10d::ReduceScatterOptions opts;
          opts.reduceOp = c10d::ReduceOp(reduceOpFromLean(op));

          // reduce_scatter_tensor expects output and a list of inputs (one per rank)
          std::vector<torch::Tensor> input_tensors;
          // Split input tensor into world_size chunks
          auto chunks = in_tensor.chunk(g_world_size);
          for (auto& chunk : chunks) {
              input_tensors.push_back(chunk);
          }

          std::vector<torch::Tensor> output_tensors = {out_tensor};
          std::vector<std::vector<torch::Tensor>> input_tensor_lists = {input_tensors};
          auto work = g_process_group->reduce_scatter(
              output_tensors,
              input_tensor_lists,
              opts);

          if (async_op) {
              uint64_t work_id = g_next_work_id++;
              g_pending_work[work_id] = work;
              return lean_io_result_mk_ok(lean_box_uint64(work_id));
          } else {
              waitForCompletion(work);
              return lean_io_result_mk_ok(lean_box_uint64(0));
          }
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Reduce-scatter failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Reduce-scatter from an explicit list of input tensors.
 *
 * This matches PyTorch DistMuon semantics where each rank provides a list
 * of per-owner gradients (one tensor per rank in the process group).
 *
 * @param output - Output tensor for this rank
 * @param inputs - Array of input tensors (must have length world_size)
 * @param op - Reduce operation
 */
lean_object* lean_torch_dist_reduce_scatter_list(
    b_lean_obj_arg output,
    b_lean_obj_arg inputs,
    uint8_t op) {
  return tyr::checkedIoCall("lean_torch_dist_reduce_scatter_list", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          auto out_tensor = borrowTensor(output);
          maybeSetCurrentCudaDeviceFromTensor(out_tensor);

          size_t n = lean_array_size(inputs);
          if (n != static_cast<size_t>(g_world_size)) {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("reduce_scatter_list requires inputs length == world size")));
          }

          std::vector<torch::Tensor> input_tensors;
          input_tensors.reserve(n);
          for (size_t i = 0; i < n; i++) {
              auto tensor_obj = lean_array_get_core(inputs, i);
              input_tensors.push_back(borrowTensor(tensor_obj));
          }

          c10d::ReduceScatterOptions opts;
          opts.reduceOp = c10d::ReduceOp(reduceOpFromLean(op));

          std::vector<torch::Tensor> output_tensors = {out_tensor};
          std::vector<std::vector<torch::Tensor>> input_tensor_lists = {input_tensors};
          auto work = g_process_group->reduce_scatter(
              output_tensors,
              input_tensor_lists,
              opts);
          waitForCompletion(work);

          return lean_io_result_mk_ok(lean_box(0));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("Reduce-scatter(list) failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * All-gather: gather tensors from all processes to all processes.
 *
 * @param output - Output tensor (concatenated from all ranks)
 * @param input - Input tensor from this rank
 * @param async_op - If true, returns work ID
 */
lean_object* lean_torch_dist_all_gather_into_tensor(
    b_lean_obj_arg output,
    b_lean_obj_arg input,
    uint8_t async_op) {
  return tyr::checkedIoCall("lean_torch_dist_all_gather_into_tensor", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          auto out_tensor = borrowTensor(output);
          auto in_tensor = borrowTensor(input);
          maybeSetCurrentCudaDeviceFromTensor(in_tensor);

          // Split output into world_size chunks to receive each rank's data
          auto output_chunks = out_tensor.chunk(g_world_size);
          std::vector<torch::Tensor> output_list;
          for (auto& chunk : output_chunks) {
              output_list.push_back(chunk);
          }

          std::vector<std::vector<torch::Tensor>> output_tensor_lists = {output_list};
          std::vector<torch::Tensor> input_tensors = {in_tensor};
          auto work = g_process_group->allgather(
              output_tensor_lists,
              input_tensors);

          if (async_op) {
              uint64_t work_id = g_next_work_id++;
              g_pending_work[work_id] = work;
              return lean_io_result_mk_ok(lean_box_uint64(work_id));
          } else {
              waitForCompletion(work);
              return lean_io_result_mk_ok(lean_box_uint64(0));
          }
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("All-gather failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * All-gather into an explicit list of output tensors.
 *
 * This matches PyTorch DistMuon semantics where each rank gathers one updated
 * owner tensor into a list of per-owner destination tensors.
 *
 * @param outputs - Array of output tensors (must have length world_size)
 * @param input - Input tensor from this rank
 */
lean_object* lean_torch_dist_all_gather_list(
    b_lean_obj_arg outputs,
    b_lean_obj_arg input) {
  return tyr::checkedIoCall("lean_torch_dist_all_gather_list", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      if (!g_initialized) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Process group not initialized")));
      }

      try {
          auto in_tensor = borrowTensor(input);
          maybeSetCurrentCudaDeviceFromTensor(in_tensor);

          size_t n = lean_array_size(outputs);
          if (n != static_cast<size_t>(g_world_size)) {
              return lean_io_result_mk_error(lean_mk_io_user_error(
                  lean_mk_string("all_gather_list requires outputs length == world size")));
          }

          std::vector<torch::Tensor> output_list;
          output_list.reserve(n);
          for (size_t i = 0; i < n; i++) {
              auto tensor_obj = lean_array_get_core(outputs, i);
              output_list.push_back(borrowTensor(tensor_obj));
          }

          std::vector<std::vector<torch::Tensor>> output_tensor_lists = {output_list};
          std::vector<torch::Tensor> input_tensors = {in_tensor};
          auto work = g_process_group->allgather(
              output_tensor_lists,
              input_tensors);
          waitForCompletion(work);

          return lean_io_result_mk_ok(lean_box(0));
      } catch (const std::exception& e) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string(("All-gather(list) failed: " + std::string(e.what())).c_str())));
      }
  });
}

/**
 * Wait for an async operation to complete.
 *
 * @param work_id - Work ID returned from async operation
 */
lean_object* lean_torch_dist_wait(uint64_t work_id) {
  return tyr::checkedIoCall("lean_torch_dist_wait", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      auto it = g_pending_work.find(work_id);
      if (it == g_pending_work.end()) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Invalid work ID")));
      }

      // Remove before waiting so both standard and unknown exceptions release the
      // handle exactly once; the local owning pointer keeps work alive until exit.
      auto work = std::move(it->second);
      g_pending_work.erase(it);
      waitForCompletion(work);
      return lean_io_result_mk_ok(lean_box(0));
  });
}

/**
 * Check if an async operation completed successfully (non-blocking).
 * Polling retains the handle; wait must still synchronize and release it.
 *
 * @param work_id - Work ID
 * @return Bool - true if complete
 */
lean_object* lean_torch_dist_is_completed(uint64_t work_id) {
  return tyr::checkedIoCall("lean_torch_dist_is_completed", [&] {
      std::lock_guard<std::mutex> lock(g_dist_mutex);

      auto it = g_pending_work.find(work_id);
      if (it == g_pending_work.end()) {
          return lean_io_result_mk_error(lean_mk_io_user_error(
              lean_mk_string("Invalid work ID")));
      }

      const auto& work = it->second;
      bool completed = work->isCompleted();
      if (completed && !work->isSuccess()) {
          if (auto error = work->exception()) std::rethrow_exception(error);
          throw std::runtime_error("Distributed operation did not complete successfully");
      }
      return lean_io_result_mk_ok(lean_box(completed ? 1 : 0));
  });
}

} // extern "C"
