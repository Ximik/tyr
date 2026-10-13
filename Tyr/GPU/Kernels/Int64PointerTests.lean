import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Codegen.Int64PointerTests

open Tyr.GPU
open Tyr.GPU.Codegen

#guard (GpuCapabilities.supportsType (arch := .SM80) .Int64) = true
#guard renderGlobalParamCppType
  { name := "targets", dtype := .Int64, isPointer := true } =
  "tyr_raw_gl<int64_t>"

@[gpu_kernel .SM90]
def int64PointerProbe
    (targets : GPtr GpuFloat.Int64)
    (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  emitRaw s!"const int64_t* target_raw = reinterpret_cast<const int64_t*>({targets.id.toIdent}.raw_ptr);"
  emitRaw s!"float* output_raw = reinterpret_cast<float*>({output.id.toIdent}.raw_ptr);"
  emitRaw "if (threadIdx.x == 0 && blockIdx.x == 0) output_raw[0] = static_cast<float>(target_raw[0]);"

#check int64PointerProbe.launch

end Tyr.GPU.Codegen.Int64PointerTests
