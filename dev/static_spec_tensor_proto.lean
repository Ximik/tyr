/- Compile-time examples for the public `torch.Tensor` API, indexed by
   `StaticSpec` (shape, dtype, device). Run with:
   lake env lean dev/static_spec_tensor_proto.lean -/
import Tyr.Typed

namespace Proto

open torch

/-! Public layers preserve their indexed shape, dtype, and device policy. -/

def linearForward3d {inDim outDim batch seq : UInt64} {d : DType} {dev : DevicePolicy}
    (lin : Typed.Linear inDim outDim d dev)
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  lin.forward3d x

def rmsNormForward3d {dim batch seq : UInt64} {d : DType} {dev : DevicePolicy}
    (rn : Typed.RMSNorm dim d dev) (x : Tensor ⟨#[batch, seq, dim], d, dev⟩)
    (float : d.isFloating = true) :
    Tensor ⟨#[batch, seq, dim], DType.promote .Float32 d, dev⟩ :=
  rn.forward3d x float

/-! ============ PROBES ============ -/

-- P1: inferred type of a mixed-dtype broadcast add
#check fun (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3], .BFloat16, .any⟩) =>
  a.addB b

-- P2: inferred type after a chain (matmul then cast)
#check fun (x : Tensor ⟨#[2, 3], .Float32, .any⟩) (w : Tensor ⟨#[3, 4], .Float32, .any⟩) =>
  (x.matmul w).toBFloat16

-- P3: generic dtype-preserving chain over a symbolic spec
def chain {σ : StaticSpec} (x : Tensor σ) : Tensor σ := (x.relu + x.relu).relu

-- P4: ascription against the computed spec (does defeq reduce?)
def ascribed (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3], .Float32, .any⟩) :
    Tensor ⟨#[2, 3], .Float32, .any⟩ :=
  a.addB b

-- P5: device pinning — CUDA-only function rejects a CPU tensor
def cudaOnly (x : Tensor ⟨#[8], .Float32, .exact (.CUDA 0)⟩) : Tensor ⟨#[8], .Float32, .exact (.CUDA 0)⟩ :=
  x.relu

-- E1: incompatible broadcast is rejected.
example (_a : Tensor ⟨#[2, 4], .Float32, .any⟩) (_b : Tensor ⟨#[3, 2], .Float32, .any⟩) : True := by
  fail_if_success
    have _ := _a.addB _b
  trivial

-- E2: mixed-dtype matmul is rejected.
example (_a : Tensor ⟨#[2, 3], .Float32, .any⟩) (_b : Tensor ⟨#[3, 4], .BFloat16, .any⟩) : True := by
  fail_if_success
    have _ := _a.matmul _b
  trivial

-- E3: a CPU tensor cannot be passed to the CUDA-only function.
example (_x : Tensor ⟨#[8], .Float32, .exact .CPU⟩) : True := by
  fail_if_success
    have _ := cudaOnly _x
  trivial

-- E4: the broadcast sum's promoted Float32 dtype cannot be ascribed BFloat16.
example (_a : Tensor ⟨#[2, 3], .Float32, .any⟩) (_b : Tensor ⟨#[3], .BFloat16, .any⟩) : True := by
  fail_if_success
    have _wrongDType : Tensor ⟨#[2, 3], .BFloat16, .any⟩ := _a.addB _b
  trivial

end Proto
