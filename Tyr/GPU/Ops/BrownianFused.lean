/-
  Tyr/GPU/Ops/BrownianFused.lean

  Host side of the device virtual-Brownian-tree increment: builds the
  value-query node schedules (mirroring `Tyr.DiffEq.Brownian.valueAux`'s
  time-driven control flow exactly) and launches the descent kernel.

  Schedule records are six Float32 each — see the encoding note in
  `Tyr.GPU.Kernels.BrownianSample`. All time tags must satisfy
  `t·10⁶ < 2²⁴` for exact f32 representation (trees over [0, ~16] in
  seconds at the CPU sampler's microsecond tag resolution).
-/
import Tyr.Typed.Ops
import Tyr.PRNG
import Tyr.GPU.Kernels.BrownianSample

namespace Tyr.GPU.Ops.BrownianFused

open torch

variable {device : Device}
open Tyr.GPU.Kernels

/-- `timeToUInt32` of the CPU sampler, as an exactly-f32 Float. -/
private def timeTag (t : Float) : Float :=
  Float.ofNat ((t * 1000000.0).toUInt64.toUInt32).toNat

private def record (tagA tagB tagE kind c0 c1 : Float) : Array Float :=
  #[tagA, tagB, tagE, kind, c0, c1]

/-- Node schedule for the value query `w(t)` on a tree over `[T0, T1]` —
    the same control flow as the CPU `valueAux`, with the per-node
    Gaussian scales precomputed. -/
partial def buildValueSchedule (T0 T1 tol : Float) (maxDepth : Nat)
    (t : Float) : Array Float := Id.run do
  let mut out := record (timeTag T0) (timeTag T1) 0.0 0.0
    (Float.sqrt (Float.abs (T1 - T0))) 0.0
  -- top-level clamps mirror valueAux's t ≤ t0 / t ≥ t1 short-circuits
  if t <= T0 then
    return out ++ record (timeTag T0) (timeTag T1) (timeTag t) 3.0 0.0 0.0
  if t >= T1 then
    return out ++ record (timeTag T0) (timeTag T1) (timeTag t) 3.0 0.0 1.0
  let mut t0 := T0
  let mut t1 := T1
  let mut depth := maxDepth
  let mut done := false
  while !done do
    let dt := Float.abs (t1 - t0)
    if depth == 0 || dt <= tol then
      -- bridge terminal
      let dtSpan := t1 - t0
      let dt0 := t - t0
      let dt1 := t1 - t
      let var := (dt0 * dt1) / dtSpan
      let sd := Float.sqrt (Float.abs var)
      let frac := dt0 / dtSpan
      out := out ++ record (timeTag t0) (timeTag t1) (timeTag t) 3.0 sd frac
      done := true
    else
      let mid := (t0 + t1) / 2.0
      let sd := Float.sqrt (Float.abs ((t1 - t0) / 4.0))
      if t == mid then
        out := out ++ record (timeTag t0) (timeTag t1) 0.0 2.0 sd 0.0
        done := true
      else if t < mid then
        out := out ++ record (timeTag t0) (timeTag t1) 0.0 0.0 sd 0.0
        t1 := mid
      else
        out := out ++ record (timeTag t0) (timeTag t1) 0.0 1.0 sd 0.0
        t0 := mid
      depth := depth - 1
  return out

/-- Upload a schedule to the device as a flat Float32 tensor; the result
    shape records the schedule length. -/
def uploadSchedule (sched : Array Float) (device : Device) :
    Tensor { shape := #[sched.size.toUInt64], dtype := .Float32, device := device } :=
  (Tensor.fromFloatArray sched).to device

/-- Per-element key-root for a tensor-valued tree: the same derivation as
    the CPU `VirtualBrownianTreeOps (T s)` instance (array tag). -/
def tensorRootState (treeSeed : UInt64) : UInt64 :=
  (PRNGKey.foldIn (PRNGKey.fromUInt64 treeSeed) 0x56544152).state

/-- Launch the device VBT increment `out[i] = w_i(tB) − w_i(tA)` for `n`
    elements. `out` must be a CUDA Float32 tensor with `n` elements (the dtype
    is now static; placement is still checked by the launcher). -/
def vbtIncrementDevice {s : Shape}
    (treeSeed : UInt64) (T0 T1 tol : Float) (maxDepth : Nat)
    (tA tB : Float)
    (out : Tensor { shape := s, dtype := .Float32, device := device }) (n : UInt64)
    (blackwell : Bool) (stream : UInt64)
    (device : Device) : IO Unit := do
  let schedA := buildValueSchedule T0 T1 tol maxDepth tA
  let schedB := buildValueSchedule T0 T1 tol maxDepth tB
  let lenA := (schedA.size / 6).toUInt64
  let lenB := (schedB.size / 6).toUInt64
  let devA := uploadSchedule schedA device
  let devB := uploadSchedule schedB device
  let root := tensorRootState treeSeed
  -- the kernel's per-thread range strides by blockDim only: one block,
  -- threads stride the elements
  if blackwell then
    vbtIncrementBlackwell.launch out.raw devA.raw devB.raw root n lenA lenB
      1 1 1 256 1 1 0 stream
  else
    vbtIncrement.launch out.raw devA.raw devB.raw root n lenA lenB
      1 1 1 256 1 1 0 stream

end Tyr.GPU.Ops.BrownianFused
