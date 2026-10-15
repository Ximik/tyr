import LeanTest
import Tyr.Typed.Ops
import Tyr.Modular.RiemannianModule

open torch

namespace Tests.DeviceIdentity

example : True := by
  fail_if_success have _ := Tensor {shape := #[2], dtype := .Float32}
  trivial

example (device : Device) : Tensor {shape := #[2], dtype := .Float32, device} :=
  Tensor.ones #[2] false device

example {source : Device} (x : Tensor ⟨#[2], .Float32, source⟩) (target : Device) :
    Tensor ⟨#[2], .Float32, target⟩ := x.to target

example {left right : Device} (_x : Tensor ⟨#[2], .Float32, left⟩)
    (_y : Tensor ⟨#[2], .Float32, right⟩) : True := by
  fail_if_success have _ := _x + _y
  fail_if_success have _ := _x.addB _y
  trivial

-- Generic model/manifold operations must retain arbitrary runtime placement.
example (device : Device) : IO (Linear 2 3 .Float32 device) :=
  Linear.init 2 3 (device := device)

example {device : Device} (model : Linear 2 3 .Float32 device)
    (input : Tensor ⟨#[1, 2], .Float32, device⟩) :
    Tensor ⟨#[1, 3], .Float32, device⟩ := model.forward2d input

example (device : Device) :
    Tyr.Modular.MatrixManifoldCarrier (fun m n => Tyr.AD.Stiefel m n .Float32 device) device :=
  inferInstance

example (device : Device) :
    Tyr.Modular.RiemannianModule (Linear 2 3 .Float32 device) 2 3 device :=
  inferInstance

example {device : Device}
    (lin : Tyr.Modular.LocalLinearization (device := device) 2 6 3) :
    Tensor ⟨#[3, 2], .Float32, device⟩ :=
  Tyr.Modular.LocalLinearization.materializeA lin

-- Equality evidence recovered at runtime permits a shared-device operation.
private def addPacked (x y : Sigma fun device => Tensor ⟨#[2], .Float32, device⟩) :
    Except String (Tensor ⟨#[2], .Float32, x.1⟩) := do
  let other ← y.2.atDevice? x.1
  pure (x.2 + other)

@[test] def testRuntimeDeviceIdentity : IO Unit := do
  let x := Tensor.fromFloatArray #[2, 3]
  let y := Tensor.fromFloatArray #[5, 7]
  let sum ← match addPacked x.packDevice y.packDevice with
    | .ok result => pure result
    | .error message => LeanTest.fail message
  LeanTest.assertEqual (← sum.toFloatArray) #[7, 10]
  match x.atDevice? (.CUDA 7) with
  | .ok _ => LeanTest.fail "CPU tensor acquired CUDA identity without transfer"
  | .error _ => pure ()
  match Tensor.ofRaw (σ := ⟨#[2], .Float32, .CUDA 7⟩) x.raw with
  | .ok _ => LeanTest.fail "raw CPU handle passed CUDA admission"
  | .error _ => pure ()
  let moved : Tensor ⟨#[2], .Float32, .CPU⟩ := x.to .CPU
  LeanTest.assertEqual (← moved.toFloatArray) #[2, 3]
  LeanTest.assertTrue (moved.runtimeDevice == moved.device)

end Tests.DeviceIdentity
