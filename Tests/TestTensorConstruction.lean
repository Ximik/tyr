import LeanTest
import Tests.Devices
import Tyr.Basic

-- Variables used only inside a deliberately failing tactic count as unused.
set_option linter.unusedVariables false

namespace Tests.TensorConstruction

open torch

-- Constructor privacy alone does not stop tactics from constructing a structure.
example {σ : StaticSpec} (_raw : RawTensor) : True := by
  fail_if_success
    have _ : Tensor σ := by constructor; exact _raw
  trivial

example {σ : StaticSpec} (_raw : RawTensor) : True := by
  fail_if_success
    have _ : Tensor σ := by refine ⟨?_⟩; exact _raw
  trivial

example {σ τ : StaticSpec} (_tensor : Tensor σ) (_raw : RawTensor) : True := by
  fail_if_success
    have _ : Tensor τ := _tensor
  fail_if_success
    have _ : Tensor σ := _raw
  fail_if_success
    have _ : Tensor σ := ⟨_raw⟩
  fail_if_success
    have _ : Tensor σ := { _tensor with raw := _raw }
  fail_if_success
    have _ : Tensor τ := { raw := _tensor.raw }
  fail_if_success
    have _ : Tensor σ := by change RawTensor; exact _raw
  fail_if_success
    have _ : Tensor σ := by unfold Tensor; exact _raw
  fail_if_success
    have _ : Tensor τ := by cases _tensor; constructor; assumption
  trivial

/-- error: (kernel) invalid declaration, it uses unsafe declaration 'unsafeCast' -/
#guard_msgs in
def unsafeCastBypass {σ : StaticSpec} (raw : RawTensor) : Tensor σ := unsafeCast raw

-- Deliberately bypass name privacy: safety must come from the unsafe marker,
-- not from whether ordinary source notation can name a private declaration.
elab "tensorPrivateImplementation% " name:ident : term =>
  return Lean.mkConst (Lean.Name.num `_private.Tyr.Basic 0 ++ name.getId)

/-- error: (kernel) invalid declaration, it uses unsafe declaration '_private.Tyr.Basic.0.torch.Tensor.zerosForImpl' -/
#guard_msgs in
def privateImplementationBypass (σ : StaticSpec)
    [DType.ConstantSupported σ.dtype σ.device] : Tensor σ :=
  (tensorPrivateImplementation% torch.Tensor.zerosForImpl) σ

@[test]
def testAdmissionAndOpaqueAccessors : IO Unit := do
  let raw := RawTensor.zeros #[2] false .CPU
  let spec : StaticSpec := { shape := #[2], dtype := .Float32, device := .CPU }
  let tensor : Tensor spec := Tensor.ofRaw! raw
  LeanTest.assertEqual tensor.raw.runtimeShape #[2]
  LeanTest.assertEqual tensor.raw.dtype .Float32
  LeanTest.assertTrue (tensor.raw.device == .CPU)
  LeanTest.assertEqual tensor.raw.getValues.toList [0, 0]
  let packaged := tensor.packDevice
  LeanTest.assertTrue (packaged.1 == Device.CPU)
  LeanTest.assertEqual packaged.2.raw.getValues.toList [0, 0]

@[test]
def testOpaqueDefaultAllocation : IO Unit := do
  let tensor : Tensor { shape := #[3], dtype := .Float64, device := .CPU } := default
  LeanTest.assertEqual tensor.raw.runtimeShape #[3]
  LeanTest.assertEqual tensor.raw.dtype .Float64
  LeanTest.assertEqual tensor.raw.getValues.toList [0, 0, 0]

end Tests.TensorConstruction
