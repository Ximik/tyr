import LeanTest
import Tyr.Typed.Neural

namespace Tests.DTypeUnary

open torch

private def expectError {α : Type} (result : Except String α) : IO Unit :=
  match result with
  | .ok _ => LeanTest.fail "Expected a checked unary-operation rejection"
  | .error _ => pure ()

private def magnitude (t : Tensor ⟨#[1], .Complex64, .CPU⟩) : Float :=
  (Tensor.maxAll (Tensor.abs t)).item

example (x : Tensor ⟨#[2], .Complex64, .CPU⟩) : Tensor ⟨#[2], .Complex64, .CPU⟩ :=
  Tensor.exp (Tensor.sqrt (Tensor.log (Tensor.sin (Tensor.cos (Tensor.atan x)))))

example (x : Tensor ⟨#[2], .Complex128, .CPU⟩) : Tensor ⟨#[2], .Complex128, .CPU⟩ :=
  Tensor.rsqrt (Tensor.tanh (Tensor.sigmoid (Tensor.pow (Tensor.log10 x) 2)))

example {σ : StaticSpec} (input : Tensor σ) :
    Except String (Tensor (σ.withDType input.actualSpec.dtype.atLeastFloat)) :=
  Tensor.Promoting.sqrt? input

example (_x : Tensor ⟨#[2], .Complex64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.gelu _x
  fail_if_success have _ := Tensor.silu _x
  fail_if_success have _ := Tensor.softmax _x
  trivial

example (_x : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.exp _x
  fail_if_success have _ := Tensor.sqrt _x
  fail_if_success have _ := Tensor.log _x
  trivial

@[test] def testComplexSquareRootAndLogarithmBranches : IO Unit := do
  let negative : Tensor ⟨#[1], .Complex64, .CPU⟩ :=
    (Tensor.fromFloatArray #[-1]).castDType .Complex64
  let one := (Tensor.ones #[1]).castDType .Complex64
  let root := Tensor.sqrt negative
  let logarithm := Tensor.log negative
  LeanTest.assertEqual root.raw.dtype .Complex64
  LeanTest.assertEqual logarithm.raw.dtype .Complex64
  LeanTest.assertTrue ((magnitude root - 1).abs < 1e-6)
  LeanTest.assertTrue (magnitude (Tensor.pow root 2 - negative) < 1e-5)
  LeanTest.assertTrue (magnitude (root * Tensor.rsqrt negative - one) < 1e-5)
  LeanTest.assertTrue ((magnitude logarithm - 3.141592653589793).abs < 1e-5)
  LeanTest.assertTrue (magnitude (Tensor.exp logarithm - negative) < 1e-5)
  LeanTest.assertTrue
    ((magnitude (Tensor.log10 negative) - 3.141592653589793 / Float.log 10).abs < 1e-5)
  LeanTest.assertTrue root.validate.toOption.isSome
  LeanTest.assertTrue (root.raw.device == negative.raw.device)

@[test] def testComplexTrigonometricAndActivationValues : IO Unit := do
  let imaginary : Tensor ⟨#[1], .Complex64, .CPU⟩ :=
    Tensor.sqrt ((Tensor.fromFloatArray #[-1]).castDType .Complex64)
  let one := (Tensor.ones #[1]).castDType .Complex64
  let opposite := Tensor.mulScalar imaginary (-1)
  let sine := Tensor.sin imaginary
  let cosine := Tensor.cos imaginary
  LeanTest.assertTrue (magnitude (sine * sine + cosine * cosine - one) < 1e-5)
  LeanTest.assertTrue ((magnitude sine - (Float.exp 1 - Float.exp (-1)) / 2).abs < 1e-5)
  LeanTest.assertTrue
    (magnitude (Tensor.sigmoid imaginary + Tensor.sigmoid opposite - one) < 1e-5)
  LeanTest.assertTrue ((magnitude (Tensor.tanh imaginary) - Float.tan 1).abs < 1e-5)
  LeanTest.assertTrue
    ((magnitude (Tensor.atan (Tensor.mulScalar imaginary 0.5)) - Float.log 3 / 2).abs < 1e-5)

@[test] def testPreservingUnaryRejectsErasedIntegralInputs : IO Unit := do
  let input := ((Tensor.fromFloatArray #[1, 4]).castDType .Int64).forgetDType
  for result in #[Tensor.Preserving.exp? input, Tensor.Preserving.sqrt? input,
      Tensor.Preserving.rsqrt? input, Tensor.Preserving.log? input,
      Tensor.Preserving.log10? input, Tensor.Preserving.sin? input,
      Tensor.Preserving.cos? input, Tensor.Preserving.atan? input,
      Tensor.Preserving.sigmoid? input, Tensor.Preserving.tanh? input,
      Tensor.Preserving.pow? input 0.5, Tensor.Preserving.gelu? input,
      Tensor.Preserving.silu? input, Tensor.Preserving.elu? input,
      Tensor.Preserving.softmax? input, Tensor.logSoftmax? input] do
    expectError result
  LeanTest.assertEqual input.raw.dtype .Int64
  LeanTest.assertEqual (← input.toFloatArray) #[1, 4]

@[test] def testUnaryBackendAndDomainErrorsAreRecoverable : IO Unit := do
  let complex := ((Tensor.ones #[2]).castDType .Complex64).forgetDType
  expectError (Tensor.gelu? complex)
  expectError (Tensor.silu? complex)
  expectError (Tensor.elu? complex)
  expectError (Tensor.softmax? complex)
  expectError (Tensor.logSoftmax? complex)
  expectError (Tensor.softmax? (Tensor.ones #[2]) 9)
  expectError (Tensor.logSoftmax? (Tensor.ones #[2]) 9)
  let shell := (Tensor.ones #[2]).castDType .Float8E4M3FN
  expectError (Tensor.sqrt? shell)

@[test] def testRuntimeUnaryPromotionOfErasedIntegers : IO Unit := do
  let input := ((Tensor.fromFloatArray #[1, 4]).castDType .Int32).forgetDType
  let cases := #[(#[1, 2], Tensor.Promoting.sqrt? input),
    (#[0, Float.log 4], Tensor.Promoting.log? input),
    (#[Float.cos 1, Float.cos 4], Tensor.Promoting.cos? input),
    (#[1, 2], Tensor.Promoting.pow? input 0.5),
    (#[Float.exp 1, Float.exp 4], Tensor.Promoting.exp? input),
    (#[1, 0.5], Tensor.Promoting.rsqrt? input),
    (#[Float.sin 1, Float.sin 4], Tensor.Promoting.sin? input),
    (#[Float.atan 1, Float.atan 4], Tensor.Promoting.atan? input),
    (#[0, Float.log 4 / Float.log 10], Tensor.Promoting.log10? input)]
  for (expected, pending) in cases do
    match pending with
    | .error message => LeanTest.fail message
    | .ok result =>
      LeanTest.assertEqual result.raw.dtype .Float32
      LeanTest.assertTrue (result.raw.device == input.raw.device)
      LeanTest.assertTrue result.validate.toOption.isSome
      let values ← result.toFloatArray
      LeanTest.assertEqual values.size expected.size
      for (actual, value) in values.zip expected do
        LeanTest.assertTrue ((actual - value).abs < 1e-4)
  LeanTest.assertEqual (Tensor.Promoting.sqrt input).raw.dtype .Float32
  LeanTest.assertEqual input.raw.dtype .Int32
  LeanTest.assertEqual (← input.toFloatArray) #[1, 4]

@[test] def testRuntimeUnaryPromotionRetainsComplexDtype : IO Unit := do
  let input := ((Tensor.fromFloatArray #[-1]).castDType .Complex64).forgetDType
  match Tensor.Promoting.sqrt? input with
  | .error message => LeanTest.fail message
  | .ok result =>
    LeanTest.assertEqual result.raw.dtype .Complex64
    LeanTest.assertTrue (result.raw.device == input.raw.device)
    let magnitudes ← (Tensor.abs result).toFloatArray
    LeanTest.assertEqual magnitudes #[1]
  LeanTest.assertEqual (Tensor.Promoting.log input).raw.dtype .Complex64

end Tests.DTypeUnary
