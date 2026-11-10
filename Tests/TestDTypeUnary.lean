import LeanTest
import Tyr.Typed.Neural

namespace Tests.DTypeUnary

open torch

private def magnitude (t : Tensor ⟨#[1], .Complex64, .CPU⟩) : Float :=
  (Tensor.maxAll (Tensor.abs t)).item

-- Floating or complex unary operations: complex dtypes are accepted, integers are not.
example (_values : Tensor ⟨#[2], .Complex64, .CPU⟩) (_ints : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  have _ := Tensor.exp _values
  fail_if_success have _ := Tensor.exp _ints
  have _ := Tensor.log _values
  fail_if_success have _ := Tensor.log _ints
  have _ := Tensor.log10 _values
  fail_if_success have _ := Tensor.log10 _ints
  have _ := Tensor.sqrt _values
  fail_if_success have _ := Tensor.sqrt _ints
  have _ := Tensor.rsqrt _values
  fail_if_success have _ := Tensor.rsqrt _ints
  have _ := Tensor.pow _values 2
  fail_if_success have _ := Tensor.pow _ints 2
  have _ := Tensor.sin _values
  fail_if_success have _ := Tensor.sin _ints
  have _ := Tensor.cos _values
  fail_if_success have _ := Tensor.cos _ints
  have _ := Tensor.atan _values
  fail_if_success have _ := Tensor.atan _ints
  have _ := Tensor.tanh _values
  fail_if_success have _ := Tensor.tanh _ints
  have _ := Tensor.sigmoid _values
  fail_if_success have _ := Tensor.sigmoid _ints
  trivial

-- Real floating activations: neither complex nor integer dtypes are accepted.
example (_values : Tensor ⟨#[2], .Float32, .CPU⟩) (_complex : Tensor ⟨#[2], .Complex64, .CPU⟩)
    (_ints : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  have _ := Tensor.gelu _values
  fail_if_success have _ := Tensor.gelu _complex
  fail_if_success have _ := Tensor.gelu _ints
  have _ := Tensor.silu _values
  fail_if_success have _ := Tensor.silu _complex
  fail_if_success have _ := Tensor.silu _ints
  have _ := Tensor.elu _values
  fail_if_success have _ := Tensor.elu _complex
  fail_if_success have _ := Tensor.elu _ints
  have _ := Tensor.softmax _values
  fail_if_success have _ := Tensor.softmax _complex
  fail_if_success have _ := Tensor.softmax _ints
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

end Tests.DTypeUnary
