import LeanTest
import Tyr.Typed.Neural

namespace Tests.DTypeUnary

open torch

private def magnitude (t : Tensor ⟨#[1], .Complex64, .CPU⟩) : Float :=
  (Tensor.maxAll (Tensor.abs t)).item

example (x : Tensor ⟨#[2], .Complex64, .CPU⟩) : Tensor ⟨#[2], .Complex64, .CPU⟩ :=
  Tensor.exp (Tensor.sqrt (Tensor.log (Tensor.sin (Tensor.cos (Tensor.atan x)))))

example (x : Tensor ⟨#[2], .Complex128, .CPU⟩) : Tensor ⟨#[2], .Complex128, .CPU⟩ :=
  Tensor.rsqrt (Tensor.tanh (Tensor.sigmoid (Tensor.pow (Tensor.log10 x) 2)))

example (x : Tensor ⟨#[2], .Float32, .CPU⟩) : Tensor ⟨#[2], .Float32, .CPU⟩ :=
  Tensor.elu (Tensor.gelu (Tensor.silu x))

example (_x : Tensor ⟨#[2], .Complex64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.gelu _x
  fail_if_success have _ := Tensor.silu _x
  fail_if_success have _ := Tensor.elu _x
  fail_if_success have _ := Tensor.softmax _x
  trivial

example (_x : Tensor ⟨#[2], .Int64, .CPU⟩) : True := by
  fail_if_success have _ := Tensor.exp _x
  fail_if_success have _ := Tensor.sqrt _x
  fail_if_success have _ := Tensor.rsqrt _x
  fail_if_success have _ := Tensor.sigmoid _x
  fail_if_success have _ := Tensor.tanh _x
  fail_if_success have _ := Tensor.atan _x
  fail_if_success have _ := Tensor.pow _x 2
  fail_if_success have _ := Tensor.gelu _x
  fail_if_success have _ := Tensor.silu _x
  fail_if_success have _ := Tensor.elu _x
  fail_if_success have _ := Tensor.log _x
  fail_if_success have _ := Tensor.log10 _x
  fail_if_success have _ := Tensor.sin _x
  fail_if_success have _ := Tensor.cos _x
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

end Tests.DTypeUnary
