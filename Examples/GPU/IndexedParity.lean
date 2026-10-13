import Examples.GPU.Parity
import Tyr.Typed.Ops

/-! Shape-indexed parity metrics for generated-kernel runners. -/
namespace Examples.GPU.IndexedParity
open torch

/-- Compare equal shapes across storage dtypes, computing GPU parity metrics in Float32. -/
def compareTensors {s : Shape} {expectedDtype actualDtype : DType}
    (label : String) (expected : DTensor s expectedDtype) (actual : DTensor s actualDtype)
    (rtol atol : Float) : TensorCheck :=
  let expected := expected.toFloat32
  let actual := actual.toFloat32
  let ok := Tensor.allclose expected actual rtol atol
  let diff := (actual - expected).abs
  let mae := diff.meanAll.item
  let maxErr := diff.maxAll.item
  { label, ok, mae, maxErr, rtol, atol }

def reportTensorComparison {s : Shape} {expectedDtype actualDtype : DType}
    (label : String) (expected : DTensor s expectedDtype) (actual : DTensor s actualDtype)
    (rtol atol : Float) : IO Bool := do
  let check := compareTensors label expected actual rtol atol
  logTensorCheck check
  pure check.ok

end Examples.GPU.IndexedParity
