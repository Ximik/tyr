import LeanTest
import Tyr.Typed.Neural

open torch

namespace Tests.TestFFIBoundary

private def values {σ : StaticSpec} (t : Tensor σ) : IO (Array Float) :=
  t.toFloatArray

/-- Exercise boxed Lean Int/Nat arguments through the real native bridge. -/
@[test]
def testBoxedIndexAndUnbind : IO Unit := do
  let x := (Tensor.arangeFrom 0 6).reshapeChecked #[2, 3]
  LeanTest.assertEqual (← values (x.getOp 0)) #[0.0, 1.0, 2.0]
  LeanTest.assertEqual (← values (x.getOp 1)) #[3.0, 4.0, 5.0]
  LeanTest.assertEqual (← values (x.getOp (-1))) #[3.0, 4.0, 5.0]
  let rows := Tensor.unbind x 0
  LeanTest.assertEqual rows.size 2
  LeanTest.assertEqual rows[0]!.runtimeShape #[3]
  LeanTest.assertEqual (← values rows[1]!) #[3.0, 4.0, 5.0]
  let columns := Tensor.unbind x 1
  LeanTest.assertEqual columns.size 3
  LeanTest.assertEqual columns[0]!.runtimeShape #[2]
  LeanTest.assertEqual (← values columns[2]!) #[2.0, 5.0]

@[test]
def testArangeLengthAndScalarWidths : IO Unit := do
  let cases : Array (UInt64 × UInt64 × UInt64) :=
    #[(0, 5, 2), (0, 6, 2), (2, 2, 1), (0, 1, 4), (4294967296, 4294967299, 1)]
  for (start, stop, step) in cases do
    let x := Tensor.arangeFrom start stop step
    LeanTest.assertEqual x.runtimeShape #[Tensor.rangeLength start stop step]
  LeanTest.assertEqual (← values (Tensor.arangeFrom 0 5 2)) #[0.0, 2.0, 4.0]
  -- Host arrays use float32 conversion; item preserves these int64 values.
  let wide := Tensor.arangeFrom 4294967296 4294967299
  LeanTest.assertEqual (Tensor.item ((wide.getOp 0).reshapeChecked #[])) 4294967296.0
  LeanTest.assertEqual (Tensor.item ((wide.getOp 1).reshapeChecked #[])) 4294967297.0
  LeanTest.assertEqual (Tensor.item ((wide.getOp 2).reshapeChecked #[])) 4294967298.0
  LeanTest.assertEqual (Tensor.eye 2).runtimeShape #[2, 2]
  LeanTest.assertEqual (← values (Tensor.linspace 0.0 1.0 3)) #[0.0, 0.5, 1.0]
  LeanTest.assertEqual (← values (Tensor.logspace 0.0 2.0 3)) #[1.0, 10.0, 100.0]

@[test]
def testExactScalarReshape : IO Unit := do
  let x := Tensor.ones #[1]
  let scalar : Tensor { shape := #[], dtype := .Float32, device := .CPU } := x.reshape #[]
  LeanTest.assertEqual scalar.runtimeShape #[]
  LeanTest.assertEqual scalar.item 1.0

@[test]
def testExactLoadRejectsSameNumelWrongShape : IO Unit :=
  IO.FS.withTempDir fun dir => do
    let path := (dir / "tensor.pt").toString
    Tensor.saveTensor (Tensor.ones #[2, 3]) path
    let exact ← Tensor.loadTensorExact {shape := #[2, 3], dtype := .Float32, device := .CPU} path
    LeanTest.assertEqual exact.runtimeShape #[2, 3]
    try
      let _ ← Tensor.loadTensorExact {shape := #[3, 2], dtype := .Float32, device := .CPU} path
      LeanTest.fail "exact loading must reject a shape mismatch"
    catch error =>
      LeanTest.assertTrue (error.toString.containsSubstr "Tensor shape mismatch")

end Tests.TestFFIBoundary
