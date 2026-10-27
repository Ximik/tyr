/-
Deliberately violate a raw native reshape contract to exercise Tyr's fatal
exception report. This probe must abort on CPU and does not need an accelerator.
-/
import Tyr.Typed

open torch

def main : IO Unit := do
  let m : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU } := Tensor.ones #[2, 2]
  IO.println "deliberately reshaping four raw elements into three..."
  (← IO.getStdout).flush
  let invalid := RawTensor.reshape m.raw #[3]
  let arr ← RawTensor.toFloatArray invalid
  IO.println s!"unexpected reshape success: {arr}"
