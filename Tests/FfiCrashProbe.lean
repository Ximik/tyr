/-
Exercise Tyr's two fatal failure paths. Selected by the first argument:
- `libtorch`: an unguarded native call raises a libtorch error, reported by
  Tyr's fatal exception handler.
- `admission`: a binding's `Tensor.ofRaw!` receives a handle that contradicts
  its claimed spec and aborts with the native tensor contract report, naming
  the binding.
- `admission-ok`: the matching claim is admitted and the process exits normally.
The failing modes must abort on CPU and need no accelerator.
-/
import Tyr.Typed

open torch

namespace Tests.FfiCrashProbe

/-- A binding whose declared result shape contradicts the native result. -/
def wrongShapeBinding (m : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU }) :
    Tensor { shape := #[3], dtype := .Float32, device := .CPU } :=
  Tensor.ofRaw! m.raw

end Tests.FfiCrashProbe

def main (args : List String) : IO UInt32 := do
  let m : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU } := Tensor.ones #[2, 2]
  match args with
  | ["libtorch"] =>
    IO.println "deliberately reshaping four raw elements into three..."
    (← IO.getStdout).flush
    let invalid := RawTensor.reshape! m.raw #[3]
    IO.println s!"unexpected reshape success: {← RawTensor.toFloatArray invalid}"
    pure 1
  | ["admission"] =>
    IO.println "deliberately admitting a [2, 2] handle as [3]..."
    (← IO.getStdout).flush
    let wrong := Tests.FfiCrashProbe.wrongShapeBinding m
    IO.println s!"unexpected admission success: {← wrong.toFloatArray}"
    pure 1
  | ["admission-ok"] =>
    let same : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU } :=
      Tensor.ofRaw! m.raw
    IO.println s!"admitted: {← same.toFloatArray}"
    pure 0
  | _ =>
    IO.eprintln "usage: ffi_crash_probe (libtorch | admission | admission-ok)"
    pure 2
