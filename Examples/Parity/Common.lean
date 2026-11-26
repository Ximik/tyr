/-
  Examples/Parity/Common.lean

  Command line, request and result files shared by the parity probes that
  `parity/run.py` launches. Every probe takes
  `--request <file> --weights <dir> --device <device> --out <dir>`, reads the
  JSON request, and writes `result.json` and, when it has tensors,
  `tensors.safetensors` into the output directory.
-/
import Tyr.Core.Torch
import Tyr.IO.SafeTensors.File

namespace Examples.Parity

open Lean torch

structure Args where
  request : System.FilePath
  weights : System.FilePath
  device : DeviceKind
  out : System.FilePath

/-- What a probe writes: integers go to `result.json`, real-valued tensors to
`tensors.safetensors`. -/
structure Output where
  json : Json
  tensors : Array (String × RawTensor) := #[]

private def flags : List String := ["--request", "--weights", "--device", "--out"]

def parseArgs (argv : List String) : Except String Args := do
  let rec collect : List String → List (String × String) → Except String (List (String × String))
    | [], acc => pure acc
    | flag :: value :: rest, acc =>
      if flags.contains flag && !(acc.any (·.1 == flag)) then collect rest ((flag, value) :: acc)
      else throw s!"unexpected argument '{flag}'"
    | [flag], _ => throw s!"missing value for '{flag}'"
  let given ← collect argv []
  let get (flag : String) : Except String String :=
    match given.lookup flag with
    | some value => pure value
    | none => throw s!"missing {flag}"
  let device ← get "--device"
  let some kind := DeviceKind.ofString? device
    | throw s!"invalid --device '{device}'"
  pure { request := ← get "--request", weights := ← get "--weights", device := kind, out := ← get "--out" }

/-- Parse the command line, run `body` on the acquired device and the parsed
request, and write its output. -/
def run (argv : List String) (body : Args → Device → Json → IO Output) : IO UInt32 := do
  let args ← match parseArgs argv with
    | .ok args => pure args
    | .error message =>
      IO.eprintln s!"{message}\nusage: --request <file> --weights <dir> --device <device> --out <dir>"
      return 2
  let device ← Device.acquire args.device
  let request ← IO.ofExcept (Json.parse (← IO.FS.readFile args.request))
  let output ← body args device request
  IO.FS.createDirAll args.out
  IO.FS.writeFile (args.out / "result.json") output.json.compress
  unless output.tensors.isEmpty do
    RawTensor.saveSafeTensors (args.out / "tensors.safetensors").toString output.tensors
  return 0

end Examples.Parity
