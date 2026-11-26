import Lean

/-! `CheckLayers` reads the imports of every module under `Tyr/` and fails if a
module imports one from a layer to its right in `Core`, `IO`, `Lib`, `Model`,
or a module under `Tyr.Core.FFI` imports one outside it. -/

open Lean

private def layers : Array String := #["Core", "IO", "Lib", "Model"]

/-- The position of `module`'s layer in `layers`; `none` outside `Tyr.<layer>`. -/
private def layerOf? (module : Name) : Option Nat :=
  match module.components with
  | `Tyr :: layer :: _ => layers.findIdx? (· == layer.toString)
  | _ => none

private def inFFI (module : Name) : Bool := (`Tyr.Core.FFI).isPrefixOf module

/-- The imports of `module` that come from a later layer, or from outside
`Tyr.Core.FFI` when `module` is in it. -/
def violations (module : Name) (imports : Array Name) : Array Name :=
  match layerOf? module with
  | none => #[]
  | some layer => imports.filter fun imported =>
      match layerOf? imported with
      | some importedLayer => importedLayer > layer || (inFFI module && !inFFI imported)
      | none => inFFI module && (`Tyr).isPrefixOf imported

private def moduleName (path : System.FilePath) : Name :=
  (path.withExtension "").components.foldl Name.mkStr .anonymous

def main : IO UInt32 := do
  let files ← System.FilePath.walkDir "Tyr"
  let mut failed := false
  for path in files.qsort (·.toString < ·.toString) do
    if path.extension != some "lean" then continue
    let (imports, _, messages) ← Elab.parseImports (← IO.FS.readFile path) path.toString
    if messages.hasErrors then
      IO.eprintln s!"{path}: cannot parse imports"
      failed := true
    let module := moduleName path
    for imported in violations module (imports.map (·.module)) do
      IO.eprintln s!"{module} imports {imported}"
      failed := true
  return if failed then 1 else 0
