import Lean

/-! `CheckLayers` fails if a module under `Tyr/` imports one from a layer to its
right in `Core`, `IO`, `Lib`, `Model`, if a module under `Tyr.Core.FFI` imports
one outside it, or if a module under `Tyr/`, `Tests/`, `Examples/` or `Tools/`
outside `Tyr.Core.FFI` declares an `extern` constant. The extern check reads each
compiled module, so it runs after the build; a module without an `.olean` is
reported and skipped. -/

open Lean

private def layers : Array String := #["Core", "IO", "Lib", "Model"]

private def sourceRoots : Array System.FilePath := #["Tyr", "Tests", "Examples", "Tools"]

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

/-- The number of `extern` attribute entries the compiled `module` exports. -/
def externCount (module : Name) : IO Nat := do
  let (data, region) ← readModuleData (← findOLean module)
  let count := (data.entries.find? (·.1 == externAttr.ext.name)).map (·.2.size) |>.getD 0
  unsafe region.free
  return count

def main : IO UInt32 := do
  let mut failed := false
  let mut built : Array Name := #[]
  initSearchPath (← findSysroot)
  for root in sourceRoots do
    let files ← System.FilePath.walkDir root
    for path in files.qsort (·.toString < ·.toString) do
      if path.extension != some "lean" then continue
      let module := moduleName path
      if ← (← findOLean module).pathExists then built := built.push module
      else IO.eprintln s!"{module}: not built, extern check skipped"
      if root != "Tyr" then continue
      let (imports, _, messages) ← Elab.parseImports (← IO.FS.readFile path) path.toString
      if messages.hasErrors then
        IO.eprintln s!"{path}: cannot parse imports"
        failed := true
      for imported in violations module (imports.map (·.module)) do
        IO.eprintln s!"{module} imports {imported}"
        failed := true
  for module in built do
    if inFFI module then continue
    let count ← externCount module
    if count > 0 then
      IO.eprintln s!"{module} declares {count} extern constants outside Tyr.Core.FFI"
      failed := true
  return if failed then 1 else 0
