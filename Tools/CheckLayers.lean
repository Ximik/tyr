import Lean

/-! `CheckLayers` fails if a module under `Tyr/` is outside `Tyr.Core`, `Tyr.IO`,
`Tyr.Lib` and `Tyr.Model`, imports a module of this package that is outside
them or in a layer to its right, or has no `.olean`; if a module under
`Tyr.Core.FFI` imports a `Tyr` module outside it; or if a module under `Tyr/`,
`Tests/`, `Examples/` or `Tools/` outside `Tyr.Core.FFI` declares an `extern`
constant. The extern check reads each compiled module, so it runs after the
build; a module of `Tests/`, `Examples/` or `Tools/` without an `.olean` is
reported and skipped. It also fails if `Tyr.lean` imports a `Tyr.Model` module
or does not reach, through imports, every module of `Tyr.Core`, `Tyr.IO` and
`Tyr.Lib` outside `Tyr.Core.FFI`. -/

open Lean

private def layers : Array String := #["Core", "IO", "Lib", "Model"]

private def sourceRoots : Array System.FilePath := #["Tyr", "Tests", "Examples", "Tools"]

/-- The position of `module`'s layer in `layers`; `none` outside `Tyr.<layer>`. -/
private def layerOf? (module : Name) : Option Nat :=
  match module.components with
  | `Tyr :: layer :: _ => layers.findIdx? (· == layer.toString)
  | _ => none

private def inFFI (module : Name) : Bool := (`Tyr.Core.FFI).isPrefixOf module

private def inPackage (module : Name) : Bool :=
  match module.components with
  | root :: _ => sourceRoots.contains root.toString
  | [] => false

/-- The imports of `module`, in layer `layer`, that come from this package and
are outside the layers, in a later layer, or outside `Tyr.Core.FFI` when
`module` is in it. -/
def violations (module : Name) (layer : Nat) (imports : Array Name) : Array Name :=
  imports.filter fun imported =>
    match layerOf? imported with
    | some importedLayer => importedLayer > layer || (inFFI module && !inFFI imported)
    | none => inPackage imported

private def modelLayer : Nat := layers.size - 1

/-- The modules reachable from `root` through `imports`, `root` included. -/
partial def reachable (imports : NameMap (Array Name)) (root : Name) : NameSet :=
  let rec visit (seen : NameSet) (module : Name) : NameSet :=
    if seen.contains module then seen
    else ((imports.find? module).getD #[]).foldl visit (seen.insert module)
  visit {} root

private def moduleName (path : System.FilePath) : Name :=
  (path.withExtension "").components.foldl Name.mkStr .anonymous

/-- The number of `extern` attribute entries the compiled `module` exports. -/
def externCount (module : Name) : IO Nat := do
  let (data, region) ← readModuleData (← findOLean module)
  let count := (data.entries.find? (·.1 == externAttr.ext.name)).map (·.2.size) |>.getD 0
  unsafe region.free
  return count

private def parseImports (path : System.FilePath) : IO (Option (Array Name)) := do
  let (imports, _, messages) ← Elab.parseImports (← IO.FS.readFile path) path.toString
  return if messages.hasErrors then none else some (imports.map (·.module))

def main : IO UInt32 := do
  let mut failed := false
  let mut built : Array Name := #[]
  let mut importsOf : NameMap (Array Name) := {}
  initSearchPath (← findSysroot)
  match ← parseImports "Tyr.lean" with
  | none =>
      IO.eprintln "Tyr.lean: cannot parse imports"
      failed := true
  | some imports =>
      importsOf := importsOf.insert `Tyr imports
      for imported in violations `Tyr (modelLayer - 1) imports do
        IO.eprintln s!"Tyr imports {imported}"
        failed := true
  for root in sourceRoots do
    let files ← System.FilePath.walkDir root
    for path in files.qsort (·.toString < ·.toString) do
      if path.extension != some "lean" then continue
      let module := moduleName path
      if ← (← findOLean module).pathExists then built := built.push module
      else if root == "Tyr" then
        IO.eprintln s!"{module}: not built"
        failed := true
      else IO.eprintln s!"{module}: not built, extern check skipped"
      if root != "Tyr" then continue
      let some layer := layerOf? module
        | IO.eprintln s!"{module} is outside Tyr.Core, Tyr.IO, Tyr.Lib and Tyr.Model"
          failed := true
          continue
      let some imports ← parseImports path
        | IO.eprintln s!"{path}: cannot parse imports"
          failed := true
          continue
      importsOf := importsOf.insert module imports
      for imported in violations module layer imports do
        IO.eprintln s!"{module} imports {imported}"
        failed := true
  let fromTyr := reachable importsOf `Tyr
  for module in (importsOf.toArray.map (·.1)).qsort (·.toString < ·.toString) do
    if (layerOf? module).any (· < modelLayer) && !inFFI module && !fromTyr.contains module then
      IO.eprintln s!"{module} is not imported from Tyr"
      failed := true
  for module in built do
    if inFFI module then continue
    let count ← externCount module
    if count > 0 then
      IO.eprintln s!"{module} declares {count} extern constants outside Tyr.Core.FFI"
      failed := true
  return if failed then 1 else 0
