import Lean

/-!
Write the C prototypes of every `@[extern]` declared in the given modules, as
Lean's compiler sees them. `libTyrC` includes the result, so a native
definition whose signature disagrees with its Lean declaration fails to compile.

It also writes `tyr::lean::DType`, `tyr::lean::DeviceKind` and
`tyr::lean::ReduceOp`, C++ enums of those inductives' constructor tags, so
native code never hard-codes them.

With `--check-exports`, it instead fails if the library exports a `lean_*`
symbol that no `@[extern]` in the given modules declares: the header only
checks definitions Lean knows about, so an orphaned one would go unchecked.

Usage: `GenerateFfiHeader <out.h> <Module> [<Module> ...]`
       `GenerateFfiHeader --check-exports <lib> <Module> [<Module> ...]`
-/

open Lean IR

/-- Mirrors `Lean.IR.EmitC.emitFnDeclAux` for extern declarations. -/
def prototype (env : Environment) (cName : String) (decl : Decl) : String :=
  let ret := EmitC.toCType decl.resultType
  if decl.params.isEmpty then
    s!"extern {ret} {cName};"
  else
    let ps := decl.params.filter (!·.ty.isVoid)
    let ps := if isExternC env decl.name then ps.filter (!·.ty.isErased) else ps
    let args := ps.map fun p =>
      if p.ty.isObj then (if p.borrow then "b_lean_obj_arg" else "lean_obj_arg")
      else EmitC.toCType p.ty
    s!"{ret} {cName}({", ".intercalate args.toList});"

/-- The extern declarations of `modules`, with their C names. -/
def externs (env : Environment) (modules : NameSet) : Array (Name × String) := Id.run do
  let mut result := #[]
  for (name, _) in env.constants.map₁.toList do
    let some idx := env.getModuleIdxFor? name | continue
    unless modules.contains env.header.moduleNames[idx.toNat]! do continue
    let some cName := getExternNameFor env `c name | continue
    result := result.push (name, cName)
  return result

def prototypes (env : Environment) (modules : NameSet) : Except String (Array String) := do
  let mut byName : Std.HashMap String String := {}
  for (name, cName) in externs env modules do
    let some decl := findEnvDecl env name
      | throw s!"{name}: extern without compiled declaration"
    let proto := prototype env cName decl
    if let some other := byName[cName]? then
      if other != proto then
        throw s!"{cName}: conflicting declarations\n  {other}\n  {proto}"
    byName := byName.insert cName proto
  return byName.toArray.qsort (·.1 < ·.1) |>.map (·.2)

/-- `tyr::lean::<Name>`: the constructor tags of `name` in declaration order,
and `tyr::lean::<Name>Count`, the number of constructors. -/
def tagEnum (env : Environment) (name : Name) : Except String String := do
  let some (.inductInfo info) := env.find? name
    | throw s!"{name}: not an inductive type"
  let tags := info.ctors.mapIdx fun tag ctor => s!"  {ctor.getString!} = {tag},"
  let typeName := name.getString!
  return s!"// Constructor tags of `{name}`.\nenum class {typeName} : unsigned \{\n" ++
    "\n".intercalate tags ++ "\n};\n" ++
    s!"inline constexpr unsigned {typeName}Count = {info.ctors.length};"

/-- The `lean_*` symbols that the shared library `lib` defines and exports. -/
def exportedLeanSymbols (lib : String) : IO (Array String) := do
  let args := if System.Platform.isOSX then #["-gU", lib] else #["-D", "--defined-only", lib]
  let out ← IO.Process.output { cmd := "nm", args }
  unless out.exitCode == 0 do
    throw <| IO.userError s!"nm {lib} failed: {out.stderr}"
  return (out.stdout.splitOn "\n").toArray.filterMap fun line =>
    let symbol := ((line.splitOn " ").getLast?.getD "")
    let symbol := if System.Platform.isOSX then toString (symbol.drop 1) else symbol
    if symbol.startsWith "lean_" then some symbol else none

def checkExports (env : Environment) (modules : NameSet) (lib : String) : IO UInt32 := do
  let declared := (externs env modules).foldl (fun set (_, cName) => set.insert cName)
    (∅ : Std.HashSet String)
  let orphans := (← exportedLeanSymbols lib).filter (!declared.contains ·)
  if orphans.isEmpty then return 0
  IO.eprintln s!"GenerateFfiHeader: {lib} exports symbols that no Lean @[extern] declares. \
    Declare each in a TyrFFI module, or make it internal (static, or not named lean_*):"
  for symbol in orphans.qsort (· < ·) do IO.eprintln s!"  {symbol}"
  return 1

def main (args : List String) : IO UInt32 := do
  let (check?, out, modules) ← match args with
    | "--check-exports" :: lib :: modules@(_ :: _) => pure (true, lib, modules)
    | out :: modules@(_ :: _) => pure (false, out, modules)
    | _ =>
      IO.eprintln "Usage: GenerateFfiHeader [--check-exports <lib> | <out.h>] <Module> [<Module> ...]"
      return 1
  let modules := modules.map String.toName
  initSearchPath (← findSysroot)
  let env ← importModules (modules.toArray.map ({ module := · })) {}
  if check? then return ← checkExports env (modules.foldl NameSet.insert {}) out
  let generated := do
    let protos ← prototypes env (modules.foldl NameSet.insert {})
    let enums ← #[`torch.DType, `torch.DeviceKind, `torch.dist.ReduceOp].mapM (tagEnum env)
    return (protos, enums)
  match generated with
  | .error message => IO.eprintln s!"GenerateFfiHeader: {message}"; return 1
  | .ok (protos, enums) =>
    let lines := #["// Generated by GenerateFfiHeader from Lean's compiled extern declarations.",
      "#pragma once", "#include <lean/lean.h>", "", "#ifdef __cplusplus", "extern \"C\" {",
      "#endif"] ++ protos ++ #["#ifdef __cplusplus", "}", "", "namespace tyr::lean {"] ++
      enums ++ #["}", "#endif"]
    let text := "\n".intercalate lines.toList ++ "\n"
    let out : System.FilePath := out
    -- Rewriting identical content would make Make recompile every includer.
    if (← out.pathExists) then
      if (← IO.FS.readFile out) == text then return 0
    if let some dir := out.parent then IO.FS.createDirAll dir
    IO.FS.writeFile out text
    return 0
