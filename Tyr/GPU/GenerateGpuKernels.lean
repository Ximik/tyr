import Lean
import Tyr.GPU.Codegen.FFI

open Lean

namespace Tyr.GPU.Codegen

structure CliConfig where
  outDir : System.FilePath := ⟨"cc/src/generated"⟩
  clean : Bool := true
  modules : Array Name := #[]
  stubsOut : Option System.FilePath := none
  deriving Inhabited

def usage : String :=
  String.intercalate "\n" [
    "Usage: lake exe GenerateGpuKernels [OPTIONS] <Module> [<Module> ...]",
    "",
    "Positional arguments:",
    "  <Module>                 Lean module to import (e.g. Tyr.GPU.Kernels.Examples)",
    "",
    "Options:",
    "  --out-dir <path>         Output directory for generated .cu files (default: cc/src/generated)",
    "  --no-clean               Keep existing generated .cu files in output directory",
    "  --stubs <path>           Instead of CUDA, write placeholder launchers for every",
    "                           kernel in the modules (used by builds without the kernels' CUDA)",
    "  --help                   Show this help"
  ]

def parseModuleName (s : String) : Except String Name := do
  let parts := (s.splitOn ".").filter (!·.isEmpty)
  if parts.isEmpty then
    throw s!"Invalid module name '{s}'"
  return parts.foldl (init := Name.anonymous) fun n p => Name.str n p

partial def parseArgs (cfg : CliConfig) : List String → Except String CliConfig
  | [] =>
    if cfg.modules.isEmpty then
      throw "At least one module is required."
    else
      return cfg
  | "--out-dir" :: dir :: rest =>
    parseArgs { cfg with outDir := ⟨dir⟩ } rest
  | "--out-dir" :: [] =>
    throw "--out-dir expects a path argument."
  | "--no-clean" :: rest =>
    parseArgs { cfg with clean := false } rest
  | "--stubs" :: path :: rest =>
    parseArgs { cfg with stubsOut := some ⟨path⟩ } rest
  | "--stubs" :: [] =>
    throw "--stubs expects a path argument."
  | "--help" :: _ =>
    return cfg
  | arg :: rest =>
    if arg.startsWith "-" then
      throw s!"Unknown option '{arg}'"
    else
      match parseModuleName arg with
      | .error err => throw err
      | .ok moduleName =>
        parseArgs { cfg with modules := cfg.modules.push moduleName } rest

private def runCoreWithEnv (env : Environment) (x : CoreM α) : IO α := do
  let ctx : Core.Context := {
    fileName := "<gpu-kernel-generate>"
    fileMap := default
  }
  let st : Core.State := { env := env }
  x.toIO' ctx st

unsafe def evalKernelConstExpr (constName : Name) : CoreM Kernel := do
  withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := 0 }) do
    Lean.Meta.MetaM.run' do
      let info ← getConstInfo constName
      let value ← match info with
        | .defnInfo info => pure info.value
        | .thmInfo info => pure info.value
        | _ => throwError "Kernel companion '{constName}' is not reducible."
      Lean.Meta.evalExpr Kernel (mkConst ``Kernel) value

unsafe def materializeKernelRefs (env : Environment)
    (refs : Array KernelCompanionRef) : IO (Array RegisteredKernelSpec) := do
  runCoreWithEnv env do
    refs.mapM fun r => do
      try
        let kernel ← evalKernelConstExpr r.kernelConst
        pure (mkRegisteredKernelSpec r.moduleName r.name kernel)
      catch e =>
        throwError "Failed to evaluate kernel '{r.kernelConst}' for '{r.name}': {e.toMessageData}"

/-- The C symbols of all GPU kernel launchers (`@[extern "lean_launch_…"]`) in
    `env`, sorted. -/
def kernelLauncherSymbols (env : Environment) : Array String :=
  let symbols := env.constants.fold (init := #[]) fun acc name _ =>
    match getExternNameFor env `c name with
    | some sym => if sym.startsWith "lean_launch_" then acc.push sym else acc
    | none => acc
  (symbols.qsort (· < ·)).foldl (init := #[]) fun acc sym =>
    if acc.back? == some sym then acc else acc.push sym

/-- Placeholder launchers, linked when a kernel's CUDA is not compiled (CPU
    builds, or kernels not in `-Kkernels`). They are weak, so a compiled
    kernel's launcher replaces them. -/
def renderLauncherStubs (symbols : Array String) : String :=
  let header := [
    "#include <lean/lean.h>",
    "#include <string>",
    "",
    "static lean_object* gpuKernelUnavailable(const char* launcher) {",
    "  std::string msg = \"GPU kernel launcher unavailable in this build (missing NVCC/CUDA): \";",
    "  msg += launcher;",
    "  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));",
    "}",
    "",
    "extern \"C\" {",
    ""]
  let stubs := symbols.toList.flatMap fun sym => [
    s!"__attribute__((weak)) lean_object* {sym}(...) \{",
    s!"  return gpuKernelUnavailable(\"{sym}\");",
    "}",
    ""]
  "\n".intercalate (header ++ stubs ++ ["} // extern \"C\"", ""])

/-- Write `text` unless the file already has it, so Make does not recompile. -/
def writeIfChanged (path : System.FilePath) (text : String) : IO Unit := do
  if (← path.pathExists) then
    if (← IO.FS.readFile path) == text then return
  if let some dir := path.parent then IO.FS.createDirAll dir
  IO.FS.writeFile path text

unsafe def main (args : List String) : IO UInt32 := do
  if args.contains "--help" then
    IO.println usage
    return (0 : UInt32)

  match parseArgs {} args with
  | .error err =>
    IO.eprintln err
    IO.eprintln ""
    IO.eprintln usage
    return (1 : UInt32)
  | .ok cfg =>
    try
      Lean.initSearchPath (← Lean.findSysroot)
      -- Match LeanTest-style driver behavior: extension initializers must run
      -- so imported modules can register entries during import.
      Lean.enableInitializersExecution
      let imports := cfg.modules.map (fun m => ({ module := m } : Import))
      let env ← Lean.importModules imports {} (loadExts := true)
      if let some stubsOut := cfg.stubsOut then
        let symbols := kernelLauncherSymbols env
        writeIfChanged stubsOut (renderLauncherStubs symbols)
        IO.println s!"Wrote {symbols.size} kernel launcher stub(s) to {stubsOut}"
        return (0 : UInt32)
      let refs := collectKernelCompanionRefsFromEnvModules env cfg.modules
      let regs ← materializeKernelRefs env refs
      setRegisteredKernels regs
      let written ← writeKernelCudaUnitsByModuleFrom regs cfg.outDir (clean := cfg.clean)
      IO.println s!"Generated {written.size} CUDA translation unit(s) in {cfg.outDir}"
      for path in written do
        IO.println s!"  {path}"
      return (0 : UInt32)
    catch e =>
      IO.eprintln s!"Kernel generation failed: {e.toString}"
      return (1 : UInt32)

end Tyr.GPU.Codegen

unsafe def main : List String → IO UInt32 :=
  Tyr.GPU.Codegen.main
