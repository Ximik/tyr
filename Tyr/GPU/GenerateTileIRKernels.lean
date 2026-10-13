import Lean
import Tyr.GPU.Codegen.TileIR.Attribute
import Tyr.GPU.Codegen.TileIR.Specialization
import Tyr.GPU.Codegen.TileIR.Toolchain

open Lean

namespace Tyr.GPU.Codegen.TileIR

structure KernelSpecialization where
  declSelector : String
  rawValues : Array String
  deriving Repr, Inhabited

structure CliConfig where
  outDir : System.FilePath := ⟨"build/tileir"⟩
  modules : Array Name := #[]
  outputKind : OutputKind := .cubin
  gpuName : String := "sm_100"
  optLevel : Nat := 3
  lineInfo : Bool := true
  normalizeMlir : Bool := true
  specializations : Array KernelSpecialization := #[]
  deriving Inhabited

def usage : String :=
  String.intercalate "\n" [
    "Usage: lake exe GenerateTileIRKernels [OPTIONS] <Module> [<Module> ...]",
    "",
    "Positional arguments:",
    "  <Module>                 Lean module to import (e.g. Tyr.GPU.Codegen.TileIR.Examples)",
    "",
    "Options:",
    "  --out-dir <path>         Output directory for generated artifacts (default: build/tileir)",
    "  --kind <mlir|bytecode|cubin>",
    "                           Final artifact to emit (default: cubin)",
    "  --gpu-name <name>        Target GPU name for tileiras (default: sm_100)",
    "  --opt-level <0-3>        Optimization level passed to tileiras (default: 3)",
    "  --no-line-info           Disable tileiras line info",
    "  --no-normalize           Skip cuda-tile-opt before bytecode generation",
    "  --specialize <decl>=<v1[,v2,...]>",
    "                           Instantiate a ct.Const kernel for export; repeatable",
    "  --help                   Show this help"
  ]

def parseModuleName (s : String) : Except String Name := do
  let parts := (s.splitOn ".").filter (!·.isEmpty)
  if parts.isEmpty then
    throw s!"Invalid module name '{s}'"
  pure <| parts.foldl (init := Name.anonymous) fun n p => Name.str n p

private def parseOutputKind (s : String) : Except String OutputKind :=
  match s with
  | "mlir" => .ok .mlir
  | "bytecode" => .ok .bytecode
  | "cubin" => .ok .cubin
  | _ => .error s!"Unknown TileIR output kind '{s}'"

private def parseSpecialization (s : String) : Except String KernelSpecialization := do
  let parts := s.splitOn "="
  if parts.length != 2 then
    throw s!"Invalid specialization '{s}'. Expected <decl>=<value[,value,...]>"
  let selector := parts[0]!.trimAscii.toString
  let rawValues := (parts[1]!.splitOn ",").toArray.map (fun v => v.trimAscii.toString)
  if selector.isEmpty then
    throw s!"Invalid specialization '{s}': missing declaration selector"
  if rawValues.isEmpty then
    throw s!"Invalid specialization '{s}': expected at least one const argument value"
  if rawValues.any (fun v => v.isEmpty) then
    throw s!"Invalid specialization '{s}': empty const argument values are not allowed"
  pure { declSelector := selector, rawValues := rawValues }

partial def parseArgs (cfg : CliConfig) : List String → Except String CliConfig
  | [] =>
      if cfg.modules.isEmpty then
        throw "At least one module is required."
      else
        pure cfg
  | "--out-dir" :: dir :: rest =>
      parseArgs { cfg with outDir := ⟨dir⟩ } rest
  | "--out-dir" :: [] =>
      throw "--out-dir expects a path argument."
  | "--kind" :: kind :: rest =>
      match parseOutputKind kind with
      | .ok outputKind => parseArgs { cfg with outputKind := outputKind } rest
      | .error err => throw err
  | "--kind" :: [] =>
      throw "--kind expects one of: mlir, bytecode, cubin."
  | "--gpu-name" :: gpuName :: rest =>
      parseArgs { cfg with gpuName := gpuName } rest
  | "--gpu-name" :: [] =>
      throw "--gpu-name expects a target such as sm_100."
  | "--opt-level" :: level :: rest =>
      match level.toNat? with
      | some optLevel => parseArgs { cfg with optLevel := optLevel } rest
      | none => throw s!"Invalid optimization level '{level}'"
  | "--opt-level" :: [] =>
      throw "--opt-level expects a number in [0, 3]."
  | "--no-line-info" :: rest =>
      parseArgs { cfg with lineInfo := false } rest
  | "--no-normalize" :: rest =>
      parseArgs { cfg with normalizeMlir := false } rest
  | "--specialize" :: spec :: rest =>
      match parseSpecialization spec with
      | .ok spec =>
          parseArgs { cfg with specializations := cfg.specializations.push spec } rest
      | .error err =>
          throw err
  | "--specialize" :: [] =>
      throw "--specialize expects <decl>=<value[,value,...]>"
  | "--help" :: _ =>
      pure cfg
  | arg :: rest =>
      if arg.startsWith "-" then
        throw s!"Unknown option '{arg}'"
      else
        match parseModuleName arg with
        | .ok moduleName =>
            parseArgs { cfg with modules := cfg.modules.push moduleName } rest
        | .error err =>
            throw err

private def runCoreWithEnv (env : Environment) (x : CoreM α) : IO α := do
  let ctx : Core.Context := {
    fileName := "<tileir-generate>"
    fileMap := default
  }
  let st : Core.State := { env := env }
  x.toIO' ctx st

unsafe def evalTileModuleConstExpr (constName : Name) : CoreM Module := do
  withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := 0 }) do
    Lean.Meta.MetaM.run' do
      let info ← getConstInfo constName
      let value ← match info with
        | .defnInfo info => pure info.value
        | .thmInfo info => pure info.value
        | _ => throwError "TileIR declaration '{constName}' is not reducible."
      Lean.Meta.evalExpr Module (mkConst ``Module) value

unsafe def evalTileModuleAppliedExpr (constName : Name) (args : Array Expr) : CoreM Module := do
  withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := 0 }) do
    Lean.Meta.MetaM.run' do
      let info ← getConstInfo constName
      let _ ← match info with
        | .defnInfo _ | .thmInfo _ => pure ()
        | _ => throwError "TileIR declaration '{constName}' is not reducible."
      let value := args.foldl (init := mkConst constName) mkApp
      Lean.Meta.evalExpr Module (mkConst ``Module) value

private def kernelExportSortKey (kernel : RegisteredTileKernel) (variant? : Option String := none) : String :=
  match variant? with
  | none => toString kernel.declName
  | some variant => s!"{kernel.declName}::{variant}"

private def parseBool? (s : String) : Option Bool :=
  if s == "true" then
    some true
  else if s == "false" then
    some false
  else
    none

private def mkConstArgExpr (kind : ConstParamKind) (raw : String) : Except String Expr :=
  match kind with
  | .nat =>
      match raw.toNat? with
      | some n => .ok <| mkNatLit n
      | none => .error s!"Expected a Nat specialization, but got '{raw}'"
  | .int =>
      match raw.toInt? with
      | some i => .ok <| mkIntLit i
      | none => .error s!"Expected an Int specialization, but got '{raw}'"
  | .bool =>
      match parseBool? raw with
      | some true => .ok <| mkConst ``Bool.true
      | some false => .ok <| mkConst ``Bool.false
      | none => .error s!"Expected a Bool specialization, but got '{raw}'"

private def sanitizePathSegment (value : String) : String :=
  let chars := value.toList.map fun c =>
    if c.isAlphanum then c else '_'
  let text := String.ofList chars
  if text.isEmpty then "tileir" else text

private def kernelMatchesSelector (kernel : RegisteredTileKernel) (selector : String) : Bool :=
  let full := toString kernel.declName
  full == selector || full.endsWith s!".{selector}"

private def variantLabel? (rawValues : Array String) : Option String :=
  if rawValues.isEmpty then
    none
  else
    some <| String.intercalate "__" rawValues.toList

private def kernelOutDir
    (cfg : CliConfig)
    (kernel : RegisteredTileKernel)
    (variant? : Option String := none)
    : System.FilePath :=
  let declSegment :=
    match variant? with
    | none => sanitizePathSegment (toString kernel.declName)
    | some variant =>
        sanitizePathSegment s!"{kernel.declName}__{variant}"
  cfg.outDir /
    sanitizePathSegment (toString kernel.moduleName) /
    declSegment

private def kernelDisplayName (kernel : RegisteredTileKernel) (variant? : Option String := none) : String :=
  match variant? with
  | none => toString kernel.declName
  | some variant => s!"{kernel.declName}[{variant}]"

unsafe def materializeTileModules
    (env : Environment)
    (kernels : Array RegisteredTileKernel)
    : IO (Array (RegisteredTileKernel × Module)) := do
  runCoreWithEnv env do
    kernels.mapM fun kernel => do
      try
        let mod ← evalTileModuleConstExpr kernel.declName
        pure (kernel, mod)
      catch e =>
        throwError m!"Failed to evaluate TileIR declaration '{kernel.declName}' from module '{kernel.moduleName}': {e.toMessageData}"

unsafe def materializeSpecializedTileModules
    (env : Environment)
    (kernel : RegisteredTileKernel)
    (rawSpecs : Array (Array String))
    : IO (Array (RegisteredTileKernel × Option String × Module)) := do
  runCoreWithEnv env do
    let info ← getConstInfo kernel.declName
    let some kinds ← Lean.Meta.MetaM.run' do
      recoverConstParamKinds? info.type
      | throwError m!"Failed to recover ct.Const parameter kinds for TileIR declaration '{kernel.declName}'."
    let mut out : Array (RegisteredTileKernel × Option String × Module) := #[]
    for rawValues in rawSpecs do
      if rawValues.size != kinds.size then
        throwError m!"TileIR specialization for '{kernel.declName}' expected {kinds.size} ct.Const value(s), but got {rawValues.size}: {repr rawValues}"
      let mut args : Array Expr := #[]
      for i in [0:kinds.size] do
        match mkConstArgExpr kinds[i]! rawValues[i]! with
        | .ok arg =>
            args := args.push arg
        | .error err =>
            throwError err
      try
        let mod ← evalTileModuleAppliedExpr kernel.declName args
        out := out.push (kernel, variantLabel? rawValues, mod)
      catch e =>
        throwError m!"Failed to evaluate specialized TileIR declaration '{kernel.declName}' with values {repr rawValues}: {Exception.toMessageData e}"
    pure out

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
        Lean.enableInitializersExecution
        let imports :=
          #[({ module := `Tyr.GPU.Codegen.TileIR.Attribute } : Import)] ++
          cfg.modules.map (fun m => ({ module := m } : Import))
        let env ← Lean.importModules imports {} (loadExts := true)
        let kernels :=
          collectRegisteredTileKernelsFromModules env cfg.modules
            |>.qsort (fun a b => kernelExportSortKey a < kernelExportSortKey b)
        if kernels.isEmpty then
          throw <| IO.userError "No @[tileir_kernel] declarations were found in the requested modules."
        for spec in cfg.specializations do
          let matchedKernels := kernels.filter (fun kernel => kernelMatchesSelector kernel spec.declSelector)
          if matchedKernels.isEmpty then
            throw <| IO.userError s!"No selected @[tileir_kernel] declaration matched specialization selector '{spec.declSelector}'."
          if matchedKernels.size > 1 then
            let names := String.intercalate ", " <| matchedKernels.toList.map (fun kernel => toString kernel.declName)
            throw <| IO.userError s!"Specialization selector '{spec.declSelector}' is ambiguous. It matches: {names}"
        let readyKernels := kernels.filter (·.constParamCount == 0)
        let parameterizedKernels := kernels.filter (·.constParamCount > 0)
        let mut missingSpecializations : Array RegisteredTileKernel := #[]
        let mut matchedSelectors : Array String := #[]
        let mut specialized : Array (RegisteredTileKernel × Option String × Module) := #[]
        for kernel in parameterizedKernels do
          let specs :=
            cfg.specializations.filter (fun spec => kernelMatchesSelector kernel spec.declSelector)
          if specs.isEmpty then
            missingSpecializations := missingSpecializations.push kernel
          else
            matchedSelectors := matchedSelectors ++ specs.map (·.declSelector)
            specialized := specialized ++ (← materializeSpecializedTileModules env kernel (specs.map (·.rawValues)))
        for spec in cfg.specializations do
          unless matchedSelectors.contains spec.declSelector do
            throw <| IO.userError s!"Specialization selector '{spec.declSelector}' did not resolve to a parameterized @[tileir_kernel] declaration."
        if !missingSpecializations.isEmpty then
          let details :=
            String.intercalate "\n" <| missingSpecializations.toList.map fun kernel =>
              s!"  - {kernel.declName} ({kernel.constParamCount} ct.Const parameter(s))"
          throw <| IO.userError s!"The following selected @[tileir_kernel] declarations need --specialize values:\n{details}"
        let materializedDirect ← materializeTileModules env readyKernels
        let materialized := materializedDirect.map (fun (kernel, mod) => (kernel, (none : Option String), mod)) ++ specialized
        if materialized.isEmpty then
          throw <| IO.userError "All selected @[tileir_kernel] declarations require ct.Const specialization. Add --specialize <decl>=<value[,value,...]>."
        let toolchain ← detectToolchain
        let opts : CompileOptions := {
          gpuName := cfg.gpuName
          optLevel := cfg.optLevel
          lineInfo := cfg.lineInfo
          normalizeMlir := cfg.normalizeMlir
          outputKind := cfg.outputKind
        }
        for (kernel, variant?, mod) in materialized do
          let outDir := kernelOutDir cfg kernel variant?
          match ← compileModuleAt mod outDir opts toolchain with
          | .ok paths =>
              let output := paths.outputPath cfg.outputKind
              IO.println s!"Compiled {kernelDisplayName kernel variant?} -> {output}"
          | .error err =>
              throw <| IO.userError s!"TileIR compilation failed for {kernelDisplayName kernel variant?}: {err}"
        return (0 : UInt32)
      catch e =>
        IO.eprintln s!"TileIR generation failed: {e}"
        return (1 : UInt32)

end Tyr.GPU.Codegen.TileIR

unsafe def main : List String → IO UInt32 :=
  Tyr.GPU.Codegen.TileIR.main
