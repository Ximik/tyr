import Lean
import Tyr.GPU.Types
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.EmitNew

/-!
# Tyr.GPU.Codegen.FFI

`Tyr.GPU.Codegen.FFI` generates the interop layer between Lean and emitted GPU kernels.

It covers both directions:

- Lean-side `@[extern]` declarations and call signatures,
- C++ launcher wrappers that marshal runtime tensors/scalars.

This module sits after kernel registration/emission and before integration into
the runtime library build.
-/

namespace Tyr.GPU.Codegen

open Lean Meta Elab Term Command
open Tyr.GPU

/-! ## Lean FFI Declaration Generation -/

/-- Convert kernel name to valid Lean identifier -/
def kernelNameToLeanIdent (name : String) : String :=
  name.replace "." "_"

/-- Convert kernel name to C FFI function name -/
def kernelNameToCFFI (name : String) : String :=
  "lean_launch_" ++ name.replace "." "_"

/-- Generate Lean parameter type for a kernel param -/
def paramToLeanType (p : KParam) : String :=
  if p.isPointer then
    "(@ Tensor)"  -- Borrowed reference to Tensor
  else
    "UInt64"  -- Scalar values as UInt64

/-- Generate Lean @[extern] declaration for a kernel -/
def generateLeanExternDecl (kernel : RegisteredKernel) : String :=
  let funcName := s!"launch{kernelNameToLeanIdent kernel.kernelName}"
  let externName := kernelNameToCFFI kernel.kernelName
  let params := kernel.params.toList.map fun p =>
    s!"({p.name} : {paramToLeanType p})"
  let paramStr := String.intercalate " " params
  s!"@[extern \"{externName}\"]\n" ++
  s!"opaque {funcName} {paramStr} (stream : CudaStream) : IO Unit"

/-- Generate Lean launch wrapper with grid/block config -/
def generateLeanLaunchWrapper (kernel : RegisteredKernel) : String :=
  let funcName := s!"launch{kernelNameToLeanIdent kernel.kernelName}"
  let externName := s!"{funcName}Impl"
  let params := kernel.params.toList.map fun p =>
    s!"({p.name} : {paramToLeanType p})"
  let paramStr := String.intercalate " " params
  let argNames := kernel.params.toList.map (·.name)
  let argStr := String.intercalate " " argNames

  s!"@[extern \"{kernelNameToCFFI kernel.kernelName}\"]\n" ++
  s!"private opaque {externName} {paramStr} (grid block : UInt64 × UInt64 × UInt64) " ++
  s!"(sharedMem : UInt64) (stream : CudaStream) : IO Unit\n\n" ++
  s!"def {funcName} {paramStr} (grid block : Nat × Nat × Nat := ((1, 1, 1), (128, 1, 1))) " ++
  s!"(sharedMem : Nat := 0) (stream : CudaStream := defaultStream) : IO Unit :=\n" ++
  s!"  let g := (grid.1.toUInt64, grid.2.1.toUInt64, grid.2.2.toUInt64)\n" ++
  s!"  let b := (block.1.toUInt64, block.2.1.toUInt64, block.2.2.toUInt64)\n" ++
  s!"  {externName} {argStr} g b sharedMem.toUInt64 stream"

/-! ## C++ Launcher Generation -/

/-- Convert GpuFloat to C++ CUDA type for pointers -/
def gpuFloatToCudaPtr (dtype : GpuFloat) : String :=
  match dtype with
  | .Int64 => "int64_t*"
  | .Float32 => "float*"
  | .Float16 => "__half*"
  | .BFloat16 => "__nv_bfloat16*"
  | .FP8E4M3 => "__nv_fp8_e4m3*"
  | .FP8E5M2 => "__nv_fp8_e5m2*"
  | .FP8E8M0 => "__nv_fp8_e8m0*"
  | .FP4E2M1X2 => "__nv_fp4x2_e2m1*"

/-- Generate C++ extern parameter declaration -/
def paramToCppExternParam (p : KParam) : String :=
  if p.isPointer then
    s!"b_lean_obj_arg {p.name}"
  else
    s!"uint64_t {p.name}"

/-- Generate C++ kernel argument (pointer extraction or scalar pass-through) -/
def paramToCppKernelArg (p : KParam) : String :=
  if p.isPointer then
    s!"{p.name}_ptr"
  else
    p.name

/-- Generate C++ pointer extraction code -/
def generatePointerExtraction (p : KParam) : String :=
  if p.isPointer then
    s!"  auto {p.name}_ptr = borrowTensor({p.name}).data_ptr<{p.dtype.toCpp}>();\n"
  else
    ""

/-- Generate C++ launcher function for a kernel registration. -/
def generateCppLauncher (kernel : RegisteredKernel) : String :=
  kernel.cppCode

/-- Generate complete C++ header for kernel launchers -/
def generateCppHeader : String :=
  "#include <lean/lean.h>\n" ++
  "#include <array>\n" ++
  "#include <stdexcept>\n" ++
  "#include <type_traits>\n" ++
  "#include <kittens.cuh>\n" ++
  "#include <torch/torch.h>\n" ++
  "#include <cuda_runtime.h>\n" ++
  "#include <cuda_fp16.h>\n" ++
  "#include <cuda_bf16.h>\n" ++
  "#include \"tyr_nat.h\"\n" ++
  "#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)\n" ++
  "#include <cuda_fp8.h>\n" ++
  "#endif\n" ++
  "#ifdef KITTENS_BLACKWELL\n" ++
  "#include <cuda_fp4.h>\n" ++
  "#endif\n\n" ++
  "using namespace kittens;\n\n" ++
  "template <typename T> struct tyr_raw_gl { T* raw_ptr; };\n\n" ++
  "// Forward declarations\n" ++
  "extern torch::Tensor borrowTensor(b_lean_obj_arg o);\n" ++
  "\n" ++
  "#ifndef CUDA_CHECK\n" ++
  "#define CUDA_CHECK(call) do { \\\n" ++
  "  cudaError_t err = call; \\\n" ++
  "  if (err != cudaSuccess) { \\\n" ++
  "    throw std::runtime_error(cudaGetErrorString(err)); \\\n" ++
  "  } \\\n" ++
  "} while(0)\n" ++
  "#endif\n\n"

/-! ## Combined Generation -/

/-- Generate all FFI bindings for a kernel -/
def generateAllBindings (kernel : RegisteredKernel) : String × String :=
  let leanCode := generateLeanExternDecl kernel
  let cppCode := generateCppLauncher kernel
  (leanCode, cppCode)

/-- Generate FFI for all registered kernels. -/
def generateAllKernelFFI : IO (String × String) := do
  let kernels ← getRegisteredKernelsIO
  let leanDecls := kernels.toList.map generateLeanExternDecl
  let cppLaunchers := kernels.toList.map generateCppLauncher

  let leanCode :=
    "-- Auto-generated kernel FFI declarations\n" ++
    "-- Do not edit manually\n\n" ++
    "namespace Tyr.GPU.Kernels\n\n" ++
    String.intercalate "\n\n" leanDecls ++
    "\n\nend Tyr.GPU.Kernels"

  let cppCode :=
    generateCppHeader ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" cppLaunchers ++
    "\n} // extern \"C\"\n"

  pure (leanCode, cppCode)

/-! ## Commands for FFI Generation -/

/-- Command to generate Lean FFI declarations for all kernels -/
syntax "#generate_lean_ffi" : command

macro_rules
  | `(#generate_lean_ffi) => `(
    #eval do
      let (leanCode, _) ← generateAllKernelFFI
      IO.println leanCode
  )

/-- Command to generate C++ launchers for all kernels -/
syntax "#generate_cpp_ffi" : command

macro_rules
  | `(#generate_cpp_ffi) => `(
    #eval do
      let (_, cppCode) ← generateAllKernelFFI
      IO.println cppCode
  )

/-- Command to write FFI files -/
syntax "#write_kernel_ffi" str str : command

macro_rules
  | `(#write_kernel_ffi $leanPath:str $cppPath:str) => `(
    #eval do
      let (leanCode, cppCode) ← generateAllKernelFFI
      IO.FS.writeFile $leanPath leanCode
      IO.FS.writeFile $cppPath cppCode
      IO.println s!"Written FFI to {$leanPath} and {$cppPath}"
  )

/-! ## Per-Kernel FFI Generation -/

/-- Generate FFI for a specific kernel by name. -/
def generateKernelFFI (name : Name) : IO (Option (String × String)) := do
  let kernels ← getRegisteredKernelsIO
  pure <| kernels.find? (·.name == name) |>.map fun k =>
    (generateLeanExternDecl k, generateCppLauncher k)

/-! ## Non-Macro FFI Functions

These functions can be called from `#eval` or Lake scripts without needing macros.
-/

/-- Write all registered kernel C++ launchers to a file. -/
def writeAllKernelCpp (path : System.FilePath) : IO Unit := do
  let kernels ← getRegisteredKernelsIO
  let header := generateCppHeader
  let launchers := kernels.toList.map generateCppLauncher
  let cppCode := header ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" launchers ++
    "\n} // extern \"C\"\n"
  IO.FS.writeFile path cppCode
  IO.println s!"Written {kernels.size} kernel launchers to {path}"

/-- Get all generated C++ code as a string. -/
def getAllKernelCpp : IO String := do
  let kernels ← getRegisteredKernelsIO
  let header := generateCppHeader
  let launchers := kernels.toList.map generateCppLauncher
  pure <| header ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" launchers ++
    "\n} // extern \"C\"\n"

/-- Get all generated Lean FFI declarations as a string. -/
def getAllKernelLean : IO String := do
  let kernels ← getRegisteredKernelsIO
  let decls := kernels.toList.map generateLeanExternDecl
  pure <| "-- Auto-generated kernel FFI declarations\n" ++
  "-- Do not edit manually\n\n" ++
  "namespace Tyr.GPU.Kernels\n\n" ++
  String.intercalate "\n\n" decls ++
  "\n\nend Tyr.GPU.Kernels"

/-- Group registered kernels by the Lean module (compilation unit) that declared them. -/
def groupRegisteredKernelsByModule (kernels : Array RegisteredKernel)
    : Std.HashMap Name (Array RegisteredKernel) :=
  kernels.foldl (init := {}) fun acc k =>
    let prev := acc.getD k.moduleName #[]
    acc.insert k.moduleName (prev.push k)

/-- Convert module name to a stable filesystem-friendly stem. -/
def moduleNameToFileStem (moduleName : Name) : String :=
  moduleName.toString.replace "." "_"

/-- Generate one CUDA translation unit for all kernels in a module. -/
def generateModuleKernelCu (moduleName : Name) (kernels : Array RegisteredKernel) : String :=
  let needStoreAdd := kernels.any (·.needsStoreAdd)
  let needLegacyTma := kernels.any (·.needsLegacyTma)
  let needSlice := kernels.any (·.needsSlice)
  let needOuter := kernels.any (·.needsOuter)
  let needEqMask := kernels.any (·.needsEqMask)
  let helpers := generateHelpersFromFlags needStoreAdd needLegacyTma needSlice needOuter needEqMask
  let kernelDefs := String.intercalate "\n" (kernels.toList.map (·.kernelDef))
  let launchers := String.intercalate "\n" (kernels.toList.map (·.cppCode))
  generateCppHeader ++
    s!"// Auto-generated kernels for module: {moduleName}\n" ++
    s!"// Kernel count: {kernels.size}\n\n" ++
    helpers ++
    kernelDefs ++
    "\nextern \"C\" {\n\n" ++
    launchers ++
    "\n} // extern \"C\"\n"

/-- Write one generated `.cu` file per Lean compilation unit from a given kernel set. -/
def writeKernelCudaUnitsByModuleFrom (kernels : Array RegisteredKernel)
    (outDir : System.FilePath) (clean : Bool := true) : IO (Array System.FilePath) := do
  let grouped := groupRegisteredKernelsByModule kernels
  let targetFiles : Std.HashSet String :=
    grouped.toList.foldl (init := {}) fun acc (moduleName, _) =>
      acc.insert s!"{moduleNameToFileStem moduleName}.cu"

  let writeIfChanged (path : System.FilePath) (content : String) : IO Unit := do
    if ← path.pathExists then
      let prev ← IO.FS.readFile path
      if prev == content then
        return ()
    IO.FS.writeFile path content

  IO.FS.createDirAll outDir
  if clean then
    let entries ← System.FilePath.readDir outDir
    for entry in entries do
      let path := entry.path
      let fileName := path.fileName.getD ""
      if path.extension == some "cu" && !targetFiles.contains fileName then
        let md ← path.metadata
        if md.type == .file then
          IO.FS.removeFile path

  let mut written : Array System.FilePath := #[]
  for (moduleName, kernels) in grouped.toList do
    let fileName := s!"{moduleNameToFileStem moduleName}.cu"
    let path := outDir / fileName
    writeIfChanged path (generateModuleKernelCu moduleName kernels)
    written := written.push path
  return written

/-- Write one generated `.cu` file per Lean compilation unit from runtime-registered kernels. -/
def writeKernelCudaUnitsByModule (outDir : System.FilePath)
    (clean : Bool := true) : IO (Array System.FilePath) := do
  let kernels ← getRegisteredKernelsIO
  writeKernelCudaUnitsByModuleFrom kernels outDir clean

end Tyr.GPU.Codegen
