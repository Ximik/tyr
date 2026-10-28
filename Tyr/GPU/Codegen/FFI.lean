import Lean
import Tyr.GPU.Types
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.EmitNew

/-!
# Tyr.GPU.Codegen.FFI

`Tyr.GPU.Codegen.FFI` writes the generated CUDA translation units: one `.cu` file
per Lean module, holding that module's kernels and their C++ launchers. The Lean
`@[extern]` launch declarations and the launcher code come from
`Tyr.GPU.Codegen.Attribute`.
-/

namespace Tyr.GPU.Codegen

open Lean Meta Elab Term Command
open Tyr.GPU

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

end Tyr.GPU.Codegen
