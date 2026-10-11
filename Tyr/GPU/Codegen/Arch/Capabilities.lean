import Tyr.GPU.Codegen.Arch.Level

/-!
# Tyr.GPU.Codegen.Arch.Capabilities

`Tyr.GPU.Codegen.Arch.Capabilities` attaches concrete tuning/configuration data
to each `ArchLevel`.

Compared with `Tyr.GPU.Capabilities` (feature gating by `GpuArch`), this layer
focuses on kernel-generation policy values such as:

- default tile sizes,
- shared-memory budgets,
- default block/pipeline settings,
- feature flags used by arch-specific dispatch.

These records are consumed by polymorphic builders and smart operation selection.
-/

namespace Tyr.GPU.Codegen.Arch

open Tyr.GPU

/-- Architecture configuration constants as a typeclass.
    This allows operations to access arch-specific parameters at compile time. -/
class ArchConfig (arch : ArchLevel) where
  /-- Optimal MMA tile size (M, N, K) -/
  mmaTileSize : Nat × Nat × Nat
  /-- Maximum shared memory per SM (bytes) -/
  maxSharedMem : Nat
  /-- Whether TMA (Tensor Memory Accelerator) is available -/
  hasTMA : Bool
  /-- Whether WGMMA (Warpgroup MMA) is available -/
  hasWGMMA : Bool
  /-- Whether FP8 datatypes are available -/
  hasFP8 : Bool
  /-- Whether `fp8e8m0` scale tiles are available -/
  hasFP8E8M0 : Bool
  /-- Whether packed `fp4e2m1_2` storage is available -/
  hasFP4 : Bool
  /-- Whether distributed shared memory is available -/
  hasDSM : Bool
  /-- Whether cluster barriers are available -/
  hasClusterBarrier : Bool
  /-- Whether tensor memory is available -/
  hasTMEM : Bool
  /-- Number of threads in a warp -/
  warpSize : Nat := 32
  /-- Number of warps in a warpgroup -/
  warpsPerWarpgroup : Nat := 4
  /-- Default block size (threads per block) -/
  defaultBlockSize : Nat
  /-- Default number of pipeline stages -/
  defaultPipelineStages : Nat
  /-- Tensor core throughput (TFLOPS for bf16) -/
  tensorCoreTFLOPS : Nat
  /-- Typical number of SMs (varies by SKU) -/
  typicalSMs : Nat

/-- Ampere (SM80) configuration: baseline architecture -/
instance : ArchConfig .Ampere where
  mmaTileSize := (16, 16, 16)
  maxSharedMem := 164 * 1024  -- 164 KB
  hasTMA := false
  hasWGMMA := false
  hasFP8 := false
  hasFP8E8M0 := false
  hasFP4 := false
  hasDSM := false
  hasClusterBarrier := false
  hasTMEM := false
  defaultBlockSize := 256
  defaultPipelineStages := 2
  tensorCoreTFLOPS := 312  -- A100 bf16
  typicalSMs := 108        -- A100

/-- Hopper (SM90) configuration: +TMA, +WGMMA, +FP8 -/
instance : ArchConfig .Hopper where
  mmaTileSize := (64, 64, 16)
  maxSharedMem := 228 * 1024  -- 228 KB
  hasTMA := true
  hasWGMMA := true
  hasFP8 := true
  hasFP8E8M0 := false
  hasFP4 := false
  hasDSM := true
  hasClusterBarrier := true
  hasTMEM := false
  defaultBlockSize := 128
  defaultPipelineStages := 4
  tensorCoreTFLOPS := 989  -- H100 bf16 (with sparsity: 1979)
  typicalSMs := 132        -- H100 SXM

/-- Blackwell (SM100) configuration: +tcgen05, +2-CTA MMA, +FP4 -/
instance : ArchConfig .Blackwell where
  mmaTileSize := (64, 64, 32)
  maxSharedMem := 256 * 1024  -- 256 KB (estimated)
  hasTMA := true
  hasWGMMA := false
  hasFP8 := true
  hasFP8E8M0 := true
  hasFP4 := true
  hasDSM := true
  hasClusterBarrier := true
  hasTMEM := true
  defaultBlockSize := 128
  defaultPipelineStages := 4
  tensorCoreTFLOPS := 2250  -- B200 bf16 (estimated)
  typicalSMs := 160         -- B200 (estimated)

/-- Get architecture config for a runtime ArchLevel -/
@[reducible] def ArchConfig.get (arch : ArchLevel) : ArchConfig arch :=
  match arch with
  | .Ampere => inferInstance
  | .Hopper => inferInstance
  | .Blackwell => inferInstance

/-- Runtime-accessible architecture capabilities record.
    Mirrors the typeclass but usable at runtime. -/
structure ArchCapabilitiesRecord where
  mmaTileSize : Nat × Nat × Nat
  maxSharedMem : Nat
  hasTMA : Bool
  hasWGMMA : Bool
  hasFP8 : Bool
  hasFP8E8M0 : Bool
  hasFP4 : Bool
  hasDSM : Bool
  hasClusterBarrier : Bool
  hasTMEM : Bool
  warpSize : Nat
  warpsPerWarpgroup : Nat
  defaultBlockSize : Nat
  defaultPipelineStages : Nat
  tensorCoreTFLOPS : Nat
  typicalSMs : Nat
  deriving Repr, Inhabited

/-- Convert typeclass to runtime record -/
def ArchConfig.toRecord [cfg : ArchConfig arch] : ArchCapabilitiesRecord where
  mmaTileSize := cfg.mmaTileSize
  maxSharedMem := cfg.maxSharedMem
  hasTMA := cfg.hasTMA
  hasWGMMA := cfg.hasWGMMA
  hasFP8 := cfg.hasFP8
  hasFP8E8M0 := cfg.hasFP8E8M0
  hasFP4 := cfg.hasFP4
  hasDSM := cfg.hasDSM
  hasClusterBarrier := cfg.hasClusterBarrier
  hasTMEM := cfg.hasTMEM
  warpSize := cfg.warpSize
  warpsPerWarpgroup := cfg.warpsPerWarpgroup
  defaultBlockSize := cfg.defaultBlockSize
  defaultPipelineStages := cfg.defaultPipelineStages
  tensorCoreTFLOPS := cfg.tensorCoreTFLOPS
  typicalSMs := cfg.typicalSMs

/-- Get capabilities record for any architecture level -/
def ArchLevel.capabilities (arch : ArchLevel) : ArchCapabilitiesRecord :=
  match arch with
  | .Ampere => @ArchConfig.toRecord .Ampere inferInstance
  | .Hopper => @ArchConfig.toRecord .Hopper inferInstance
  | .Blackwell => @ArchConfig.toRecord .Blackwell inferInstance

/-- Type-level boolean for TMA support -/
class HasTMACapability (arch : ArchLevel) where
  proof : arch.capabilities.hasTMA = true

instance : HasTMACapability .Hopper := ⟨rfl⟩
instance : HasTMACapability .Blackwell := ⟨rfl⟩

/-- Type-level boolean for WGMMA support -/
class HasWGMMACapability (arch : ArchLevel) where
  proof : arch.capabilities.hasWGMMA = true

instance : HasWGMMACapability .Hopper := ⟨rfl⟩
/-- Type-level boolean for FP8 support -/
class HasFP8Capability (arch : ArchLevel) where
  proof : arch.capabilities.hasFP8 = true

instance : HasFP8Capability .Hopper := ⟨rfl⟩
instance : HasFP8Capability .Blackwell := ⟨rfl⟩

/-- Type-level boolean for MX-scale fp8e8m0 support. -/
class HasFP8E8M0Capability (arch : ArchLevel) where
  proof : arch.capabilities.hasFP8E8M0 = true

instance : HasFP8E8M0Capability .Blackwell := ⟨rfl⟩

/-- Type-level boolean for packed FP4 support. -/
class HasFP4Capability (arch : ArchLevel) where
  proof : arch.capabilities.hasFP4 = true

instance : HasFP4Capability .Blackwell := ⟨rfl⟩

/-- Type-level boolean for DSM support -/
class HasDSMCapability (arch : ArchLevel) where
  proof : arch.capabilities.hasDSM = true

instance : HasDSMCapability .Hopper := ⟨rfl⟩
instance : HasDSMCapability .Blackwell := ⟨rfl⟩

/-- Check if a dtype is supported by an architecture -/
def ArchLevel.supportsDtype (arch : ArchLevel) (dtype : GpuFloat) : Bool :=
  match dtype with
  | .Int64 | .Float32 | .Float16 | .BFloat16 => true
  | .FP8E4M3 | .FP8E5M2 => arch.capabilities.hasFP8
  | .FP8E8M0 => arch.capabilities.hasFP8E8M0
  | .FP4E2M1X2 => arch.capabilities.hasFP4

end Tyr.GPU.Codegen.Arch
