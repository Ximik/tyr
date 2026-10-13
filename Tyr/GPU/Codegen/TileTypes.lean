import Tyr.GPU.Types
import Tyr.GPU.Tile
import Tyr.GPU.Codegen.Var
import Tyr.Basic.Types

/-!
# Tyr.GPU.Codegen.TileTypes

`Tyr.GPU.Codegen.TileTypes` defines typed handles used during kernel construction.
Each handle wraps a `VarId` but carries rich phantom information:

- dtype (`GpuFloat`),
- dimensions (`rows`, `cols`, `len`),
- layout/location (`Row`/`Col`, register/shared/global).

These wrappers are what make many DSL operations type-safe: mismatched matrix
shapes or incompatible layouts are rejected by Lean's typechecker before IR
emission.
-/

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Register tile with dimensions embedded in type -/
structure RT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId
  deriving Repr

/-- Shared memory tile -/
structure ST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId
  deriving Repr

/-- Shared memory tile array, matching ThunderKittens allocator arrays such as
    `st_bf<128,64> (&k_smem)[4]`. -/
structure STArray (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    (len : Nat) where
  id : VarId
  deriving Repr

/-- Register vector -/
structure RV (dtype : GpuFloat) (len : Nat) where
  id : VarId
  deriving Repr

/-- Shared vector -/
structure SV (dtype : GpuFloat) (len : Nat) where
  id : VarId
  deriving Repr

/-- Shared row vector associated with a shared tile descriptor.

ThunderKittens represents `row_vec<st<T, rows, cols>>` as an `sv<T, cols>` in
shared memory, but the descriptor shape matters for TMA vector loads/stores. -/
structure STRowVec (dtype : GpuFloat) (rows cols : Nat) where
  id : VarId
  deriving Repr

/-- Shared column vector associated with a shared tile descriptor.

ThunderKittens represents `col_vec<st<T, rows, cols>>` as an `sv<T, rows>` in
shared memory, but the descriptor shape matters for TMA vector loads/stores. -/
structure STColVec (dtype : GpuFloat) (rows cols : Nat) where
  id : VarId
  deriving Repr

/-- Shared column-vector tile array, used for per-consumer LSE staging. -/
structure STColVecArray (dtype : GpuFloat) (rows cols len : Nat) where
  id : VarId
  deriving Repr

/-- Shared semaphore array for TK ring buffers. -/
structure SemaphoreArray (len : Nat) where
  id : VarId
  deriving Repr

/-- Tensor memory tile (Blackwell SM100 TMEM) -/
structure TT (dtype : GpuFloat) (rows cols : Nat) where
  id : VarId
  deriving Repr

/-- TMEM allocation pool handle -/
structure TMEMPool where
  id : VarId
  deriving Repr

/-! ## Kernel Parameter Types

These types represent kernel parameters as first-class Lean values.
They wrap VarId so they can be used in operations like tmaLoad.
-/

/-- Global memory pointer (kernel parameter)
    This is a first-class Lean value usable in TMA operations -/
structure GPtr (dtype : GpuFloat) where
  id : VarId
  name : String  -- Original parameter name for debugging/codegen
  deriving Repr

/-- Kernel scalar value parameter
    Usable in index calculations, loop bounds, etc. -/
structure KVal (T : Type) where
  id : VarId
  name : String
  deriving Repr

/-- Linear shared-memory pointer produced by dynamic shared-memory allocation.

    This models a TileLang-style `alloc_shared` for SIMT code whose extent is
    runtime-sized. The backend owns the CUDA lowering; kernels manipulate it
    through typed load/store primitives rather than raw code. -/
structure KShared (T : Type) where
  id : VarId
  name : String
  deriving Repr

/-- Get VarId from global pointer -/
def GPtr.varId {dtype : GpuFloat} (p : GPtr dtype) : VarId := p.id

/-- Get VarId from kernel value -/
def KVal.varId {T : Type} (v : KVal T) : VarId := v.id

/-- Get VarId from dynamic shared-memory pointer. -/
def KShared.varId {T : Type} (p : KShared T) : VarId := p.id

/-- Get VarId from register tile -/
def RT.varId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : RT dtype rows cols layout) : VarId := t.id

/-- Get VarId from shared tile -/
def ST.varId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : ST dtype rows cols layout) : VarId := t.id

/-- Get VarId from shared tile array. -/
def STArray.varId {dtype : GpuFloat} {rows cols len : Nat} {layout : TileLayout}
    (t : STArray dtype rows cols layout len) : VarId := t.id

/-- Get VarId from register vector -/
def RV.varId {dtype : GpuFloat} {len : Nat} (v : RV dtype len) : VarId := v.id

/-- Get VarId from shared vector -/
def SV.varId {dtype : GpuFloat} {len : Nat} (v : SV dtype len) : VarId := v.id

/-- Get VarId from shared row vector tile descriptor. -/
def STRowVec.varId {dtype : GpuFloat} {rows cols : Nat}
    (v : STRowVec dtype rows cols) : VarId := v.id

/-- Get VarId from shared column vector tile descriptor. -/
def STColVec.varId {dtype : GpuFloat} {rows cols : Nat}
    (v : STColVec dtype rows cols) : VarId := v.id

/-- Get VarId from shared column vector tile array descriptor. -/
def STColVecArray.varId {dtype : GpuFloat} {rows cols len : Nat}
    (v : STColVecArray dtype rows cols len) : VarId := v.id

/-- Get VarId from a semaphore array. -/
def SemaphoreArray.varId {len : Nat} (v : SemaphoreArray len) : VarId := v.id

/-- Get VarId from tensor memory tile -/
def TT.varId {dtype : GpuFloat} {rows cols : Nat} (t : TT dtype rows cols) : VarId := t.id

/-- Get VarId from TMEM pool -/
def TMEMPool.varId (p : TMEMPool) : VarId := p.id

/-! ## Complex Number Types -/

/-- Complex register tile (pair of real/imag tiles) -/
structure CRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  real : RT dtype rows cols layout
  imag : RT dtype rows cols layout
  deriving Repr

/-- Complex shared memory tile -/
structure CST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  real : ST dtype rows cols layout
  imag : ST dtype rows cols layout
  deriving Repr

/-- Complex register vector -/
structure CRV (dtype : GpuFloat) (len : Nat) where
  real : RV dtype len
  imag : RV dtype len
  deriving Repr

/-- Complex shared vector -/
structure CSV (dtype : GpuFloat) (len : Nat) where
  real : SV dtype len
  imag : SV dtype len
  deriving Repr

/-- Get real part VarIds -/
def CRT.realId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : CRT dtype rows cols layout) : VarId := t.real.id

def CRT.imagId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : CRT dtype rows cols layout) : VarId := t.imag.id

/-! ## FFI Types for Kernel Launching

For now, we reuse `torch.TSpec.type` (all `torch.T s` share the same underlying type)
so kernels can be launched directly with existing torch tensors.

Streams are represented as raw `UInt64` (interpreted as `cudaStream_t`).
Use `0` for the default stream.
-/

/-- Untyped tensor handle for FFI (backed by libtorch). -/
abbrev Tensor : Type := torch.T #[]

/-- CUDA stream handle for kernel launches (`cudaStream_t` cast to `UInt64`). -/
abbrev CudaStream : Type := UInt64

end Tyr.GPU.Codegen

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Bridge to Core Tile Abstractions

These instances let Codegen tile types participate in the core Tile/SharedTile
typeclasses so helpers like `Tile.bytes` and `Tile.validForMMA` work.
-/

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    Tile (RT dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  location := .Register
  layout := layout

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    RegisterTile (RT dtype rows cols layout) where
  location_is_register := rfl
  height := rows / 16
  width := cols / 16

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    Tile (ST dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  location := .Shared
  layout := layout

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    SharedTile (ST dtype rows cols layout) where
  location_is_shared := rfl

instance {dtype : GpuFloat} {rows cols len : Nat} {layout : TileLayout} :
    Tile (STArray dtype rows cols layout len) where
  dtype := dtype
  rows := rows
  cols := cols
  location := .Shared
  layout := layout

instance {dtype : GpuFloat} {rows cols len : Nat} {layout : TileLayout} :
    SharedTile (STArray dtype rows cols layout len) where
  location_is_shared := rfl

end Tyr.GPU.Codegen
