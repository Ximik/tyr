/-
  Tyr/GPU/Kernels/Based.lean

  ThunderKittens-shaped "Based" linear attention kernels.

  The canonical forward surface tracks the three recurrent state components used
  by `external/ThunderKittens/kernels/based/linear_attn.cu`:

  - `a0`: cumulative value bias term
  - `a1`: first-order recurrent `VᵀK` state
  - `a2`: second-order recurrent quadratic state, split into four 64x64 tiles

  The DSL still cannot express the exact warp-shuffle implementation of
  `mul_slice_row` / `mul_slice_col`, but the helpers below now materialize the
  same quadratic feature groups via explicit slice + broadcast operations
  instead of the previous placeholder `kSq` path.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Based

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Extract a single source column as a row-reduction vector. -/
private def extractColumnVec {dtype : GpuFloat} {rows cols : Nat}
    (src : RT dtype rows cols .Row) (startCol : Nat) : KernelM (RV dtype rows) := do
  let colTile : RT dtype rows 1 ← allocRT dtype rows 1
  let colVec : RV dtype rows ← allocRV dtype rows
  sliceCols colTile src startCol 1
  rowSum colVec colTile
  pure colVec

/-- ThunderKittens-style quadratic feature group.

For a `[rows x 16]` source tile this builds one `[rows x 64]` group by
concatenating four column-scaled copies of the source, matching the logical
shape of the CUDA `mul_slice_row` / `mul_slice_col` helpers even though it uses
portable slice/broadcast ops instead of warp shuffles. -/
private def quadraticFeatureGroup {dtype : GpuFloat} {rows : Nat}
    (src : RT dtype rows 16 .Row) (startCol : Nat) : KernelM (RT dtype rows 64 .Row) := do
  let v0 ← extractColumnVec src startCol
  let v1 ← extractColumnVec src (startCol + 1)
  let v2 ← extractColumnVec src (startCol + 2)
  let v3 ← extractColumnVec src (startCol + 3)

  let t0 : RT dtype rows 16 ← allocRT dtype rows 16
  let t1 : RT dtype rows 16 ← allocRT dtype rows 16
  let t2 : RT dtype rows 16 ← allocRT dtype rows 16
  let t3 : RT dtype rows 16 ← allocRT dtype rows 16
  mulCol t0 src v0
  mulCol t1 src v1
  mulCol t2 src v2
  mulCol t3 src v3

  let left : RT dtype rows 32 ← allocRT dtype rows 32
  let right : RT dtype rows 32 ← allocRT dtype rows 32
  let dst : RT dtype rows 64 ← allocRT dtype rows 64
  concatCols left t0 t1
  concatCols right t2 t3
  concatCols dst left right
  pure dst

/-- Accumulate one recurrent quadratic-state contribution into the output tile. -/
private def addA2Contribution
    (o : RT GpuFloat.Float32 64 64 .Row)
    (qQuad : RT GpuFloat.Float32 64 64 .Row)
    (a2 : RT GpuFloat.Float32 64 64 .Row) : KernelM Unit := do
  let qQuadBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let a2BfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let a2Bf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let contrib : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  convert qQuadBf qQuad
  convert a2BfRow a2
  swapLayout a2Bf a2BfRow
  mma contrib qQuadBf a2Bf contrib
  add o o contrib

/-- Update one recurrent quadratic-state tile from the current K/V chunk. -/
private def updateA2State
    (a2 : RT GpuFloat.Float32 64 64 .Row)
    (kQuad : RT GpuFloat.Float32 64 64 .Row)
    (v : RT GpuFloat.BFloat16 64 64 .Col) : KernelM Unit := do
  let kQuadBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kQuadCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let updated : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  convert kQuadBf kQuad
  swapLayout kQuadCol kQuadBf
  mmaAtB updated kQuadCol v a2
  copy a2 updated

/-! ## Based Linear Attention

This is the canonical source-backed surface for the vendored ThunderKittens
`based/linear_attn.cu` kernel. It keeps the same three-part recurrent state and
adds the missing local causal polynomial attention contribution.
-/

/-- Based Linear Attention forward pass with explicit `a0/a1/a2` state. -/
@[gpu_kernel .SM90]
def basedLinearAttnFwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (a0_ptr : GPtr GpuFloat.Float32) (a1_ptr : GPtr GpuFloat.Float32)
    (a2_0_ptr : GPtr GpuFloat.Float32) (a2_1_ptr : GPtr GpuFloat.Float32)
    (a2_2_ptr : GPtr GpuFloat.Float32) (a2_3_ptr : GPtr GpuFloat.Float32)
    (_batch_size : KVal UInt64) (_num_heads : KVal UInt64) (_seq_len : KVal UInt64)
    (_d_qk : KVal UInt64) (_d_v : KVal UInt64) : KernelM Unit := do
  comment "=== Based Linear Attention Forward ==="
  comment "Local causal polynomial attention + recurrent a0/a1/a2 state"

  let coord ← blockCoord2D

  let q : RT GpuFloat.BFloat16 64 16 ← allocRT .BFloat16 64 16
  let k : RT GpuFloat.BFloat16 64 16 ← allocRT .BFloat16 64 16
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let qF : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let qScaled : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let qScaledBf : RT GpuFloat.BFloat16 64 16 ← allocRT .BFloat16 64 16
  let kF : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16

  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let outBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let a0 : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let a1 : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let a1T : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
  let a1TCol : RT GpuFloat.Float32 16 64 .Col ← allocRT .Float32 16 64 .Col
  let a1TBf : RT GpuFloat.BFloat16 16 64 .Col ← allocRT .BFloat16 16 64 .Col

  let a2_0 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let a2_1 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let a2_2 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let a2_3 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  let localScores : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let localSq : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let localPoly : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let localPolyBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let localOut : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let temp : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kv : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let vSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let vFCol : RT GpuFloat.Float32 64 64 .Col ← allocRT .Float32 64 64 .Col
  let vFRow : RT GpuFloat.Float32 64 64 .Row ← allocRT .Float32 64 64 .Row
  let kCol : RT GpuFloat.BFloat16 64 16 .Col ← allocRT .BFloat16 64 16 .Col

  let qShared : ST GpuFloat.BFloat16 64 16 ← allocST .BFloat16 64 16
  let kShared : ST GpuFloat.BFloat16 64 16 ← allocST .BFloat16 64 16
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let oShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let a0Shared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let a1Shared : ST GpuFloat.Float32 64 16 ← allocST .Float32 64 16
  let a2Shared_0 : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let a2Shared_1 : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let a2Shared_2 : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let a2Shared_3 : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  comment "Load recurrent a0/a1/a2 state"
  loadVecGlobalCoord a0Shared a0_ptr coord.c
  loadGlobal a1Shared a1_ptr coord
  loadGlobal a2Shared_0 a2_0_ptr coord
  loadGlobal a2Shared_1 a2_1_ptr coord
  loadGlobal a2Shared_2 a2_2_ptr coord
  loadGlobal a2Shared_3 a2_3_ptr coord
  sync
  loadVec a0 a0Shared
  load a1 a1Shared
  load a2_0 a2Shared_0
  load a2_1 a2Shared_1
  load a2_2 a2Shared_2
  load a2_3 a2Shared_3

  comment "Process sequence chunks"
  for chunkIdx in krange 0 16 do
    loadGlobal qShared Q_ptr (coord.withRow chunkIdx.id)
    loadGlobal kShared K_ptr (coord.withRow chunkIdx.id)
    loadGlobal vShared V_ptr (coord.withRow chunkIdx.id)
    sync
    load q qShared
    load k kShared
    load v vShared
    convert qF q
    convert kF k
    zero o

    comment "Local causal polynomial attention: 1 + qk + qk^2 / 2"
    zero localScores
    mmaT localScores q k localScores
    scalarMul localScores localScores 0.25
    mul localSq localScores localScores
    scalarMul localSq localSq 0.5
    add localPoly localSq localScores
    scalarAdd localPoly localPoly 1.0
    makeCausal localPoly localPoly (some 0.0)
    convert localPolyBf localPoly
    zero localOut
    mma localOut localPolyBf v localOut
    add o o localOut

    comment "Recurrent zeroth-order contribution"
    addRow o o a0

    comment "Recurrent first-order contribution"
    scalarMul qScaled qF 0.25
    convert qScaledBf qScaled
    transpose a1T a1
    swapLayout a1TCol a1T
    convert a1TBf a1TCol
    zero temp
    mma temp qScaledBf a1TBf temp
    add o o temp

    comment "Recurrent second-order contribution using four quadratic groups"
    let qQuad0 ← quadraticFeatureGroup qScaled 0
    let qQuad1 ← quadraticFeatureGroup qScaled 4
    let qQuad2 ← quadraticFeatureGroup qScaled 8
    let qQuad3 ← quadraticFeatureGroup qScaled 12
    addA2Contribution o qQuad0 a2_0
    addA2Contribution o qQuad1 a2_1
    addA2Contribution o qQuad2 a2_2
    addA2Contribution o qQuad3 a2_3

    comment "Update a0 = a0 + sum(V)"
    convert vFCol v
    swapLayout vFRow vFCol
    colSum vSum vFRow
    addVec a0 a0 vSum

    comment "Update a1 = a1 + V^T K"
    swapLayout kCol k
    mmaAtB kv v kCol a1
    copy a1 kv

    comment "Update a2 groups from K quadratic features"
    let kQuad0 ← quadraticFeatureGroup kF 0
    let kQuad1 ← quadraticFeatureGroup kF 4
    let kQuad2 ← quadraticFeatureGroup kF 8
    let kQuad3 ← quadraticFeatureGroup kF 12
    updateA2State a2_0 kQuad0 v
    updateA2State a2_1 kQuad1 v
    updateA2State a2_2 kQuad2 v
    updateA2State a2_3 kQuad3 v

    convert outBf o
    store oShared outBf
    storeGlobal O_ptr oShared (coord.withRow chunkIdx.id)
    sync

  comment "Store scaled recurrent state"
  scalarMul a1 a1 0.5
  scalarMul a2_0 a2_0 (0.70710678118 * 0.25)
  scalarMul a2_1 a2_1 (0.70710678118 * 0.25)
  scalarMul a2_2 a2_2 (0.70710678118 * 0.25)
  scalarMul a2_3 a2_3 (0.70710678118 * 0.25)
  storeVec a0Shared a0
  store a1Shared a1
  store a2Shared_0 a2_0
  store a2Shared_1 a2_1
  store a2Shared_2 a2_2
  store a2Shared_3 a2_3
  sync
  storeVecGlobalCoord a0_ptr a0Shared coord.c
  storeGlobal a1_ptr a1Shared coord
  storeGlobal a2_0_ptr a2Shared_0 coord
  storeGlobal a2_1_ptr a2Shared_1 coord
  storeGlobal a2_2_ptr a2Shared_2 coord
  storeGlobal a2_3_ptr a2Shared_3 coord

/-- Compatibility inference kernel.

This keeps the older first-order recurrent surface for single-token experiments.
The canonical ThunderKittens-aligned stateful path is `basedLinearAttnFwd`,
which owns `a0`, `a1`, and the four `a2` tiles explicitly. -/
@[gpu_kernel .SM90]
def basedLinearAttnInference (q_ptr : GPtr GpuFloat.BFloat16) (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16) (o_ptr : GPtr GpuFloat.Float32)
    (a0_ptr : GPtr GpuFloat.Float32) (a1_ptr : GPtr GpuFloat.Float32)
    (_batch_size : KVal UInt64) (_num_heads : KVal UInt64)
    (_d_qk : KVal UInt64) (_d_v : KVal UInt64) : KernelM Unit := do
  comment "=== Based Linear Attention Inference Compatibility Kernel ==="

  let coord ← blockCoord2D

  let q : RV GpuFloat.BFloat16 16 ← allocRV .BFloat16 16
  let k : RV GpuFloat.BFloat16 16 ← allocRV .BFloat16 16
  let v : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64
  let o : RV GpuFloat.Float32 64 ← zeroRV .Float32 64
  let a0 : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let a1 : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16

  let a0Shared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let a1Shared : ST GpuFloat.Float32 64 16 ← allocST .Float32 64 16
  let qShared : SV GpuFloat.BFloat16 16 ← allocSV .BFloat16 16
  let kShared : SV GpuFloat.BFloat16 16 ← allocSV .BFloat16 16
  let vShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let oShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  let qF : RV GpuFloat.Float32 16 ← allocRV .Float32 16
  let kF : RV GpuFloat.Float32 16 ← allocRV .Float32 16
  let vF : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let outerVK : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let a1Scaled : RT GpuFloat.Float32 64 16 ← allocRT .Float32 64 16
  let a1qSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  loadVecGlobalCoord qShared q_ptr coord.c
  loadVecGlobalCoord kShared k_ptr coord.c
  loadVecGlobalCoord vShared v_ptr coord.c
  loadVecGlobalCoord a0Shared a0_ptr coord.c
  loadGlobal a1Shared a1_ptr coord
  sync

  loadVec q qShared
  loadVec k kShared
  loadVec v vShared
  loadVec a0 a0Shared
  load a1 a1Shared

  convertVec qF q
  convertVec kF k
  convertVec vF v

  copyVec o a0
  copy a1Scaled a1
  mulRow a1Scaled a1 qF
  rowSum a1qSum a1Scaled
  addVec o o a1qSum

  addVec a0 a0 vF
  outer outerVK vF kF
  add a1 a1 outerVK

  storeVec a0Shared a0
  store a1Shared a1
  storeVec oShared o
  sync
  storeVecGlobalCoord a0_ptr a0Shared coord.c
  storeGlobal a1_ptr a1Shared coord
  storeVecGlobalCoord o_ptr oShared coord.c

/-- Canonical ThunderKittens-aligned name. -/
abbrev tkBasedLinearAttnFwd := basedLinearAttnFwd

end Tyr.GPU.Kernels.Based
