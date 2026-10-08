import Std.Data.HashMap
import Std.Data.HashSet
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.AST

/-!
# Tyr.GPU.Codegen.EmitNew

`Tyr.GPU.Codegen.EmitNew` lowers `Kernel` IR into backend CUDA/C++ source.

Responsibilities include:

- deterministic naming and declaration ordering,
- operation-specific code templates (mma, reductions, async I/O, etc.),
- helper synthesis and layout bookkeeping needed by emitted kernels.

This is the final translation stage after typed kernel construction.
If `Ops`/`Notation` are "authoring-time DSL", this module is the backend compiler.
-/

namespace Tyr.GPU.Codegen

/-- Render a Float as an exact C/CUDA literal. Lean's `Float.toString`
    formats at 6 decimals, which silently truncated kernel constants
    (e.g. `1e-12` became `0.000000f`); hex-float literals round-trip the
    full IEEE-754 value and are valid C++17/CUDA. -/
def floatToCLiteral (value : Float) : String :=
  if value != value then
    "NAN"
  else if value == 0.0 then
    (if 1.0 / value < 0.0 then "-0.0" else "0.0")
  else if value.isInf then
    (if value > 0.0 then "INFINITY" else "-INFINITY")
  else
    let bits := value.toBits
    let sign := if bits >>> 63 == 1 then "-" else ""
    let expBits : Int := ((bits >>> 52) &&& 0x7FF).toNat
    let mant := (bits &&& 0xFFFFFFFFFFFFF).toNat
    let hexDigits := String.ofList ((Nat.toDigits 16 mant))
    let pad := String.ofList (List.replicate (13 - hexDigits.length) '0')
    if expBits == 0 then
      s!"{sign}0x0.{pad}{hexDigits}p-1022"
    else
      s!"{sign}0x1.{pad}{hexDigits}p{expBits - 1023}"


open Tyr.GPU

/-- Zero literal for generated mutable scalar declarations. -/
private def KScalarType.zeroCpp : KScalarType → String
  | .Float | .Float32 => "0.0f"
  | .Bool => "false"
  | _ => "0"

/-- Convert a flat global element into the requested scalar type. -/
private def renderFlatGlobalLoadExpr (srcDtype : GpuFloat) (dstTy : KScalarType)
    (src offset : VarId) : String :=
  let value := s!"{src.toIdent}.raw_ptr[{offset.toIdent}]"
  match srcDtype, dstTy with
  | .BFloat16, .Float32 => s!"__bfloat162float({value})"
  | .BFloat16, .Float => s!"static_cast<double>(__bfloat162float({value}))"
  | .Float32, .Float32 => value
  | .Float32, .Float => s!"static_cast<double>({value})"
  | _, _ => s!"static_cast<{dstTy.toCpp}>({value})"

/-- Convert a scalar value for a flat global store. -/
private def renderFlatGlobalStoreExpr (dstDtype : GpuFloat) (srcTy : KScalarType)
    (src : VarId) : String :=
  let value := src.toIdent
  match dstDtype, srcTy with
  | .BFloat16, .Float32 => s!"__float2bfloat16(static_cast<float>({value}))"
  | .BFloat16, .Float => s!"__float2bfloat16(static_cast<float>({value}))"
  | .Float32, .Float32 => value
  | .Float32, .Float => s!"static_cast<float>({value})"
  | _, _ => s!"static_cast<{dstDtype.toCpp}>({value})"

/-- Register vector layout for ThunderKittens row/col operations. -/
inductive RVLayout where
  | Naive
  | Align
  | Ortho
  deriving Repr, BEq, Hashable, Inhabited

private def RVLayout.toCpp : RVLayout → String
  | .Naive => "ducks::rv_layout::naive"
  | .Align => "ducks::rv_layout::align"
  | .Ortho => "ducks::rv_layout::ortho"

private def rowVecLayout : TileLayout → RVLayout
  | .Row => .Align
  | .Col => .Ortho

private def colVecLayout : TileLayout → RVLayout
  | .Row => .Ortho
  | .Col => .Align

private def indentRaw (indent code : String) : String :=
  let lines := code.splitOn "\n"
  lines.map (fun line => if line.isEmpty then "" else indent ++ line) |> String.intercalate "\n"

structure RVLayoutState where
  layouts : Std.HashMap VarId RVLayout := {}
  conflicts : Std.HashSet VarId := {}

private inductive TileKind where
  | RT
  | ST
  | STArray
  | STRowVec
  | STColVec
  | STColVecArray
  | TT  -- Tensor memory tile (Blackwell)
  deriving Repr, BEq, Hashable, Inhabited

private structure TileInfo where
  kind : TileKind
  rows : Nat
  cols : Nat
  dtype : GpuFloat
  layout : TileLayout := .Row
  len : Nat := 1
  deriving Repr, BEq, Inhabited

private def addRvLayout (st : RVLayoutState) (v : VarId) (layout : RVLayout) : RVLayoutState :=
  match st.layouts[v]? with
  | none => { st with layouts := st.layouts.insert v layout }
  | some existing =>
      if existing == layout then st
      else
        { layouts := st.layouts, conflicts := st.conflicts.insert v }

private def collectRtLayoutsStmt (acc : Std.HashMap VarId TileLayout) : KStmt → Std.HashMap VarId TileLayout
  | .declRT v _ _ _ layout => acc.insert v layout
  | .forLoop _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .forLoopVal _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .forLoopRev _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .forLoopValRev _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .forLoopStrideVal _ _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectRtLayoutsStmt acc
      elseBody.foldl collectRtLayoutsStmt acc'
  | .ifWarpGroup _ body => body.foldl collectRtLayoutsStmt acc
  | _ => acc

private def collectTileInfoStmt (acc : Std.HashMap VarId TileInfo) : KStmt → Std.HashMap VarId TileInfo
  | .declRT v dtype rows cols layout => acc.insert v { kind := .RT, rows := rows, cols := cols, dtype := dtype, layout := layout }
  | .declST v dtype rows cols layout => acc.insert v { kind := .ST, rows := rows, cols := cols, dtype := dtype, layout := layout }
  | .declSTArray v dtype rows cols layout len => acc.insert v { kind := .STArray, rows := rows, cols := cols, dtype := dtype, layout := layout, len := len }
  | .declSTAlias v dtype rows cols layout _ _ => acc.insert v { kind := .ST, rows := rows, cols := cols, dtype := dtype, layout := layout }
  | .declSTRowVec v dtype rows cols => acc.insert v { kind := .STRowVec, rows := rows, cols := cols, dtype := dtype }
  | .declSTColVec v dtype rows cols => acc.insert v { kind := .STColVec, rows := rows, cols := cols, dtype := dtype }
  | .declSTColVecArray v dtype rows cols len => acc.insert v { kind := .STColVecArray, rows := rows, cols := cols, dtype := dtype, len := len }
  | .declTT v dtype rows cols => acc.insert v { kind := .TT, rows := rows, cols := cols, dtype := dtype }
  | .forLoop _ _ _ body => body.foldl collectTileInfoStmt acc
  | .forLoopVal _ _ _ body => body.foldl collectTileInfoStmt acc
  | .forLoopRev _ _ _ body => body.foldl collectTileInfoStmt acc
  | .forLoopValRev _ _ _ body => body.foldl collectTileInfoStmt acc
  | .forLoopStrideVal _ _ _ _ body => body.foldl collectTileInfoStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectTileInfoStmt acc
      elseBody.foldl collectTileInfoStmt acc'
  | .ifWarpGroup _ body => body.foldl collectTileInfoStmt acc
  | _ => acc

private def collectRvDeclsStmt (acc : Std.HashSet VarId) : KStmt → Std.HashSet VarId
  | .declRV v _ _ => acc.insert v
  | .forLoop _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .forLoopVal _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .forLoopRev _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .forLoopValRev _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .forLoopStrideVal _ _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectRvDeclsStmt acc
      elseBody.foldl collectRvDeclsStmt acc'
  | .ifWarpGroup _ body => body.foldl collectRvDeclsStmt acc
  | _ => acc

private def rtLayoutOf (rtLayouts : Std.HashMap VarId TileLayout) (v : VarId) : TileLayout :=
  match rtLayouts[v]? with
  | some layout => layout
  | none => .Row

private def unifyRvVars (rvVars : Std.HashSet VarId) (st : RVLayoutState)
    (a b : VarId) : RVLayoutState :=
  if !(rvVars.contains a && rvVars.contains b) then
    st
  else
    match st.layouts[a]?, st.layouts[b]? with
    | none, none => st
    | some la, none => addRvLayout st b la
    | none, some lb => addRvLayout st a lb
    | some la, some lb =>
      if la == lb then st
      else
        { st with conflicts := (st.conflicts.insert a).insert b }

private def collectRvLayoutsStmt
    (rtLayouts : Std.HashMap VarId TileLayout)
    (rvVars : Std.HashSet VarId)
    (st : RVLayoutState) :
    KStmt → RVLayoutState
  | .reduce _ axis dst src =>
      let tileLayout := rtLayoutOf rtLayouts src
      let layout := match axis with
        | .Row => colVecLayout tileLayout
        | .Col => rowVecLayout tileLayout
        | .Full => .Naive  -- Full reduction doesn't need specific layout
      addRvLayout st dst layout
  | .reduceAccum _ axis dst src accum =>
      let tileLayout := rtLayoutOf rtLayouts src
      let layout := match axis with
        | .Row => colVecLayout tileLayout
        | .Col => rowVecLayout tileLayout
        | .Full => .Naive  -- Full reduction doesn't need specific layout
      let st' := addRvLayout st dst layout
      addRvLayout st' accum layout
  | .broadcast axis dst vec =>
      let tileLayout := rtLayoutOf rtLayouts dst
      let layout := match axis with
        | .Row => rowVecLayout tileLayout
        | .Col => colVecLayout tileLayout
      addRvLayout st vec layout
  | .binaryBroadcast _ axis _ tile vec =>
      let tileLayout := rtLayoutOf rtLayouts tile
      let layout := match axis with
        | .Row => rowVecLayout tileLayout
        | .Col => colVecLayout tileLayout
      addRvLayout st vec layout
  | .outer dst a b =>
      let tileLayout := rtLayoutOf rtLayouts dst
      let st' := addRvLayout st a (colVecLayout tileLayout)
      addRvLayout st' b (rowVecLayout tileLayout)
  | .convert dst src =>
      unifyRvVars rvVars st dst src
  | .unary _ dst src =>
      unifyRvVars rvVars st dst src
  | .convertVecLayout _ _ =>
      st
  | .binary _ dst a b =>
      let st' := unifyRvVars rvVars st dst a
      unifyRvVars rvVars st' dst b
  | .ternary _ dst a b c =>
      let st1 := unifyRvVars rvVars st dst a
      let st2 := unifyRvVars rvVars st1 dst b
      unifyRvVars rvVars st2 dst c
  | .scalarMul dst src _ =>
      unifyRvVars rvVars st dst src
  | .scalarAdd dst src _ =>
      unifyRvVars rvVars st dst src
  | .forLoop _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .forLoopVal _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .forLoopRev _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .forLoopValRev _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .forLoopStrideVal _ _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .ifStmt _ thenBody elseBody =>
      let st' := thenBody.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
      elseBody.foldl (collectRvLayoutsStmt rtLayouts rvVars) st'
  | .ifWarpGroup _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | _ => st

private def inferRvDecls (k : Kernel) : Std.HashSet VarId :=
  k.body.foldl collectRvDeclsStmt {}

private def inferRvLayouts (k : Kernel) : RVLayoutState :=
  let rtLayouts := k.body.foldl collectRtLayoutsStmt {}
  let rvVars := inferRvDecls k
  Id.run do
    let mut st : RVLayoutState := {}
    -- Fixed-point approximation for layout propagation along op chains.
    for _ in [:16] do
      st := k.body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
    return st

private def rvLayoutSuffix (rvLayouts : Std.HashMap VarId RVLayout) (v : VarId) : String :=
  match rvLayouts[v]? with
  | none => ""
  | some layout => s!", {layout.toCpp}"

private def layoutDiagnostics (conflicts : Std.HashSet VarId) : String :=
  conflicts.toList.foldl (fun acc v =>
    acc ++ s!"static_assert(false, \"RV layout conflict for {v.toIdent}\");\n") ""

private def inferTileInfo (k : Kernel) : Std.HashMap VarId TileInfo :=
  k.body.foldl collectTileInfoStmt {}

/-- Extra ThunderKittens TMA descriptor type that must be attached to a `gl`.
    We currently infer shared-tile descriptors, which are enough for the
    generated TMA load/store/store-add paths used by Hopper attention kernels. -/
inductive GlobalParamTmaType where
  | st (dtype : GpuFloat) (rows cols : Nat)
  | rowVecSt (dtype : GpuFloat) (rows cols : Nat)
  | colVecSt (dtype : GpuFloat) (rows cols : Nat)
  deriving Repr, BEq, Hashable, Inhabited

/-- Render a TMA descriptor type in ThunderKittens C++ syntax. -/
def GlobalParamTmaType.toCpp : GlobalParamTmaType → String
  | .st dtype rows cols => s!"st<{dtype.toCpp}, {rows}, {cols}>"
  | .rowVecSt dtype rows cols => s!"row_vec<st<{dtype.toCpp}, {rows}, {cols}>>"
  | .colVecSt dtype rows cols => s!"col_vec<st<{dtype.toCpp}, {rows}, {cols}>>"

private def descriptorToGlobalParamTmaType : GlobalTileDescriptor → GlobalParamTmaType
  | GlobalTileDescriptor.st dtype rows cols => GlobalParamTmaType.st dtype rows cols
  | GlobalTileDescriptor.rowVecSt dtype rows cols => GlobalParamTmaType.rowVecSt dtype rows cols
  | GlobalTileDescriptor.colVecSt dtype rows cols => GlobalParamTmaType.colVecSt dtype rows cols

private def tileInfoToGlobalParamTmaType? : TileInfo → Option GlobalParamTmaType
  | { kind := .ST, dtype := dtype, rows := rows, cols := cols, .. } =>
      some (.st dtype rows cols)
  | { kind := .STArray, dtype := dtype, rows := rows, cols := cols, .. } =>
      some (.st dtype rows cols)
  | { kind := .STRowVec, dtype := dtype, rows := rows, cols := cols, .. } =>
      some (.rowVecSt dtype rows cols)
  | { kind := .STColVec, dtype := dtype, rows := rows, cols := cols, .. } =>
      some (.colVecSt dtype rows cols)
  | { kind := .STColVecArray, dtype := dtype, rows := rows, cols := cols, .. } =>
      some (.colVecSt dtype rows cols)
  | _ => none

private def insertGlobalParamTmaType
    (acc : Std.HashMap Nat (Array GlobalParamTmaType))
    (paramIdx : Nat) (tmaTy : GlobalParamTmaType) :
    Std.HashMap Nat (Array GlobalParamTmaType) :=
  let existing := match acc[paramIdx]? with
    | some tys => tys
    | none => #[]
  if existing.contains tmaTy then acc
  else acc.insert paramIdx (existing.push tmaTy)

private def insertParamTmaFromTile
    (params : Array KParam)
    (tileInfo : Std.HashMap VarId TileInfo)
    (acc : Std.HashMap Nat (Array GlobalParamTmaType))
    (paramVar tileVar : VarId) :
    Std.HashMap Nat (Array GlobalParamTmaType) :=
  let paramIdx := paramVar.idx
  if h : paramIdx < params.size then
    let param := params[paramIdx]
    if !param.isPointer then acc
    else
      match tileInfo[tileVar]? >>= tileInfoToGlobalParamTmaType? with
      | some tmaTy => insertGlobalParamTmaType acc paramIdx tmaTy
      | none => acc
  else
    acc

private partial def collectGlobalParamTmaTypesStmt
    (params : Array KParam)
    (tileInfo : Std.HashMap VarId TileInfo)
    (acc : Std.HashMap Nat (Array GlobalParamTmaType)) :
    KStmt → Std.HashMap Nat (Array GlobalParamTmaType)
  | .loadGlobalAsync dst src _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .loadGlobalAsyncWarp dst src _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .loadGlobalAsyncIdx dst _ src _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .loadGlobalAsyncIdxSemIdx dst _ src _ _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .loadGlobalAsyncWarpIdx dst _ src _ _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .storeGlobalAsync dst src _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc dst src
  | .storeGlobalAsyncIdx dst src _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc dst src
  | .storeGlobalAdd dst src _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc dst src
  | .storeGlobalAddWarp dst src _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc dst src
  | .requireGlobalTma ptr descriptor =>
      insertGlobalParamTmaType acc ptr.idx (descriptorToGlobalParamTmaType descriptor)
  | .clusterTmaLoad dst src _ _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc src dst
  | .clusterTmaStore dst src _ _ _ _ =>
      insertParamTmaFromTile params tileInfo acc dst src
  | .forLoop _ _ _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | .forLoopVal _ _ _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | .forLoopRev _ _ _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | .forLoopValRev _ _ _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | .forLoopStrideVal _ _ _ _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
      elseBody.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc'
  | .ifWarpGroup _ body =>
      body.foldl (collectGlobalParamTmaTypesStmt params tileInfo) acc
  | _ => acc

/-- Infer which kernel pointer params require ThunderKittens TMA descriptor
    types, based on the shared tiles they participate with in TMA ops. -/
def inferGlobalParamTmaTypes (k : Kernel) : Std.HashMap Nat (Array GlobalParamTmaType) :=
  let tileInfo := inferTileInfo k
  k.body.foldl (collectGlobalParamTmaTypesStmt k.params tileInfo) {}

/-- Render the concrete `gl<...>` type for a kernel pointer parameter, including
    any inferred TMA descriptor types. -/
def renderGlobalParamCppType
    (p : KParam) (tmaTypes : Array GlobalParamTmaType := #[]) : String :=
  let tmaSuffix :=
    if tmaTypes.isEmpty then ""
    else ", " ++ String.intercalate ", " (tmaTypes.toList.map GlobalParamTmaType.toCpp)
  -- All four dims are runtime so the TMA descriptor (and gl operator[]) reflect
  -- the actual tensor shape. Hard-coding batch/depth to compile-time 1 silently
  -- discarded the launcher's `v{idx}_shape[0,1]` arguments (TK's
  -- `make_unsafe_gl_arg<1>` returns nullptr), which made multi-batch /
  -- multi-head kernels (e.g. decode) TMA-OOB-zero-fill any (batch>0, head>0)
  -- coordinate.
  if p.dtype == .Int64 then
    "tyr_raw_gl<int64_t>"
  else
    s!"gl<{p.dtype.toCpp}, -1, -1, -1, -1{tmaSuffix}>"

private def renderGlobalCoord
    (tileInfo : Std.HashMap VarId TileInfo)
    (tileVar coordB coordD coordR coordC : VarId) : String :=
  match tileInfo[tileVar]? with
  | some { kind := .STRowVec, .. } =>
      s!"kittens::coord<std::remove_reference_t<decltype({tileVar.toIdent})>>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"
  | some { kind := .STColVec, .. } =>
      s!"kittens::coord<std::remove_reference_t<decltype({tileVar.toIdent})>>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"
  | some info =>
      s!"kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {info.rows}), ({coordC.toIdent} * {info.cols}))"
  | none =>
      s!"kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"

private def renderGlobalCoordIdx
    (tileInfo : Std.HashMap VarId TileInfo)
    (tileVar idx coordB coordD coordR coordC : VarId) : String :=
  match tileInfo[tileVar]? with
  | some { kind := .STArray, .. } =>
      s!"kittens::coord<std::remove_reference_t<decltype({tileVar.toIdent}[{idx.toIdent}])>>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"
  | some { kind := .STColVecArray, .. } =>
      s!"kittens::coord<std::remove_reference_t<decltype({tileVar.toIdent}[{idx.toIdent}])>>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"
  | some info =>
      s!"kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {info.rows}), ({coordC.toIdent} * {info.cols}))"
  | none =>
      s!"kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent})"

/-- Generate C++ for a single statement -/
partial def generateStmt (rvLayouts : Std.HashMap VarId RVLayout)
    (rvVars : Std.HashSet VarId)
    (tileInfo : Std.HashMap VarId TileInfo)
    (useDynamicShared : Bool := false)
    (indent : String := "  ") : KStmt → String
  -- Declarations
  | .declRT v dtype rows cols layout =>
    s!"{indent}rt<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declST v dtype rows cols layout =>
    let ty := s!"st<{dtype.toCpp}, {rows}, {cols}>"
    if useDynamicShared then
      match layout with
      | .Row =>
        s!"{indent}auto &{v.toIdent} = al.allocate<{ty}>(); // layout: row_l\n"
      | .Col =>
        s!"{indent}auto &{v.toIdent} = al.allocate<{ty}>(); // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)\n"
    else
      -- ThunderKittens shared tiles do not carry a row/col layout template
      -- parameter (unlike rt<>): the `st<T, rows, cols, swizzle, swizzle_bytes>`
      -- template only has swizzle parameters. Row- vs column-major traversal
      -- is handled by the consuming ops (swap_layout, transpose, MMA transpose
      -- flags, or by transposing the register tile that loads into/out of
      -- this shared tile). We preserve the Lean-level layout annotation as a
      -- source comment so the emitted C++ documents the intended semantics.
      match layout with
      | .Row =>
        s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent}; // layout: row_l\n"
      | .Col =>
        -- TK has no native col-layout shared tile; the tile is physically
        -- row-major in SMEM and the Lean `.Col` annotation indicates the
        -- producer/consumer loads it transposed (see RT col_l ops that pair
        -- with this ST). Emit a tag comment so callers can audit this.
        s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent}; // layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)\n"
  | .declSTArray v dtype rows cols layout len =>
    let ty := s!"st<{dtype.toCpp}, {rows}, {cols}>"
    let layoutComment := match layout with
      | .Row => "layout: row_l"
      | .Col => "layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)"
    if useDynamicShared then
      s!"{indent}{ty} (&{v.toIdent})[{len}] = al.allocate<{ty}, {len}>(); // {layoutComment}\n"
    else
      s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent}[{len}]; // {layoutComment}\n"
  | .declSTAlias v dtype rows cols layout src comment =>
    let ty := s!"st<{dtype.toCpp}, {rows}, {cols}>"
    let layoutComment := match layout with
      | .Row => "layout: row_l"
      | .Col => "layout: col_l (Tyr .Col; TK has no col-layout ST, traversed transposed)"
    let aliasComment :=
      if comment.isEmpty then layoutComment else s!"{layoutComment}; {comment}"
    s!"{indent}auto &{v.toIdent} = *reinterpret_cast<{ty}*>(&{src.toIdent}); // {aliasComment}\n"
  | .declRV v dtype len =>
    s!"{indent}rv<{dtype.toCpp}, {len}{rvLayoutSuffix rvLayouts v}> {v.toIdent};\n"
  | .declSV v dtype len =>
    s!"{indent}__shared__ sv<{dtype.toCpp}, {len}> {v.toIdent};\n"
  | .declSTRowVec v dtype rows cols =>
    let ty := s!"row_vec<st<{dtype.toCpp}, {rows}, {cols}>>"
    if useDynamicShared then
      s!"{indent}auto &{v.toIdent} = al.allocate<{ty}>();\n"
    else
      s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent};\n"
  | .declSTColVec v dtype rows cols =>
    let ty := s!"col_vec<st<{dtype.toCpp}, {rows}, {cols}>>"
    if useDynamicShared then
      s!"{indent}auto &{v.toIdent} = al.allocate<{ty}>();\n"
    else
      s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent};\n"
  | .declSTColVecArray v dtype rows cols len =>
    let ty := s!"col_vec<st<{dtype.toCpp}, {rows}, {cols}>>"
    if useDynamicShared then
      s!"{indent}{ty} (&{v.toIdent})[{len}] = al.allocate<{ty}, {len}>();\n"
    else
      s!"{indent}__shared__ KITTENS_ALIGN_AS(1024) {ty} {v.toIdent}[{len}];\n"
  | .declTT v dtype rows cols =>
    s!"{indent}tt<{dtype.toCpp}, {rows}, {cols}> {v.toIdent};\n"
  | .declTMEMPool v numCtas clusterSize =>
    s!"{indent}tensor_allocator<{numCtas}, {clusterSize}> {v.toIdent}" ++ "{};\n"
  | .declSemaphore v =>
    s!"{indent}__shared__ semaphore {v.toIdent};\n"
  | .declSemaphoreArray v len =>
    s!"{indent}__shared__ semaphore {v.toIdent}[{len}];\n"

  -- Kernel parameter declarations (these are part of the signature, not body)
  -- When they appear in the body, just emit a comment for debugging
  | .declGPtr v dtype name =>
    s!"{indent}// param: {dtype.toCpp}* {name} (v{v.idx})\n"
  | .declKVal v dtype name =>
    s!"{indent}// param: {dtype.toCpp} {name} (v{v.idx})\n"
  | .declScalar v ty init =>
    let initExpr := match init with
      | some src => src.toIdent
      | none => ty.zeroCpp
    s!"{indent}{ty.toCpp} {v.toIdent} = {initExpr};\n"

  -- Memory operations
  | .load dst src => s!"{indent}warp::load({dst.toIdent}, {src.toIdent});\n"
  | .loadSubtile dst src rows cols rowIdx colIdx =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}.template subtile<{rows}, {cols}>(make_int2({rowIdx}, {colIdx})));\n"
  | .loadSubtileVal dst src rows cols rowIdx colIdx =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}.template subtile<{rows}, {cols}>(make_int2({rowIdx.toIdent}, {colIdx.toIdent})));\n"
  | .warpgroupLoad dst src => s!"{indent}group<4>::load({dst.toIdent}, {src.toIdent});\n"
  | .store dst src => s!"{indent}warp::store({dst.toIdent}, {src.toIdent});\n"
  | .loadAsync dst src => s!"{indent}warp::load_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAsync dst src => s!"{indent}warp::store_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAdd dst src => s!"{indent}store_add({dst.toIdent}, {src.toIdent});\n"
  | .storeAddAsync dst src => s!"{indent}warp::tma::store_add_async({dst.toIdent}, {src.toIdent});\n"
  | .storeMinAsync dst src => s!"{indent}warp::tma::store_min_async({dst.toIdent}, {src.toIdent});\n"
  | .warpgroupStore dst src => s!"{indent}warpgroup::store({dst.toIdent}, {src.toIdent});\n"
  | .warpgroupStoreIdx dst dstIdx src => s!"{indent}warpgroup::store({dst.toIdent}[{dstIdx.toIdent}], {src.toIdent});\n"
  | .tmaStoreCommitGroup => s!"{indent}warp::tma::store_commit_group();\n"
  | .tmaStoreAsyncWait => s!"{indent}warp::tma::store_async_wait();\n"
  | .prefetch src => s!"{indent}warp::tma::prefetch({src.toIdent});\n"
  | .tmaExpect barrier bytes => s!"{indent}warp::tma::expect_bytes({barrier.toIdent}, {bytes});\n"
  | .blockSync => s!"{indent}__syncthreads();\n"
  | .groupSync warps barrierId => s!"{indent}group<{warps}>::sync({barrierId});\n"
  | .groupSyncVal _ barrierId => s!"{indent}warpgroup::sync({barrierId.toIdent});\n"

  -- TMA operations with global pointers
  | .tmaLoad dst src coord =>
    s!"{indent}tma_load({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"
  | .tmaStore dst src coord =>
    s!"{indent}tma_store({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"

  -- Global memory operations with 4D coordinates (ThunderKittens style)
  | .loadGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo dst coordB coordD coordR coordC
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .warpgroupLoadGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo dst coordB coordD coordR coordC
    s!"{indent}group<4>::load({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .loadRegisterGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo dst coordB coordD coordR coordC
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .storeGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .warpgroupStoreGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}group<4>::store({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .storeRegisterGlobal dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .loadGlobalAsync dst src coordB coordD coordR coordC sem =>
    let coord := renderGlobalCoord tileInfo dst coordB coordD coordR coordC
    s!"{indent}if (threadIdx.x == 0) \{\n" ++
    s!"{indent}  tma::load_async({dst.toIdent}, {src.toIdent}, {coord}, {sem.toIdent});\n" ++
    s!"{indent}}\n"
  | .loadGlobalAsyncWarp dst src coordB coordD coordR coordC sem =>
    let coord := renderGlobalCoord tileInfo dst coordB coordD coordR coordC
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, {coord}, {sem.toIdent});\n"
  | .loadGlobalAsyncIdx dst dstIdx src coordB coordD coordR coordC sem =>
    let coord := renderGlobalCoordIdx tileInfo dst dstIdx coordB coordD coordR coordC
    s!"{indent}if (threadIdx.x == 0) \{\n" ++
    s!"{indent}  tma::load_async({dst.toIdent}[{dstIdx.toIdent}], {src.toIdent}, {coord}, {sem.toIdent});\n" ++
    s!"{indent}}\n"
  | .loadGlobalAsyncIdxSemIdx dst dstIdx src coordB coordD coordR coordC sem semIdx =>
    let coord := renderGlobalCoordIdx tileInfo dst dstIdx coordB coordD coordR coordC
    s!"{indent}if (threadIdx.x == 0) \{\n" ++
    s!"{indent}  tma::load_async({dst.toIdent}[{dstIdx.toIdent}], {src.toIdent}, {coord}, {sem.toIdent}[{semIdx.toIdent}]);\n" ++
    s!"{indent}}\n"
  | .loadGlobalAsyncWarpIdx dst dstIdx src coordB coordD coordR coordC sem semIdx =>
    let coord := renderGlobalCoordIdx tileInfo dst dstIdx coordB coordD coordR coordC
    s!"{indent}warp::tma::load_async({dst.toIdent}[{dstIdx.toIdent}], {src.toIdent}, {coord}, {sem.toIdent}[{semIdx.toIdent}]);\n"
  | .storeGlobalAsync dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}warp::tma::store_async({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .storeGlobalAsyncIdx dst src srcIdx coordB coordD coordR coordC =>
    let coord := renderGlobalCoordIdx tileInfo src srcIdx coordB coordD coordR coordC
    s!"{indent}warp::tma::store_async({dst.toIdent}, {src.toIdent}[{srcIdx.toIdent}], {coord});\n"
  | .storeGlobalAdd dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}if (kittens::warpid() == 0) \{\n" ++
    s!"{indent}  warp::tma::store_add_async({dst.toIdent}, {src.toIdent}, {coord});\n" ++
    s!"{indent}}\n"
  | .storeGlobalAddWarp dst src coordB coordD coordR coordC =>
    let coord := renderGlobalCoord tileInfo src coordB coordD coordR coordC
    s!"{indent}warp::tma::store_add_async({dst.toIdent}, {src.toIdent}, {coord});\n"
  | .requireGlobalTma _ _ =>
    ""
  | .layoutDim dst src .Batch =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}.batch();\n"
  | .layoutDim dst src .Depth =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}.depth();\n"
  | .layoutDim dst src .Rows =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}.rows();\n"
  | .layoutDim dst src .Cols =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}.cols();\n"

  -- Vector global memory operations
  | .loadVecGlobal dst src offset =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"
  | .storeVecGlobal dst src offset =>
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"
  | .storeVecGlobalAdd dst src offset =>
    s!"{indent}store_add({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"
  | .loadVecGlobalCoord dst src coordB coordD coordR coordC =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}));\n"
  | .storeVecGlobalCoord dst src coordB coordD coordR coordC =>
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}));\n"
  | .storeVecGlobalAddCoord dst src coordB coordD coordR coordC =>
    s!"{indent}store_add({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}));\n"
  | .loadScalarGlobal dst src offset =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}[{offset.toIdent}];\n"
  | .storeScalarGlobal dst src offset =>
    s!"{indent}{dst.toIdent}[{offset.toIdent}] = {src.toIdent};\n"
  | .loadFlatGlobal dst src offset srcDtype dstTy =>
    s!"{indent}{dstTy.toCpp} {dst.toIdent} = {renderFlatGlobalLoadExpr srcDtype dstTy src offset};\n"
  | .storeFlatGlobal dst offset src dstDtype srcTy =>
    s!"{indent}{dst.toIdent}.raw_ptr[{offset.toIdent}] = {renderFlatGlobalStoreExpr dstDtype srcTy src};\n"
  | .declSharedLinear v offsetElems ty =>
    s!"{indent}{ty.toCpp}* {v.toIdent} = reinterpret_cast<{ty.toCpp}*>(__shm) + {offsetElems.toIdent};\n"
  | .loadSharedLinear dst buf offset ty =>
    s!"{indent}{ty.toCpp} {dst.toIdent} = {buf.toIdent}[{offset.toIdent}];\n"
  | .storeSharedLinear buf offset src _ =>
    s!"{indent}{buf.toIdent}[{offset.toIdent}] = {src.toIdent};\n"

  -- Distributed / Multimem operations
  | .multimemLoadReduce op dst src =>
    s!"{indent}multimem::load_reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .multimemStore dst src =>
    s!"{indent}multimem::store({dst.toIdent}, {src.toIdent});\n"
  | .multimemRed op dst src =>
    s!"{indent}multimem::reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"

  -- MMA operations
  | .mma trans dst a b c =>
    s!"{indent}warp::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"
  | .mm trans dst a b =>
    s!"{indent}warp::mm_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .warpgroupMma trans dst a b =>
    s!"{indent}warpgroup::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .warpgroupMm trans dst a b =>
    s!"{indent}warpgroup::mm_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .warpgroupMmaIdx trans dst a aIdx b bIdx =>
    s!"{indent}warpgroup::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}[{aIdx.toIdent}], {b.toIdent}[{bIdx.toIdent}]);\n"
  | .warpgroupMmIdx trans dst a aIdx b bIdx =>
    s!"{indent}warpgroup::mm_{trans.toSuffix}({dst.toIdent}, {a.toIdent}[{aIdx.toIdent}], {b.toIdent}[{bIdx.toIdent}]);\n"
  | .warpgroupMmaRhsIdx trans dst a b bIdx =>
    s!"{indent}warpgroup::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}[{bIdx.toIdent}]);\n"
  | .mmaFence dst => s!"{indent}warpgroup::mma_fence({dst.toIdent});\n"
  | .mmaCommitGroup => s!"{indent}warpgroup::mma_commit_group();\n"
  | .mmaAsyncWait n => s!"{indent}warpgroup::mma_async_wait<{n}>();\n"

  -- Blackwell-specific MMA (tcgen05 / 2-CTA MMA)
  | .tcgen05Mm trans dst a b =>
    s!"{indent}warpgroup::mm2_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .tcgen05Mma trans dst a b c =>
    s!"{indent}warpgroup::mma2_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"
  | .tcgen05MmaScaled trans dst a b c scaleA scaleB =>
    s!"{indent}warpgroup::mma2_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent}, {scaleA.toIdent}, {scaleB.toIdent});\n"
  | .tcgen05Commit sem clusterSize =>
    s!"{indent}detail::tcgen05::commit<{clusterSize}>({sem.toIdent});\n"
  | .tensorLoadAsync dst src =>
    s!"{indent}warpgroup::load_async({dst.toIdent}, {src.toIdent});\n"
  | .tensorLoadWait =>
    s!"{indent}tensor_load_wait();\n"

  -- Tensor memory operations (Blackwell SM100)
  | .tmemAllocate dst pool dtype rows cols superlane offset =>
    let ty := s!"tt<{dtype.toCpp}, {rows}, {cols}>"
    if rows == 64 then
      s!"{indent}auto {dst.toIdent} = {pool.toIdent}.allocate<{ty}>({superlane}, {offset});\n"
    else
      s!"{indent}auto {dst.toIdent} = {pool.toIdent}.allocate<{ty}>({offset});\n"
  | .tmemProvision pool clusterSize =>
    s!"{indent}if(elect_one) {pool.toIdent}.provision<{clusterSize}>(tmem_addr);\n"
  | .tmemDeprovision pool =>
    s!"{indent}if(elect_one) {pool.toIdent}.deprovision();\n"
  | .loadScaleTmem dst src stage =>
    s!"{indent}load_mxnv_scale_async2({dst.toIdent}[{stage}], {src.toIdent});\n"
  | .tmemSubtile dst src offset =>
    s!"{indent}auto {dst.toIdent} = {src.toIdent}.subtile({offset});\n"

  -- Cluster operations (SM90+ clusters, SM100 2-CTA)
  | .clusterIdx dst axis =>
    s!"{indent}int {dst.toIdent} = clusterIdx().{match axis with | 0 => "x" | 1 => "y" | _ => "z"};\n"
  | .clusterTmaLoad dst src coordB coordD coordR coordC sem =>
    let (rowScale, colScale) := match tileInfo[dst]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}tma::cluster::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})), {sem.toIdent});\n"
  | .clusterTmaStore dst src coordB coordD coordR coordC =>
    let (rowScale, colScale) := match tileInfo[src]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}tma::cluster::store_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})));\n"
  | .clusterArrive sem =>
    s!"{indent}cluster::arrive({sem.toIdent});\n"
  | .clusterWait sem =>
    s!"{indent}cluster::wait({sem.toIdent});\n"

  -- Architecture-specific load variants
  | .cpAsyncLoad dst src coordB coordD coordR coordC _sem =>
    s!"{indent}warp::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}));\n"
  | .tmaLoadAsync dst src coordB coordD coordR coordC sem =>
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}), {sem.toIdent});\n"

  -- Element-wise unary
  | .unary .Neg dst src =>
    if dst == src then
      s!"{indent}warp::mul({dst.toIdent}, {src.toIdent}, -1.0f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::mul({dst.toIdent}, {dst.toIdent}, -1.0f);\n"
  | .unary .Zero dst _ =>
    s!"{indent}warp::zero({dst.toIdent});\n"
  | .unary .One dst _ =>
    s!"{indent}warp::one({dst.toIdent});\n"
  | .unary .PosInfty dst _ =>
    s!"{indent}warp::pos_infty({dst.toIdent});\n"
  | .unary .NegInfty dst _ =>
    s!"{indent}warp::neg_infty({dst.toIdent});\n"
  | .unary .Sqrt dst src =>
    if rvVars.contains dst && rvVars.contains src then
      let pre :=
        if dst == src then
          ""
        else
          s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
      pre ++
      s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [] __device__ (int _i, auto _x) \{\n" ++
      s!"{indent}  return static_cast<decltype(_x)>(::sqrtf(static_cast<float>(_x)));\n" ++
      s!"{indent}" ++ "});\n"
    else if dst == src then
      s!"{indent}warp::sqrt({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::sqrt({dst.toIdent}, {dst.toIdent});\n"
  | .unary .Rsqrt dst src =>
    if rvVars.contains dst && rvVars.contains src then
      let pre :=
        if dst == src then
          ""
        else
          s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
      pre ++
      s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [] __device__ (int _i, auto _x) \{\n" ++
      s!"{indent}  return static_cast<decltype(_x)>(::rsqrtf(static_cast<float>(_x)));\n" ++
      s!"{indent}" ++ "});\n"
    else if dst == src then
      s!"{indent}warp::rsqrt({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::rsqrt({dst.toIdent}, {dst.toIdent});\n"
  | .unary .Copy dst src =>
    s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
  | .unary op dst src =>
    if dst == src then
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {dst.toIdent});\n"

  -- Element-wise binary
  | .binary op dst a b =>
    if dst == a then
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {a.toIdent});\n" ++
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {dst.toIdent}, {b.toIdent});\n"

  -- Element-wise ternary (FMA)
  | .ternary op dst a b c =>
    s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"

  -- Element-wise comparison masks
  | .eqMask dst a b =>
    s!"{indent}tk_eq_mask({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Scalar operations
  | .scalarMul dst src scalar =>
    if dst == src then
      s!"{indent}warp::mul({dst.toIdent}, {src.toIdent}, {floatToCLiteral scalar}f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::mul({dst.toIdent}, {dst.toIdent}, {floatToCLiteral scalar}f);\n"
  | .scalarAdd dst src scalar =>
    if dst == src then
      s!"{indent}warp::add({dst.toIdent}, {src.toIdent}, {floatToCLiteral scalar}f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::add({dst.toIdent}, {dst.toIdent}, {floatToCLiteral scalar}f);\n"

  -- Broadcasting
  | .broadcast axis dst vec =>
    let suffix := match axis with
      | .Row => "_col"
      | .Col => "_row"
    s!"{indent}warp::broadcast{suffix}({dst.toIdent}, {vec.toIdent});\n"
  | .binaryBroadcast op axis dst tile vec =>
    let suffix := match axis with
      | .Row => "_col"
      | .Col => "_row"
    s!"{indent}warp::{op.toCpp}{suffix}({dst.toIdent}, {tile.toIdent}, {vec.toIdent});\n"

  -- Reductions
  | .reduce op axis dst src =>
    s!"{indent}warp::{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .reduceAccum op axis dst src accum =>
    s!"{indent}warp::{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent}, {accum.toIdent});\n"

  -- Scan operations
  | .cumsum axis dst src =>
    s!"{indent}warp::{axis.toPrefix}cumsum({dst.toIdent}, {src.toIdent});\n"
  | .cumprod axis dst src =>
    s!"{indent}warp::{axis.toPrefix}cumprod({dst.toIdent}, {src.toIdent});\n"

  -- Outer product
  | .outer dst a b =>
    s!"{indent}tk_outer({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Layout/type conversions
  | .swapLayout dst src => s!"{indent}warp::swap_layout({dst.toIdent}, {src.toIdent});\n"
  | .transpose dst src =>
    if dst == src then
      s!"{indent}warp::transpose_inplace({dst.toIdent});\n"
    else
      s!"{indent}warp::transpose_sep({dst.toIdent}, {src.toIdent});\n"
  | .convert dst src => s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
  | .convertVecLayout dst src => s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"

  -- Masking
  | .mask op dst src fillVal =>
    let fillStr := fillVal.map (fun v => s!", {v}") |>.getD ""
    match op with
    | .Tril d => s!"{indent}warp::tril({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .Triu d => s!"{indent}warp::triu({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .MakeCausal => s!"{indent}warp::make_causal({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .MakeCausalT => s!"{indent}warp::make_causal_t({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .RightFill c => s!"{indent}warp::right_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .LeftFill c => s!"{indent}warp::left_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .UpperFill r => s!"{indent}warp::upper_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .LowerFill r => s!"{indent}warp::lower_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .UpperRightFill r c => s!"{indent}warp::upper_right_fill({dst.toIdent}, {src.toIdent}, {r}, {c}{fillStr});\n"
  | .maskRightFillVal dst src colVar fillVal =>
    let fillStr := fillVal.map (fun v => s!", {v}") |>.getD ""
    s!"{indent}warp::right_fill({dst.toIdent}, {src.toIdent}, {colVar.toIdent}{fillStr});\n"

  -- Tile slicing
  | .sliceRows dst src startRow numRows =>
    match tileInfo[dst]?, tileInfo[src]? with
    | some dstInfo, some srcInfo =>
      match dstInfo.kind, srcInfo.kind with
      | .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_row_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_row_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_row_t>;\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {numRows}, \"slice rows: dst rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {srcInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}static_assert({startRow} + {numRows} <= {srcInfo.rows}, \"slice rows: out of bounds\");\n" ++
        s!"{indent}static_assert({startRow} % _tk_tile_rows == 0, \"slice rows: unaligned start\");\n" ++
        s!"{indent}static_assert({numRows} % _tk_tile_rows == 0, \"slice rows: unaligned size\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} % _tk_tile_cols == 0, \"slice rows: unaligned cols\");\n" ++
        s!"{indent}constexpr int _tk_start_tile = {startRow} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {numRows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_col_tiles = {dstInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_col_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {src.toIdent}.tiles[_tk_start_tile + _tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {numRows}, \"slice rows: dst rows mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.cols} == {dstInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}static_assert({startRow} + {numRows} <= {srcInfo.rows}, \"slice rows: out of bounds\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{numRows}, {srcInfo.cols}>(make_int2({startRow}, 0));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | .RT, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {numRows}, \"slice rows: dst rows mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.cols} == {dstInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}static_assert({startRow} + {numRows} <= {srcInfo.rows}, \"slice rows: out of bounds\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{numRows}, {srcInfo.cols}>(make_int2({startRow}, 0));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | .ST, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({srcInfo.rows} == {numRows}, \"slice rows: src rows mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.cols} == {dstInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}static_assert({startRow} + {numRows} <= {dstInfo.rows}, \"slice rows: out of bounds\");\n" ++
        s!"{indent}auto _tk_dst_sub = {dst.toIdent}.template subtile<{numRows}, {dstInfo.cols}>(make_int2({startRow}, 0));\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_sub, {src.toIdent});\n" ++
        s!"{indent}}\n"
      | _, _ =>
        s!"{indent}static_assert(false, \"unsupported sliceRows between non-matching tile kinds\");\n"
    | _, _ =>
      s!"{indent}static_assert(false, \"unresolved tile info for sliceRows\");\n"
  | .sliceCols dst src startCol numCols =>
    match tileInfo[dst]?, tileInfo[src]? with
    | some dstInfo, some srcInfo =>
      match dstInfo.kind, srcInfo.kind with
      | .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_col_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_col_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_col_t>;\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {srcInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}static_assert({startCol} + {numCols} <= {srcInfo.cols}, \"slice cols: out of bounds\");\n" ++
        s!"{indent}static_assert({startCol} % _tk_tile_cols == 0, \"slice cols: unaligned start\");\n" ++
        s!"{indent}static_assert({numCols} % _tk_tile_cols == 0, \"slice cols: unaligned size\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} % _tk_tile_rows == 0, \"slice cols: unaligned rows\");\n" ++
        s!"{indent}constexpr int _tk_start_tile = {startCol} / _tk_tile_cols;\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {dstInfo.rows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_col_tiles = {numCols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_col_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {src.toIdent}.tiles[_tk_i][_tk_start_tile + _tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.rows} == {dstInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{srcInfo.rows}, {numCols}>(make_int2(0, {startCol}));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | .RT, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.rows} == {dstInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{srcInfo.rows}, {numCols}>(make_int2(0, {startCol}));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | .ST, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.rows} == {dstInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}auto _tk_dst_sub = {dst.toIdent}.template subtile<{dstInfo.rows}, {numCols}>(make_int2(0, {startCol}));\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_sub, {src.toIdent});\n" ++
        s!"{indent}}\n"
      | _, _ =>
        s!"{indent}static_assert(false, \"unsupported sliceCols between non-matching tile kinds\");\n"
    | _, _ =>
      s!"{indent}static_assert(false, \"unresolved tile info for sliceCols\");\n"
  | .concatCols dst left right =>
    match tileInfo[dst]?, tileInfo[left]?, tileInfo[right]? with
    | some dstInfo, some leftInfo, some rightInfo =>
      match dstInfo.kind, leftInfo.kind, rightInfo.kind with
      | .RT, .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_concat_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_concat_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_concat_t>;\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} % _tk_tile_rows == 0, \"concat cols: unaligned rows\");\n" ++
        s!"{indent}static_assert({leftInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned left cols\");\n" ++
        s!"{indent}static_assert({rightInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned right cols\");\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {dstInfo.rows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_left_tiles = {leftInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}constexpr int _tk_right_tiles = {rightInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_left_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {left.toIdent}.tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_right_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_left_tiles + _tk_j] = {right.toIdent}.tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}auto _tk_dst_left = {dst.toIdent}.template subtile<{dstInfo.rows}, {leftInfo.cols}>(make_int2(0, 0));\n" ++
        s!"{indent}auto _tk_dst_right = {dst.toIdent}.template subtile<{dstInfo.rows}, {rightInfo.cols}>(make_int2(0, {leftInfo.cols}));\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_left, {left.toIdent});\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_right, {right.toIdent});\n" ++
        s!"{indent}}\n"
      | .ST, _, _ =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}auto _tk_dst_left = {dst.toIdent}.template subtile<{dstInfo.rows}, {leftInfo.cols}>(make_int2(0, 0));\n" ++
        s!"{indent}auto _tk_dst_right = {dst.toIdent}.template subtile<{dstInfo.rows}, {rightInfo.cols}>(make_int2(0, {leftInfo.cols}));\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_left, {left.toIdent});\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_right, {right.toIdent});\n" ++
        s!"{indent}}\n"
      | .RT, _, _ =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_concat_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_concat_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_concat_t>;\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} % _tk_tile_rows == 0, \"concat cols: unaligned rows\");\n" ++
        s!"{indent}static_assert({leftInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned left cols\");\n" ++
        s!"{indent}static_assert({rightInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned right cols\");\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {dstInfo.rows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_left_tiles = {leftInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}constexpr int _tk_right_tiles = {rightInfo.cols} / _tk_tile_cols;\n" ++
        (if leftInfo.kind == .ST then
          s!"{indent}rt<{dstInfo.dtype.toCpp}, {leftInfo.rows}, {leftInfo.cols}, {dstInfo.layout.toCpp}> _tk_left_rt;\n" ++
          s!"{indent}auto _tk_left_sub = {left.toIdent}.template subtile<{leftInfo.rows}, {leftInfo.cols}>(make_int2(0, 0));\n" ++
          s!"{indent}kittens::warp::copy(_tk_left_rt, _tk_left_sub);\n"
         else "") ++
        (if rightInfo.kind == .ST then
          s!"{indent}rt<{dstInfo.dtype.toCpp}, {rightInfo.rows}, {rightInfo.cols}, {dstInfo.layout.toCpp}> _tk_right_rt;\n" ++
          s!"{indent}auto _tk_right_sub = {right.toIdent}.template subtile<{rightInfo.rows}, {rightInfo.cols}>(make_int2(0, 0));\n" ++
          s!"{indent}kittens::warp::copy(_tk_right_rt, _tk_right_sub);\n"
         else "") ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_left_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = " ++
          (if leftInfo.kind == .ST then "_tk_left_rt" else left.toIdent) ++ s!".tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_right_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_left_tiles + _tk_j] = " ++
          (if rightInfo.kind == .ST then "_tk_right_rt" else right.toIdent) ++ s!".tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | _, _, _ =>
        s!"{indent}static_assert(false, \"unsupported concatCols between non-matching tile kinds\");\n"
    | _, _, _ =>
      s!"{indent}static_assert(false, \"unresolved tile info for concatCols\");\n"

  -- Synchronization
  | .sync barrierId => s!"{indent}warp::sync({barrierId});\n"
  | .arrive barrierId => s!"{indent}warp::arrive({barrierId});\n"
  | .arriveAndWait barrierId => s!"{indent}warp::arrive_and_wait(kittens::barrier<1>({barrierId}));\n"

  -- Named barriers (for FA3 warp specialization)
  | .namedBarrierSync id numThreads =>
    let numWarps := numThreads / 32
    s!"{indent}kittens::arrive_and_wait(kittens::barrier<{numWarps}>({id}));\n"
  | .namedBarrierArrive id numThreads =>
    let numWarps := numThreads / 32
    s!"{indent}kittens::arrive(kittens::barrier<{numWarps}>({id}));\n"

  -- Warp group operations (for FA3 warp specialization)
  | .warpGroupIdx dst =>
    s!"{indent}int {dst.toIdent} = kittens::warpgroup::groupid();\n"
  | .warpGroupLaneId dst =>
    s!"{indent}int {dst.toIdent} = kittens::warpgroup::laneid();\n"
  | .warpId dst =>
    s!"{indent}int {dst.toIdent} = kittens::warpid();\n"
  | .laneId dst =>
    s!"{indent}int {dst.toIdent} = kittens::laneid();\n"
  | .electOneSync dst =>
    s!"{indent}bool {dst.toIdent} = (kittens::laneid() == (__ffs(__activemask()) - 1));\n"
  | .warpgroupDecreaseRegisters n =>
    s!"{indent}warpgroup::decrease_registers<{n}>();\n"
  | .warpgroupIncreaseRegisters n =>
    s!"{indent}warpgroup::increase_registers<{n}>();\n"

  -- Fence operations (for WGMMA pipelining)
  | .fenceViewAsyncShared =>
    s!"{indent}__syncwarp();\n{indent}kittens::tma::fence_view_async_shared();\n"
  | .fenceProxyAsync =>
    s!"{indent}kittens::tma::fence_proxy_async();\n"

  -- Semaphore operations
  | .semaphore op sem =>
    match op with
    | .Init threadCount transactionCount =>
      s!"{indent}if (threadIdx.x == 0) \{\n" ++
      s!"{indent}  init_semaphore({sem.toIdent}, {threadCount}, {transactionCount});\n" ++
      s!"{indent}}\n"
    | .Invalidate => s!"{indent}invalidate_semaphore({sem.toIdent});\n"
    | .Expect bytes =>
      s!"{indent}if (threadIdx.x == 0) \{\n" ++
      s!"{indent}  tma::expect_bytes({sem.toIdent}, {bytes});\n" ++
      s!"{indent}}\n"
    | .Wait phase => s!"{indent}wait({sem.toIdent}, {phase});\n"
    | .Arrive count => s!"{indent}arrive({sem.toIdent}, {count});\n"
    | .ArriveAndWait => s!"{indent}arrive_and_wait({sem.toIdent});\n"
  | .semaphoreWarp op sem =>
    match op with
    | .Init threadCount transactionCount =>
      s!"{indent}init_semaphore({sem.toIdent}, {threadCount}, {transactionCount});\n"
    | .Invalidate => s!"{indent}invalidate_semaphore({sem.toIdent});\n"
    | .Expect bytes =>
      s!"{indent}warp::tma::expect_bytes({sem.toIdent}, {bytes});\n"
    | .Wait phase => s!"{indent}wait({sem.toIdent}, {phase});\n"
    | .Arrive count => s!"{indent}arrive({sem.toIdent}, {count});\n"
    | .ArriveAndWait => s!"{indent}arrive_and_wait({sem.toIdent});\n"
  | .semaphoreWaitVal sem phase =>
    s!"{indent}wait({sem.toIdent}, {phase.toIdent});\n"
  | .semaphoreArray op sem idx =>
    match op with
    | .Init threadCount transactionCount =>
      s!"{indent}if (threadIdx.x == 0) \{\n" ++
      s!"{indent}  init_semaphore({sem.toIdent}[{idx.toIdent}], {threadCount}, {transactionCount});\n" ++
      s!"{indent}}\n"
    | .Invalidate => s!"{indent}invalidate_semaphore({sem.toIdent}[{idx.toIdent}]);\n"
    | .Expect bytes =>
      s!"{indent}if (threadIdx.x == 0) \{\n" ++
      s!"{indent}  tma::expect_bytes({sem.toIdent}[{idx.toIdent}], {bytes});\n" ++
      s!"{indent}}\n"
    | .Wait phase => s!"{indent}wait({sem.toIdent}[{idx.toIdent}], {phase});\n"
    | .Arrive count => s!"{indent}arrive({sem.toIdent}[{idx.toIdent}], {count});\n"
    | .ArriveAndWait => s!"{indent}arrive_and_wait({sem.toIdent}[{idx.toIdent}]);\n"
  | .semaphoreArrayWarp op sem idx =>
    match op with
    | .Init threadCount transactionCount =>
      s!"{indent}init_semaphore({sem.toIdent}[{idx.toIdent}], {threadCount}, {transactionCount});\n"
    | .Invalidate => s!"{indent}invalidate_semaphore({sem.toIdent}[{idx.toIdent}]);\n"
    | .Expect bytes =>
      s!"{indent}warp::tma::expect_bytes({sem.toIdent}[{idx.toIdent}], {bytes});\n"
    | .Wait phase => s!"{indent}wait({sem.toIdent}[{idx.toIdent}], {phase});\n"
    | .Arrive count => s!"{indent}arrive({sem.toIdent}[{idx.toIdent}], {count});\n"
    | .ArriveAndWait => s!"{indent}arrive_and_wait({sem.toIdent}[{idx.toIdent}]);\n"
  | .semaphoreArrayWaitVal sem idx phase =>
    s!"{indent}wait({sem.toIdent}[{idx.toIdent}], {phase.toIdent});\n"
  | .semaphoreArrayArrive sem idx count =>
    s!"{indent}arrive({sem.toIdent}[{idx.toIdent}], {count});\n"

  -- Control flow
  | .forLoop v lo hi body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {lo}; {v.toIdent} < {hi}; {v.toIdent}++) \{\n{bodyStr}{indent}}\n"
  | .forLoopVal v lo hi body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {lo}; {v.toIdent} < {hi.toIdent}; {v.toIdent}++) \{\n{bodyStr}{indent}}\n"
  | .forLoopRev v lo hi body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {hi}; {v.toIdent}-- > {lo}; ) \{\n{bodyStr}{indent}}\n"
  | .forLoopValRev v lo hi body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {hi.toIdent}; {v.toIdent}-- > {lo}; ) \{\n{bodyStr}{indent}}\n"
  | .forLoopStrideVal v start hi step body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (auto {v.toIdent} = {start.toIdent}; {v.toIdent} < {hi.toIdent}; {v.toIdent} += {step.toIdent}) \{\n{bodyStr}{indent}}\n"
  | .ifStmt cond thenBody elseBody =>
    let thenStr := thenBody.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    let elseStr := elseBody.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    if elseBody.isEmpty then
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}}\n"
    else
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}} else \{\n{elseStr}{indent}}\n"
  | .ifWarpGroup wgIdx body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo useDynamicShared (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}if (kittens::warpgroup::groupid() == {wgIdx}) \{\n{bodyStr}{indent}}\n"
  | .comment text => s!"{indent}// {text}\n"

  -- Block/thread index accessors
  | .getBlockIdx dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = blockIdx.{axisName};\n"
  | .getThreadIdx dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = threadIdx.{axisName};\n"
  | .getGridDim dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = gridDim.{axisName};\n"
  | .getBlockDim dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = blockDim.{axisName};\n"

  -- Constants
  | .constInt dst value =>
    s!"{indent}int {dst.toIdent} = {value};\n"
  | .constUInt64 dst value =>
    s!"{indent}uint64_t {dst.toIdent} = {value}ULL;\n"
  | .constFloat dst value =>
    s!"{indent}float {dst.toIdent} = {floatToCLiteral value}f;\n"
  | .scalarUnary .Neg dst src =>
    s!"{indent}auto {dst.toIdent} = -{src.toIdent};\n"
  | .scalarUnary .Exp dst src =>
    s!"{indent}auto {dst.toIdent} = ::expf(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Exp2 dst src =>
    s!"{indent}auto {dst.toIdent} = ::exp2f(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Log dst src =>
    s!"{indent}auto {dst.toIdent} = ::logf(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Log2 dst src =>
    s!"{indent}auto {dst.toIdent} = ::log2f(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Rsqrt dst src =>
    s!"{indent}auto {dst.toIdent} = ::rsqrtf(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Sqrt dst src =>
    s!"{indent}auto {dst.toIdent} = ::sqrtf(static_cast<float>({src.toIdent}));\n"
  | .scalarUnary .Cos dst src =>
    s!"{indent}auto {dst.toIdent} = ::cosf(static_cast<float>({src.toIdent}));\n"
  | .scalarCompare .Eq dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} == {b.toIdent});\n"
  | .scalarCompare .Lt dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} < {b.toIdent});\n"
  | .scalarCompare .Le dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} <= {b.toIdent});\n"
  | .scalarCompare .Gt dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} > {b.toIdent});\n"
  | .scalarCompare .Ge dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} >= {b.toIdent});\n"
  | .scalarBinary .Add dst a b =>
    s!"{indent}auto {dst.toIdent} = {a.toIdent} + {b.toIdent};\n"
  | .scalarBinary .Sub dst a b =>
    s!"{indent}auto {dst.toIdent} = {a.toIdent} - {b.toIdent};\n"
  | .scalarBinary .Mul dst a b =>
    s!"{indent}auto {dst.toIdent} = {a.toIdent} * {b.toIdent};\n"
  | .scalarBinary .Div dst a b =>
    s!"{indent}auto {dst.toIdent} = {a.toIdent} / {b.toIdent};\n"
  | .scalarBinary .Mod dst a b =>
    s!"{indent}auto {dst.toIdent} = {a.toIdent} % {b.toIdent};\n"
  | .scalarBinary .Min dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} < {b.toIdent}) ? {a.toIdent} : {b.toIdent};\n"
  | .scalarBinary .Max dst a b =>
    s!"{indent}auto {dst.toIdent} = ({a.toIdent} > {b.toIdent}) ? {a.toIdent} : {b.toIdent};\n"
  | .scalarSelect dst cond ifTrue ifFalse =>
    s!"{indent}auto {dst.toIdent} = {cond.toIdent} ? {ifTrue.toIdent} : {ifFalse.toIdent};\n"
  | .scalarAssign dst src =>
    s!"{indent}{dst.toIdent} = {src.toIdent};\n"
  | .scalarCast dst src dstTy =>
    s!"{indent}{dstTy.toCpp} {dst.toIdent} = static_cast<{dstTy.toCpp}>({src.toIdent});\n"
  | .blockReduce .Sum dst src .Float32 =>
    let stem := dst.toIdent
    let x := s!"_tyr_reduce_x_{stem}"
    let offset := s!"_tyr_reduce_offset_{stem}"
    let warps := s!"_tyr_reduce_warps_{stem}"
    let out := s!"_tyr_reduce_out_{stem}"
    let lane := s!"_tyr_reduce_lane_{stem}"
    let warp := s!"_tyr_reduce_warp_{stem}"
    s!"{indent}float {dst.toIdent};\n" ++
    s!"{indent}\{\n" ++
    s!"{indent}  float {x} = static_cast<float>({src.toIdent});\n" ++
    s!"{indent}  for (int {offset} = warpSize / 2; {offset} > 0; {offset} /= 2) \{\n" ++
    s!"{indent}    {x} += __shfl_down_sync(0xffffffff, {x}, {offset});\n" ++
    s!"{indent}  }\n" ++
    s!"{indent}  __shared__ float {warps}[32];\n" ++
    s!"{indent}  __shared__ float {out};\n" ++
    s!"{indent}  const int {lane} = threadIdx.x & 31;\n" ++
    s!"{indent}  const int {warp} = threadIdx.x >> 5;\n" ++
    s!"{indent}  if ({lane} == 0) {warps}[{warp}] = {x};\n" ++
    s!"{indent}  __syncthreads();\n" ++
    s!"{indent}  {x} = (threadIdx.x < ((blockDim.x + 31) / 32)) ? {warps}[{lane}] : 0.0f;\n" ++
    s!"{indent}  if ({warp} == 0) \{\n" ++
    s!"{indent}    for (int {offset} = warpSize / 2; {offset} > 0; {offset} /= 2) \{\n" ++
    s!"{indent}      {x} += __shfl_down_sync(0xffffffff, {x}, {offset});\n" ++
    s!"{indent}    }\n" ++
    s!"{indent}    if ({lane} == 0) {out} = {x};\n" ++
    s!"{indent}  }\n" ++
    s!"{indent}  __syncthreads();\n" ++
    s!"{indent}  {dst.toIdent} = {out};\n" ++
    s!"{indent}  __syncthreads();\n" ++
    s!"{indent}}\n"
  | .blockReduce op _ _ ty =>
    s!"{indent}static_assert(false, \"unsupported blockReduce op/type: {repr op} {repr ty}\");\n"
  | .vecIota dst start step =>
    s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [] __device__ (int _i, auto _x) \{\n" ++
    s!"{indent}  return static_cast<decltype(_x)>({floatToCLiteral start}f + {floatToCLiteral step}f * static_cast<float>(_i));\n" ++
    s!"{indent}" ++ "});\n"
  | .vecFillScalar dst scalar =>
    s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [&] __device__ (int _i, auto _x) \{\n" ++
    s!"{indent}  return static_cast<decltype(_x)>({scalar.toIdent});\n" ++
    s!"{indent}" ++ "});\n"
  | .raw code =>
    indentRaw indent code ++ "\n"

/-- Generate kernel parameter list. -/
def generateParams (k : Kernel) : String :=
  let paramTmaTypes := inferGlobalParamTmaTypes k
  let paramStrs := Id.run do
    let mut out : List String := []
    for h : idx in [:k.params.size] do
      let p := k.params[idx]
      if p.isPointer then
        let tmaTypes := match paramTmaTypes[idx]? with
          | some tys => tys
          | none => #[]
        let cppTy := renderGlobalParamCppType p tmaTypes
        if tmaTypes.isEmpty then
          out := out.concat s!"{cppTy} v{idx}"
        else
          out := out.concat s!"const __grid_constant__ {cppTy} v{idx}"
      else
        out := out.concat s!"{p.scalarTy.toCpp} v{idx}"
    return out
  String.intercalate ", " paramStrs

private partial def stmtUses (p : KStmt → Bool) : KStmt → Bool
  | stmt =>
    if p stmt then true else
      match stmt with
      -- Recursive cases: constructors whose payload contains `Array KStmt`.
      | .forLoop _ _ _ body => body.any (stmtUses p)
      | .forLoopVal _ _ _ body => body.any (stmtUses p)
      | .forLoopRev _ _ _ body => body.any (stmtUses p)
      | .forLoopValRev _ _ _ body => body.any (stmtUses p)
      | .forLoopStrideVal _ _ _ _ body => body.any (stmtUses p)
      | .ifStmt _ thenBody elseBody =>
        thenBody.any (stmtUses p) || elseBody.any (stmtUses p)
      | .ifWarpGroup _ body => body.any (stmtUses p)
      -- Leaf cases: no nested `KStmt`s. Listed explicitly so that adding a new
      -- `KStmt` constructor causes a non-exhaustive match error rather than
      -- silently falling through to `false`.
      | .declRT .. | .declST .. | .declSTArray .. | .declSTAlias .. | .declRV .. | .declSV ..
      | .declSTRowVec .. | .declSTColVec .. | .declSTColVecArray ..
      | .declSemaphore .. | .declSemaphoreArray ..
      | .declTT .. | .declTMEMPool ..
      | .declGPtr .. | .declKVal ..
      | .declScalar ..
      | .load .. | .loadSubtile .. | .loadSubtileVal .. | .store .. | .loadAsync .. | .storeAsync ..
      | .storeAdd .. | .storeAddAsync .. | .storeMinAsync ..
      | .warpgroupLoad .. | .warpgroupStore .. | .warpgroupStoreIdx ..
      | .tmaStoreCommitGroup | .tmaStoreAsyncWait
      | .prefetch .. | .tmaExpect ..
      | .blockSync | .groupSync .. | .groupSyncVal ..
      | .tmaLoad .. | .tmaStore ..
      | .loadGlobal .. | .warpgroupLoadGlobal .. | .loadRegisterGlobal ..
      | .storeGlobal .. | .warpgroupStoreGlobal .. | .storeRegisterGlobal ..
      | .loadGlobalAsync .. | .loadGlobalAsyncWarp ..
      | .loadGlobalAsyncIdx .. | .loadGlobalAsyncIdxSemIdx .. | .loadGlobalAsyncWarpIdx ..
      | .storeGlobalAsync .. | .storeGlobalAsyncIdx ..
      | .storeGlobalAdd .. | .storeGlobalAddWarp ..
      | .requireGlobalTma ..
      | .layoutDim ..
      | .loadVecGlobal .. | .storeVecGlobal .. | .storeVecGlobalAdd ..
      | .loadVecGlobalCoord .. | .storeVecGlobalCoord .. | .storeVecGlobalAddCoord ..
      | .loadScalarGlobal .. | .storeScalarGlobal ..
      | .loadFlatGlobal .. | .storeFlatGlobal ..
      | .declSharedLinear .. | .loadSharedLinear .. | .storeSharedLinear ..
      | .multimemLoadReduce .. | .multimemStore .. | .multimemRed ..
      | .mma .. | .mm .. | .warpgroupMma .. | .warpgroupMm ..
      | .warpgroupMmaIdx .. | .warpgroupMmIdx .. | .warpgroupMmaRhsIdx ..
      | .mmaFence .. | .mmaCommitGroup | .mmaAsyncWait ..
      | .tcgen05Mm .. | .tcgen05Mma .. | .tcgen05MmaScaled .. | .tcgen05Commit ..
      | .tensorLoadAsync .. | .tensorLoadWait
      | .tmemAllocate .. | .tmemProvision .. | .tmemDeprovision ..
      | .loadScaleTmem .. | .tmemSubtile ..
      | .clusterIdx .. | .clusterTmaLoad .. | .clusterTmaStore ..
      | .clusterArrive .. | .clusterWait ..
      | .cpAsyncLoad .. | .tmaLoadAsync ..
      | .unary .. | .binary .. | .ternary .. | .eqMask ..
      | .scalarMul .. | .scalarAdd ..
      | .broadcast .. | .binaryBroadcast ..
      | .reduce .. | .reduceAccum ..
      | .cumsum .. | .cumprod ..
      | .outer ..
      | .swapLayout .. | .transpose .. | .convert .. | .convertVecLayout ..
      | .mask .. | .maskRightFillVal ..
      | .sliceRows .. | .sliceCols .. | .concatCols ..
      | .sync .. | .arrive .. | .arriveAndWait ..
      | .namedBarrierSync .. | .namedBarrierArrive ..
      | .warpGroupIdx .. | .warpGroupLaneId .. | .warpId .. | .laneId .. | .electOneSync ..
      | .warpgroupDecreaseRegisters .. | .warpgroupIncreaseRegisters ..
      | .fenceViewAsyncShared | .fenceProxyAsync
      | .semaphore .. | .semaphoreWarp .. | .semaphoreWaitVal ..
      | .semaphoreArray .. | .semaphoreArrayWarp ..
      | .semaphoreArrayWaitVal .. | .semaphoreArrayArrive ..
      | .comment ..
      | .getBlockIdx .. | .getThreadIdx .. | .getGridDim .. | .getBlockDim ..
      | .constInt .. | .constUInt64 .. | .constFloat ..
      | .scalarUnary .. | .scalarCompare .. | .scalarBinary .. | .scalarSelect ..
      | .scalarAssign .. | .scalarCast .. | .blockReduce ..
      | .vecIota .. | .vecFillScalar ..
      | .raw .. => false

private def bodyUses (p : KStmt → Bool) (body : Array KStmt) : Bool :=
  body.any (stmtUses p)

private def usesStoreAdd (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .storeAdd .. | .storeVecGlobalAdd .. => true
    | _ => false) k.body

private def usesLegacyTma (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .tmaLoad .. | .tmaStore .. => true
    | _ => false) k.body

private def usesSlice (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .sliceRows .. | .sliceCols .. | .concatCols .. => true
    | _ => false) k.body

private def usesOuter (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .outer .. => true
    | _ => false) k.body

private def usesEqMask (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .eqMask .. => true
    | _ => false) k.body

private def usesSharedLinear (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .declSharedLinear .. | .loadSharedLinear .. | .storeSharedLinear .. => true
    | _ => false) k.body

private def storeAddHelpers : String :=
  "template<typename Dst, typename Src>\n" ++
  "__device__ inline void store_add(Dst &dst, const Src &src) {\n" ++
  "  kittens::warp::store(dst, src);\n" ++
  "}\n" ++
  "template<typename Dst, typename Src, typename Offset>\n" ++
  "__device__ inline void store_add(Dst &dst, const Src &src, const Offset &offset) {\n" ++
  "  kittens::warp::store(dst, src, offset);\n" ++
  "}\n\n"

private def legacyTmaHelpers : String :=
  "template<typename ST, typename GL, typename CoordT>\n" ++
  "__device__ inline void tma_load(ST &dst, const GL &src, CoordT coord) {\n" ++
  "  kittens::coord<> idx(static_cast<int>(coord));\n" ++
  "  kittens::warp::load(dst, src, idx);\n" ++
  "}\n" ++
  "template<typename GL, typename ST, typename CoordT>\n" ++
  "__device__ inline void tma_store(GL &dst, const ST &src, CoordT coord) {\n" ++
  "  kittens::coord<> idx(static_cast<int>(coord));\n" ++
  "  kittens::warp::store(dst, src, idx);\n" ++
  "}\n\n"

private def sliceHelpers : String :=
  "template<int START_ROW, int NUM_ROWS, typename DST, typename SRC>\n" ++
  "__device__ inline void tk_slice_rows(DST &dst, const SRC &src) {\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<SRC>) {\n" ++
  "    using T = typename DST::T;\n" ++
  "    constexpr int tile_rows = kittens::TILE_ROW_DIM<T>;\n" ++
  "    static_assert(DST::rows == NUM_ROWS, \"slice rows: dst rows mismatch\");\n" ++
  "    static_assert(START_ROW % tile_rows == 0, \"slice rows: unaligned start\");\n" ++
  "    static_assert(NUM_ROWS % tile_rows == 0, \"slice rows: unaligned size\");\n" ++
  "    constexpr int start_tile = START_ROW / tile_rows;\n" ++
  "    constexpr int row_tiles = NUM_ROWS / tile_rows;\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < row_tiles; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < DST::width; j++) {\n" ++
  "        dst.tiles[i][j] = src.tiles[start_tile + i][j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<DST> && kittens::ducks::rt::all<SRC>) {\n" ++
  "    static_assert(SRC::rows == NUM_ROWS, \"slice rows: src rows mismatch\");\n" ++
  "    static_assert(START_ROW + NUM_ROWS <= DST::rows, \"slice rows: out of bounds\");\n" ++
  "    auto dst_sub = dst.template subtile<NUM_ROWS, DST::cols>(int2{START_ROW, 0});\n" ++
  "    kittens::warp::copy(dst_sub, src);\n" ++
  "  } else if constexpr (kittens::ducks::st::all<SRC>) {\n" ++
  "    static_assert(DST::rows == NUM_ROWS, \"slice rows: dst rows mismatch\");\n" ++
  "    static_assert(START_ROW + NUM_ROWS <= SRC::rows, \"slice rows: out of bounds\");\n" ++
  "    auto src_sub = src.template subtile<NUM_ROWS, SRC::cols>(int2{START_ROW, 0});\n" ++
  "    kittens::warp::copy(dst, src_sub);\n" ++
  "  }\n" ++
  "}\n\n" ++
  "template<int START_COL, int NUM_COLS, typename DST, typename SRC>\n" ++
  "__device__ inline void tk_slice_cols(DST &dst, const SRC &src) {\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<SRC>) {\n" ++
  "    using T = typename DST::T;\n" ++
  "    constexpr int tile_cols = kittens::TILE_COL_DIM<T>;\n" ++
  "    static_assert(DST::cols == NUM_COLS, \"slice cols: dst cols mismatch\");\n" ++
  "    static_assert(START_COL % tile_cols == 0, \"slice cols: unaligned start\");\n" ++
  "    static_assert(NUM_COLS % tile_cols == 0, \"slice cols: unaligned size\");\n" ++
  "    constexpr int start_tile = START_COL / tile_cols;\n" ++
  "    constexpr int col_tiles = NUM_COLS / tile_cols;\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < DST::height; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < col_tiles; j++) {\n" ++
  "        dst.tiles[i][j] = src.tiles[i][start_tile + j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<SRC>) {\n" ++
  "    static_assert(DST::cols == NUM_COLS, \"slice cols: dst cols mismatch\");\n" ++
  "    auto src_sub = src.template subtile<SRC::rows, NUM_COLS>(int2{0, START_COL});\n" ++
  "    kittens::warp::copy(dst, src_sub);\n" ++
  "  }\n" ++
  "}\n\n" ++
  "template<typename DST, typename LEFT, typename RIGHT>\n" ++
  "__device__ inline void tk_concat_cols(DST &dst, const LEFT &left, const RIGHT &right) {\n" ++
  "  static_assert(DST::rows == LEFT::rows, \"concat cols: left rows mismatch\");\n" ++
  "  static_assert(DST::rows == RIGHT::rows, \"concat cols: right rows mismatch\");\n" ++
  "  static_assert(DST::cols == LEFT::cols + RIGHT::cols, \"concat cols: dst cols mismatch\");\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<LEFT> && kittens::ducks::rt::all<RIGHT>) {\n" ++
  "    static_assert(DST::height == LEFT::height, \"concat cols: left height mismatch\");\n" ++
  "    static_assert(DST::height == RIGHT::height, \"concat cols: right height mismatch\");\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < DST::height; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < LEFT::width; j++) {\n" ++
  "        dst.tiles[i][j] = left.tiles[i][j];\n" ++
  "      }\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < RIGHT::width; j++) {\n" ++
  "        dst.tiles[i][LEFT::width + j] = right.tiles[i][j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<DST>) {\n" ++
  "    auto dst_left = dst.template subtile<DST::rows, LEFT::cols>(int2{0, 0});\n" ++
  "    auto dst_right = dst.template subtile<DST::rows, RIGHT::cols>(int2{0, LEFT::cols});\n" ++
  "    kittens::warp::copy(dst_left, left);\n" ++
  "    kittens::warp::copy(dst_right, right);\n" ++
  "  }\n" ++
  "}\n\n"

private def outerHelpers : String :=
  "template<typename RT, typename RVRow, typename RVCol>\n" ++
  "__device__ inline void tk_outer(RT &dst, const RVRow &row_vals, const RVCol &col_vals) {\n" ++
  "  RT row_tile;\n" ++
  "  RT col_tile;\n" ++
  "  kittens::warp::broadcast_row(row_tile, row_vals);\n" ++
  "  kittens::warp::broadcast_col(col_tile, col_vals);\n" ++
  "  kittens::warp::mul(dst, row_tile, col_tile);\n" ++
  "}\n\n"

private def eqMaskHelpers : String :=
  "template<ducks::rt::all T>\n" ++
  "__device__ inline void tk_eq_mask(T &dst, const T &lhs, const T &rhs) {\n" ++
  "  kittens::warp::copy(dst, lhs);\n" ++
  "  kittens::warp::sub(dst, dst, rhs);\n" ++
  "  kittens::warp::abs(dst, dst);\n" ++
  "  kittens::warp::apply(dst, dst, [] __device__ (int _row, int _col, auto _x) {\n" ++
  "    return (_x == static_cast<decltype(_x)>(0)) ? static_cast<decltype(_x)>(1) : static_cast<decltype(_x)>(0);\n" ++
  "  });\n" ++
  "}\n" ++
  "template<ducks::rv::all V>\n" ++
  "__device__ inline void tk_eq_mask(V &dst, const V &lhs, const V &rhs) {\n" ++
  "  kittens::warp::copy(dst, lhs);\n" ++
  "  kittens::warp::sub(dst, dst, rhs);\n" ++
  "  kittens::warp::abs(dst, dst);\n" ++
  "  kittens::warp::apply(dst, dst, [] __device__ (int _idx, auto _x) {\n" ++
  "    return (_x == static_cast<decltype(_x)>(0)) ? static_cast<decltype(_x)>(1) : static_cast<decltype(_x)>(0);\n" ++
  "  });\n" ++
  "}\n\n"

private def generateHelpers (k : Kernel) : String := Id.run do
  let mut helpers := ""
  if usesStoreAdd k then
    helpers := helpers ++ storeAddHelpers
  if usesLegacyTma k then
    helpers := helpers ++ legacyTmaHelpers
  if usesSlice k then
    helpers := helpers ++ sliceHelpers
  if usesOuter k then
    helpers := helpers ++ outerHelpers
  if usesEqMask k then
    helpers := helpers ++ eqMaskHelpers
  return helpers

private def generateHelpersForKernels (kernels : Array Kernel) : String := Id.run do
  let mut needStoreAdd := false
  let mut needLegacyTma := false
  let mut needSlice := false
  let mut needOuter := false
  let mut needEqMask := false
  for k in kernels do
    if usesStoreAdd k then
      needStoreAdd := true
    if usesLegacyTma k then
      needLegacyTma := true
    if usesSlice k then
      needSlice := true
    if usesOuter k then
      needOuter := true
    if usesEqMask k then
      needEqMask := true
  let mut helpers := ""
  if needStoreAdd then
    helpers := helpers ++ storeAddHelpers
  if needLegacyTma then
    helpers := helpers ++ legacyTmaHelpers
  if needSlice then
    helpers := helpers ++ sliceHelpers
  if needOuter then
    helpers := helpers ++ outerHelpers
  if needEqMask then
    helpers := helpers ++ eqMaskHelpers
  return helpers

/-- Per-kernel emission metadata that can be persisted safely. -/
structure KernelEmitInfo where
  /-- Kernel definition without helper templates. -/
  definition : String
  /-- Whether this kernel needs `store_add` helper templates. -/
  needsStoreAdd : Bool
  /-- Whether this kernel needs legacy TMA helper templates. -/
  needsLegacyTma : Bool
  /-- Whether this kernel needs slice helper templates. -/
  needsSlice : Bool
  /-- Whether this kernel needs outer-product helper templates. -/
  needsOuter : Bool
  /-- Whether this kernel needs equality-mask helper templates. -/
  needsEqMask : Bool
  deriving Repr, Inhabited

/-- Generate helper template block from precomputed helper flags. -/
def generateHelpersFromFlags (needStoreAdd needLegacyTma needSlice needOuter needEqMask : Bool) : String := Id.run do
  let mut helpers := ""
  if needStoreAdd then
    helpers := helpers ++ storeAddHelpers
  if needLegacyTma then
    helpers := helpers ++ legacyTmaHelpers
  if needSlice then
    helpers := helpers ++ sliceHelpers
  if needOuter then
    helpers := helpers ++ outerHelpers
  if needEqMask then
    helpers := helpers ++ eqMaskHelpers
  return helpers

private def generateKernelDefinition (k : Kernel) (emitSharedDecl : Bool := false) : String :=
  let rvState := inferRvLayouts k
  let rvVars := inferRvDecls k
  let tileInfo := inferTileInfo k
  let archGuard := s!"#if defined({k.family.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* empty parameter list */" else generateParams k
  let launchBounds := match k.launchBounds with
    | some (maxThreads, minBlocks) => s!" __launch_bounds__({maxThreads}, {minBlocks})"
    | none => ""
  let signature := s!"__global__{launchBounds} void {k.name}({paramStr}) \{\n"
  let useDynamicShared := emitSharedDecl
  let sharedDecl := if useDynamicShared
    then "  extern __shared__ int __shm[];\n  tma_swizzle_allocator al(__shm);\n"
    else ""
  let body := k.body.toList.map (generateStmt rvState.layouts rvVars tileInfo useDynamicShared "  ") |>.foldl (· ++ ·) ""
  let footer := "}\n#endif\n"
  layoutDiagnostics rvState.conflicts ++ archGuard ++ signature ++ sharedDecl ++ body ++ footer

/-- Generate emission metadata for a single kernel definition. -/
def generateKernelEmitInfo (k : Kernel) : KernelEmitInfo :=
  {
    definition := generateKernelDefinition k (k.sharedMemBytes > 0 || usesSharedLinear k)
    needsStoreAdd := usesStoreAdd k
    needsLegacyTma := usesLegacyTma k
    needsSlice := usesSlice k
    needsOuter := usesOuter k
    needsEqMask := usesEqMask k
  }

/-- Generate one or more kernel definitions for inclusion in a single `.cu` translation unit.
    Assumes CUDA/ThunderKittens headers are already included by the caller. -/
def generateKernelDefinitions (kernels : Array Kernel) : String :=
  generateHelpersForKernels kernels ++
  (kernels.toList.map (fun k => generateKernelDefinition k (k.sharedMemBytes > 0 || usesSharedLinear k)) |> String.intercalate "\n")

/-- Generate full kernel C++ code -/
def generateKernel (k : Kernel) : String :=
  let header := "#include <type_traits>\n#include <kittens.cuh>\nusing namespace kittens;\n\n"
  header ++ generateKernelDefinitions #[k]

/-- Generate kernel with extern shared memory declaration -/
def generateKernelWithShared (k : Kernel) : String :=
  let header :=
    "#include <type_traits>\n#include <kittens.cuh>\nusing namespace kittens;\n\n" ++
    generateHelpers k
  header ++ generateKernelDefinition k true

/-- Generate CUDA launch configuration -/
structure LaunchCfg where
  gridDim : Nat × Nat × Nat := (1, 1, 1)
  blockDim : Nat × Nat × Nat := (128, 1, 1)
  sharedMem : Nat := 0
  deriving Repr, Inhabited

/-- Generate CUDA launch code -/
def generateLaunch (k : Kernel) (cfg : LaunchCfg) (args : List String) : String :=
  let (gx, gy, gz) := cfg.gridDim
  let (bx, byy, bz) := cfg.blockDim
  let argStr := String.intercalate ", " args
  s!"{k.name}<<<dim3({gx}, {gy}, {gz}), dim3({bx}, {byy}, {bz}), {cfg.sharedMem}>>>({argStr});\n"

/-- Write kernel to file -/
def writeKernelFile (k : Kernel) (path : String) : IO Unit := do
  let code := generateKernel k
  IO.FS.writeFile path code

end Tyr.GPU.Codegen
