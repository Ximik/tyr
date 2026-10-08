import LeanTest
import Tyr.GPU.Codegen.TileIR

namespace Tests.GPUTileIR

open LeanTest
open Tyr.GPU.Codegen.TileIR

private def f32Tile64x64 : TileType := tileTy .f32 #[64, 64]
private def i8Tile16x16 : TileType := tileTy .i8 #[16, 16]
private def i32Tile16x16 : TileType := tileTy .i32 #[16, 16]
private def f32Scalar : TileType := scalarTileTy .f32
private def i1Scalar : TileType := scalarTileTy .i1
private def i32Scalar : TileType := scalarTileTy .i32
private def f32PtrTile : TileType := ptrTileTy .f32
private def i32PtrTile : TileType := ptrTileTy .i32

private def countSubstr (haystack needle : String) : Nat :=
  (haystack.splitOn needle).length - 1

private def mmaiFrontendDemo : Module :=
  tileir_module "mmai_frontend_demo" do
    tileir.entry "mmai_demo"
        (a : i8Tile16x16)
        (b : i8Tile16x16)
        (c : i32Tile16x16)
        (outPtr : i32PtrTile)
        (tok0 : .token) do
      let acc := mmai a, b, c
      let _ ← tileir.store_ptr_tko outPtr, acc, tok0
      pure ()

private def dceDemo : Module := {
  name := "dce_demo"
  entries := #[
    {
      name := "demo"
      params := #[
        { name := "out_ptr", ty := f32PtrTile },
        { name := "tok0", ty := .token }
      ]
      body := #[
        .const { name := "dead", ty := f32Scalar } (.float 2.0),
        .const { name := "live", ty := f32Scalar } (.float 1.0),
        .broadcast { name := "out", ty := f32Tile64x64 } "live" f32Scalar,
        .storePtrTko
          { name := "tok1", ty := .token }
          .weak
          "out_ptr"
          "out"
          "tok0"
          f32PtrTile
          f32Tile64x64
      ]
    }
  ]
}

private def hoistDemo : Module := {
  name := "hoist_demo"
  entries := #[
    {
      name := "demo"
      params := #[
        { name := "out_ptr", ty := f32PtrTile },
        { name := "tok0", ty := .token }
      ]
      body := #[
        .const { name := "zero", ty := i32Scalar } (.int 0),
        .const { name := "upper", ty := i32Scalar } (.int 4),
        .const { name := "step", ty := i32Scalar } (.int 1),
        .const { name := "seed", ty := f32Scalar } (.float 1.0),
        .forOp
          #[{ name := "acc", ty := f32Scalar }]
          { name := "iv", ty := i32Scalar }
          "zero"
          "upper"
          "step"
          #[
            { binder := { name := "running", ty := f32Scalar }, init := "seed" }
          ]
          #[
            .const { name := "inv", ty := f32Scalar } (.float 2.0),
            .unary { name := "next", ty := f32Scalar } .sqrt "inv",
            .continueOp #["next"]
          ],
        .broadcast { name := "out", ty := f32Tile64x64 } "acc" f32Scalar,
        .storePtrTko
          { name := "tok1", ty := .token }
          .weak
          "out_ptr"
          "out"
          "tok0"
          f32PtrTile
          f32Tile64x64
      ]
    }
  ]
}

private def cseDemo : Module := {
  name := "cse_demo"
  entries := #[
    {
      name := "demo"
      params := #[
        { name := "out_ptr", ty := f32PtrTile },
        { name := "tok0", ty := .token }
      ]
      body := #[
        .const { name := "value", ty := f32Scalar } (.float 3.0),
        .broadcast { name := "tile0", ty := f32Tile64x64 } "value" f32Scalar,
        .broadcast { name := "tile1", ty := f32Tile64x64 } "value" f32Scalar,
        .binary { name := "sum", ty := f32Tile64x64 } .addf "tile0" "tile1",
        .storePtrTko
          { name := "tok1", ty := .token }
          .weak
          "out_ptr"
          "sum"
          "tok0"
          f32PtrTile
          f32Tile64x64
      ]
    }
  ]
}

private def comparisonCseDemo : Module := {
  name := "comparison_cse_demo"
  entries := #[
    {
      name := "demo"
      params := #[
        { name := "lhsf", ty := f32Scalar },
        { name := "rhsf", ty := f32Scalar },
        { name := "lhsi", ty := i32Scalar },
        { name := "rhsi", ty := i32Scalar },
        { name := "out_ptr", ty := f32PtrTile },
        { name := "tok0", ty := .token }
      ]
      body := #[
        .cmpf { name := "predf0", ty := i1Scalar } .lessThan .ordered "lhsf" "rhsf" f32Scalar,
        .cmpf { name := "predf1", ty := i1Scalar } .lessThan .ordered "lhsf" "rhsf" f32Scalar,
        .cmpi { name := "predi0", ty := i1Scalar } .greaterThan "lhsi" "rhsi" .signed i32Scalar,
        .cmpi { name := "predi1", ty := i1Scalar } .greaterThan "lhsi" "rhsi" .signed i32Scalar,
        .select { name := "chosen", ty := f32Scalar } "predf1" "lhsf" "rhsf" i1Scalar f32Scalar,
        .assertOp "predi1" i1Scalar "integer comparison should remain live after CSE",
        .storePtrTko
          { name := "tok1", ty := .token }
          .weak
          "out_ptr"
          "chosen"
          "tok0"
          f32PtrTile
          f32Scalar
      ]
    }
  ]
}

private def runEntryAction {α} (action : EntryM α) (st : EntryState := {}) : Except String α :=
  let (result, _) := action.run st
  result

private def runLeanScriptExpectingError
    (fileName : String)
    (lines : List String)
    : IO IO.Process.Output := do
  let script : System.FilePath := ⟨s!"/tmp/{fileName}"⟩
  let scriptText := String.intercalate "\n" lines
  IO.FS.writeFile script scriptText
  IO.Process.output {
    cmd := "lake"
    args := #["env", "lean", toString script]
  }

@[test]
def testTypeRendering : IO Unit := do
  let tile : TileType := tileTy .f32 #[64, 128]
  let ptrTile : TileType := ptrTileTy .f32
  let tensorView : TileType := tensorViewTy .i32 #[.dynamic, .static 32] #[.dynamic, .static 1]
  let partitionView : TileType :=
    partitionViewTy
      #[2, 2]
      { elem := .f32, shape := #[.dynamic, .static 32], strides := #[.dynamic, .static 1] }
      #[0, 1]
  assertEqual (renderType tile) "!cuda_tile.tile<64x128xf32>"
    "TileIR tile types should render in public MLIR syntax"
  assertEqual (renderType ptrTile) "!cuda_tile.tile<ptr<f32>>"
    "TileIR pointer-tile types should render in public MLIR syntax"
  assertEqual (renderType tensorView) "!cuda_tile.tensor_view<?x32xi32, strides=[?, 1]>"
    "Tensor-view types should preserve dynamic dimensions and strides"
  assertEqual
    (renderType partitionView)
    "!cuda_tile.partition_view<tile=(2x2), tensor_view<?x32xf32, strides=[?, 1]>, dim_map=[0, 1]>"
    "Partition-view types should render in public MLIR syntax"

@[test]
def testTileAlgebraModuleRendering : IO Unit := do
  let text := renderModule tileAlgebraDemo
  assertTrue (text.containsSubstr "cuda_tile.module @tileAlgebraDemo")
    "Annotated TileIR modules should render the top-level module op"
  assertTrue (text.containsSubstr "cuda_tile.entry @tileAlgebraDemo")
    "Annotated TileIR modules should render entry ops"
  assertTrue (text.containsSubstr "cuda_tile.load_ptr_tko weak")
    "Tile algebra demo should exercise token-ordered pointer loads"
  assertTrue (text.containsSubstr "cuda_tile.constant <f32: 1.250000>")
    "Tile algebra demo should render scalar constants for broadcasted weights"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "Tile algebra demo should render tile broadcast"
  assertTrue (text.containsSubstr "cuda_tile.addf ")
    "Tile algebra demo should render binary tile algebra"
  assertTrue (text.containsSubstr "cuda_tile.exp2 ")
    "Tile algebra demo should render unary tile algebra"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "Tile algebra demo should render token-ordered stores"

@[test]
def testControlFlowRendering : IO Unit := do
  let text := renderModule controlFlowDemo
  assertTrue (text.containsSubstr "cuda_tile.make_tensor_view %base")
    "Control-flow demo should render tensor-view construction"
  assertTrue (text.containsSubstr "cuda_tile.make_partition_view ")
    "Control-flow demo should render partition-view construction"
  assertTrue (text.containsSubstr "cuda_tile.offset %ptrs, %idxs")
    "Control-flow demo should render pointer offset operations"
  assertTrue (text.containsSubstr "cuda_tile.if %pred")
    "Control-flow demo should render structured if"
  assertTrue (text.containsSubstr "cuda_tile.for ")
    "Control-flow demo should render structured for"
  assertTrue (text.containsSubstr "cuda_tile.continue ")
    "Control-flow demo should render loop continue"
  assertTrue (text.containsSubstr "cuda_tile.print_tko \"partition view prepared\"")
    "Control-flow demo should render token-ordered print"

@[test]
def testMmaLoopRendering : IO Unit := do
  let text := renderModule mmaLoopDemo
  assertTrue (text.containsSubstr "cuda_tile.mmaf ")
    "MMA loop demo should render floating-point MMA operations"
  assertTrue (text.containsSubstr "cuda_tile.maxf ")
    "MMA loop demo should render named max tile algebra"
  assertTrue (text.containsSubstr "cuda_tile.minf ")
    "MMA loop demo should render named min tile algebra"
  assertTrue (text.containsSubstr "cuda_tile.for ")
    "MMA loop demo should render structured frontend loops"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "MMA loop demo should keep its final effectful store after optimization"

@[test]
def testCutileStyleSurfaceTileAlgebraRendering : IO Unit := do
  let text := renderModule surfaceTileAlgebraDemo
  assertTrue (text.containsSubstr "cuda_tile.entry @surfaceTileAlgebraDemo")
    "Attribute-style cutile surface kernels should synthesize a single TileIR entry"
  assertTrue (text.containsSubstr "cuda_tile.load_ptr_tko weak")
    "The cutile-style surface should lower ct.load to token-ordered loads"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "The cutile-style surface should lower ct.broadcast to TileIR broadcast"
  assertTrue (text.containsSubstr "cuda_tile.exp2 ")
    "The cutile-style surface should support namespaced unary tile algebra"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "The cutile-style surface should synthesize the hidden token chain into a final store"

@[test]
def testCutileStyleSurfaceControlFlowRendering : IO Unit := do
  let text := renderModule surfaceControlFlowDemo
  assertTrue (text.containsSubstr "cuda_tile.make_tensor_view %base")
    "The cutile-style surface should lower ct.tensor_view to TileIR tensor views"
  assertTrue (text.containsSubstr "cuda_tile.make_partition_view ")
    "The cutile-style surface should lower ct.partition_view to TileIR partition views"
  assertTrue (text.containsSubstr "cuda_tile.offset %ptrs, %idxs")
    "The cutile-style surface should lower ct.offset to TileIR pointer offsets"
  assertTrue (text.containsSubstr "cuda_tile.print_tko \"surface partition view prepared\"")
    "The cutile-style surface should lower ct.print with implicit token sequencing"
  assertTrue (text.containsSubstr "cuda_tile.if %pred")
    "The cutile-style surface should preserve structured if lowering"

@[test]
def testCutileStyleSurfaceMmaRendering : IO Unit := do
  let text := renderModule surfaceMmaDemo
  assertTrue (text.containsSubstr "cuda_tile.mmaf ")
    "The cutile-style surface should lower ct.mma to TileIR MMA"
  assertTrue (text.containsSubstr "cuda_tile.maxf ")
    "The cutile-style surface should lower ct.max to TileIR maxf"
  assertTrue (text.containsSubstr "cuda_tile.minf ")
    "The cutile-style surface should lower ct.min to TileIR minf"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "The cutile-style surface should synthesize stores from ct.store"

@[test]
def testCutileStyleSurfaceBlockIdRendering : IO Unit := do
  let text := renderModule surfaceBlockIdDemo
  assertTrue (text.containsSubstr "cuda_tile.get_tile_block_id")
    "The cutile-style surface should lower ct.bid to TileIR block-id queries"
  assertTrue (text.containsSubstr "cuda_tile.get_num_tile_blocks")
    "The cutile-style surface should lower ct.num_blocks to TileIR grid-dimension queries"
  assertTrue (text.containsSubstr "cuda_tile.assert ")
    "The cutile-style surface should lower ct.assert to TileIR runtime asserts"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "The cutile-style surface should keep effectful stores after block-id lowering"

@[test]
def testCutileStyleSurfaceIotaRendering : IO Unit := do
  let text := renderModule surfaceIotaDemo
  assertTrue (text.containsSubstr "cuda_tile.iota")
    "The cutile-style surface should lower ct.iota to TileIR iota"
  assertTrue (text.containsSubstr "cuda_tile.assert ")
    "The cutile-style surface should permit runtime asserts alongside iota"

@[test]
def testCutileStyleSurfaceRangeRendering : IO Unit := do
  let text := renderModule surfaceRangeDemo
  assertTrue (text.containsSubstr "cuda_tile.for ")
    "The cutile-style surface should lower `for ... in ct.range(...) do` to TileIR structured loops"
  assertTrue (text.containsSubstr "cuda_tile.print_tko \"surface range iteration\"")
    "The cutile-style surface should allow effectful loop bodies with implicit token threading"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "The cutile-style surface should preserve post-loop effectful stores"

@[test]
def testCutileStyleStaticRangeRendering : IO Unit := do
  let text := renderModule surfaceStaticRangeDemo8
  assertFalse (text.containsSubstr "cuda_tile.for ")
    "ct.static_range should unroll through ordinary Lean `for`, not lower a runtime TileIR loop"
  assertEqual (countSubstr text "cuda_tile.print_tko \"surface static range iteration\"") 4
    "ct.static_range(ct.cdiv(...)) should specialize into the expected number of unrolled effectful iterations"
  assertTrue (text.containsSubstr "!cuda_tile.tile<8xf32>")
    "ct.cdiv/static_range demos should still preserve ct.Const-derived tile shapes after specialization"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "ct.static_range demos should still compose with implicit-token stores"

@[test]
def testCutileStyleStaticIterListRendering : IO Unit := do
  let text := renderModule surfaceStaticIterListDemo
  assertFalse (text.containsSubstr "cuda_tile.for ")
    "ct.static_iter over a Lean list should reuse the ordinary `ForIn` elaborator instead of lowering a TileIR loop"
  assertEqual (countSubstr text "cuda_tile.print_tko \"surface static list iteration\"") 3
    "ct.static_iter should preserve the source collection length during compile-time unrolling"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "ct.static_iter demos should still compose with effectful stores"

@[test]
def testCutileStyleStaticHelpers : IO Unit := do
  assertEqual (ct.cdiv(17, 8)) 3
    "ct.cdiv should round integer divisions up at compile time"
  assertEqual (ct.static_range(1, 8, 2)) [1, 3, 5, 7]
    "ct.static_range(start, stop, step) should expose Lean-native compile-time iteration values"
  assertEqual (ct.static_iter(([0, 1, 2] : List Nat))) [0, 1, 2]
    "ct.static_iter should preserve the underlying compile-time collection"

@[test]
def testCutileStyleSurfaceViewRendering : IO Unit := do
  let text := renderModule surfaceViewDemo
  assertTrue (text.containsSubstr "cuda_tile.load_view_tko")
    "The cutile-style surface should lower ct.load_view to TileIR view loads"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "The cutile-style surface should lower ct.store_view to TileIR view stores"
  assertTrue (text.containsSubstr "cuda_tile.get_tile_block_id")
    "The cutile-style surface should permit block-id queries as view indices"
  assertTrue (text.containsSubstr "cuda_tile.assert ")
    "The cutile-style surface should allow runtime asserts in view-based kernels"

@[test]
def testCutileStyleIndexedLoadStore1DRendering : IO Unit := do
  let text := renderModule (surfaceIndexedVecAdd1D 8)
  assertTrue (text.containsSubstr "cuda_tile.entry @surfaceIndexedVecAdd1D")
    "Indexed cutile-style kernels should still synthesize a single TileIR entry after specialization"
  assertTrue (text.containsSubstr "cuda_tile.make_tensor_view %a")
    "Indexed ct.load should synthesize tensor views from raw arrays"
  assertTrue (text.containsSubstr "cuda_tile.make_partition_view ")
    "Indexed ct.load should synthesize partition views for tile coordinates"
  assertTrue (text.containsSubstr "cuda_tile.load_view_tko")
    "Indexed ct.load should lower through TileIR view loads"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Indexed ct.store should lower through TileIR view stores"
  assertTrue (text.containsSubstr "!cuda_tile.tile<8xf32>")
    "Indexed ct.load should preserve ct.Const tile shapes during specialization"

@[test]
def testCutileStyleIndexedLoadStore2DRendering : IO Unit := do
  let text := renderModule (surfaceIndexedVecAdd2D 4 8)
  assertTrue (text.containsSubstr "cuda_tile.get_tile_block_id")
    "Indexed 2D ct.load should compose with multidimensional block ids"
  assertTrue (text.containsSubstr "cuda_tile.partition_view<tile=(4x8)")
    "Indexed 2D ct.load should synthesize rank-2 partition views from the tile shape"
  assertTrue (text.containsSubstr "cuda_tile.load_view_tko")
    "Indexed 2D ct.load should lower through TileIR view loads"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Indexed 2D ct.store should lower through TileIR view stores"

@[test]
def testCutileStyleArangeRendering : IO Unit := do
  let text := renderModule (surfaceArangeDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.iota")
    "ct.arange should lower to TileIR iota"
  assertTrue (text.containsSubstr "!cuda_tile.tile<8xi32>")
    "ct.arange should preserve compile-time lane counts in its result type"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "ct.arange results should participate in the usual implicit-token store sequencing"

@[test]
def testCutileStyleGatherScatterRendering : IO Unit := do
  let text := renderModule (surfaceGatherScatterDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.iota")
    "ct.gather/ct.scatter demos should compose naturally with ct.arange-generated offsets"
  assertTrue (text.containsSubstr "cuda_tile.offset %input")
    "ct.gather should synthesize pointer offsets from flat indices"
  assertTrue (text.containsSubstr "cuda_tile.load_ptr_tko weak")
    "ct.gather should lower through token-ordered pointer loads"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "ct.gather/ct.scatter demos should compose with tile broadcasts in the cutile-style surface"
  assertTrue (text.containsSubstr "cuda_tile.offset %output")
    "ct.scatter should synthesize pointer offsets for the destination array"
  assertTrue (text.containsSubstr "cuda_tile.store_ptr_tko weak")
    "ct.scatter should lower through token-ordered pointer stores"

@[test]
def testCutileStyleFullZerosOnesRendering : IO Unit := do
  let text := renderModule (surfaceFillCastDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.offset %input")
    "Scalar indexed ct.load with shape=() should synthesize a pointer offset"
  assertTrue (text.containsSubstr "cuda_tile.load_ptr_tko weak")
    "Scalar indexed ct.load with shape=() should lower through token-ordered pointer loads"
  assertTrue (text.containsSubstr "cuda_tile.itof ")
    "ct.full should cast scalar integer inputs to the requested floating-point dtype when needed"
  assertTrue (text.containsSubstr "cuda_tile.constant <f32: 0.000000>")
    "ct.zeros should materialize a scalar zero constant of the requested dtype"
  assertTrue (text.containsSubstr "cuda_tile.constant <f32: 1.000000>")
    "ct.ones should materialize a scalar one constant of the requested dtype"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "ct.full/ct.zeros/ct.ones should broadcast scalar seeds to the requested tile shape"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "ct.store should keep using indexed view stores after constructor lowering"

@[test]
def testCutileStyleReshapeRendering : IO Unit := do
  let text := renderModule (surfaceReshapeDemo 2 4)
  assertTrue (text.containsSubstr "cuda_tile.reshape ")
    "ct.reshape(shape) should derive and lower a reshape op from the source tile type"
  assertTrue (text.containsSubstr "!cuda_tile.tile<2x4xf32>")
    "ct.reshape demos should preserve the source tile shape before reshaping"
  assertTrue (text.containsSubstr "!cuda_tile.tile<8xf32>")
    "ct.reshape demos should compute the flattened target tile shape during specialization"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Reshaped tiles should still compose with indexed ct.store lowering"

@[test]
def testCutileStyleShapeMetadataRendering : IO Unit := do
  let text := renderModule (surfaceShapeMetadataDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.make_partition_view ")
    "Positional ct.load/ct.store syntax should still lower through partition views"
  assertTrue (text.containsSubstr "cuda_tile.itof ")
    "Shape-metadata demos should cast integer tiles to the output dtype"
  assertTrue (text.containsSubstr "cuda_tile.reshape ")
    "Shape-metadata demos should exercise reshape lowering"
  assertTrue (text.containsSubstr "!cuda_tile.tile<1x8xf32>")
    "Shape-metadata demos should preserve reshaped matrix types in the rendered TileIR"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Positional ct.store syntax should lower through indexed view stores"

@[test]
def testCutileStyleShapeMetadataHelper : IO Unit := do
  let value := arg "tile" (tileTy .i32 #[2, 4])
  assertEqual value.shape #[2, 4]
    "Value.shape should expose static TileIR tile dimensions as Lean metadata"
  assertEqual value.rank 2
    "Value.rank should expose the number of static TileIR tile dimensions"

@[test]
def testCutileStyleMethodSyntaxRendering : IO Unit := do
  let text := renderModule (surfaceMethodSyntaxDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.make_partition_view ")
    "Method-syntax demos should still lower positional ct.load into partition views"
  assertTrue (text.containsSubstr "cuda_tile.itof ")
    "x.astype(dtype) should lower to TileIR integer-to-float conversion"
  assertTrue (text.containsSubstr "cuda_tile.reshape ")
    "x.reshape(...) should lower to TileIR reshape"
  assertTrue (text.containsSubstr "!cuda_tile.tile<1x8xf32>")
    "Method-syntax demos should preserve the reshaped output tile type"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Method-syntax demos should keep indexed stores after lowering"
  assertFalse (text.containsSubstr "cuda_tile.assert ")
    "Method-syntax metadata checks should specialize away instead of lowering runtime asserts"

@[test]
def testCutileStyleTransformRendering : IO Unit := do
  let text := renderModule (surfaceTransformDemo 4 4)
  assertTrue (text.containsSubstr "cuda_tile.extract ")
    "ct.extract should lower to TileIR extract ops"
  assertTrue (text.containsSubstr "cuda_tile.cat ")
    "ct.cat should lower to TileIR cat ops"
  assertTrue (text.containsSubstr "cuda_tile.permute ")
    "Value.permute should lower to TileIR permute ops"
  assertTrue (text.containsSubstr "cuda_tile.select ")
    "ct.where should lower to TileIR select ops"
  assertTrue (text.containsSubstr "!cuda_tile.tile<4x4xf32>")
    "Transform demos should preserve the transformed tile type in rendered TileIR"

@[test]
def testCutileStyleTransposeBroadcastRendering : IO Unit := do
  let text := renderModule (surfaceTransposeBroadcastDemo 2 4)
  assertTrue (text.containsSubstr "cuda_tile.permute ")
    "ct.transpose should lower through TileIR permute"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "ct.broadcast_to should lower through TileIR broadcast"
  assertTrue (text.containsSubstr "!cuda_tile.tile<4x2xf32>")
    "Transpose/broadcast demos should preserve the swapped static tile type"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Transpose/broadcast demos should still lower indexed stores through view stores"

@[test]
def testCutileStyleComparisonSelectRendering : IO Unit := do
  let text := renderModule (surfaceCompareSelectDemo 2 4)
  assertTrue (text.containsSubstr "cuda_tile.cmpf greater_than ordered")
    "ct.greater should lower to TileIR floating-point comparisons"
  assertTrue (text.containsSubstr "!cuda_tile.tile<2x4xi1>")
    "Typed comparisons should produce i1 mask tiles with the source shape"
  assertTrue (text.containsSubstr "cuda_tile.select ")
    "ct.where should lower to TileIR select ops after comparison lowering"
  assertTrue (text.containsSubstr "cuda_tile.broadcast ")
    "Scalar compare/select operands should broadcast to the destination tile shape"
  assertTrue (text.containsSubstr "cuda_tile.store_view_tko")
    "Comparison/select demos should still lower indexed stores through view stores"

@[test]
def testCutileStyleMaximumMinimumRendering : IO Unit := do
  let text := renderModule (surfaceMaximumMinimumDemo 2 4)
  assertTrue (text.containsSubstr "cuda_tile.maxf ")
    "ct.maximum should lower to TileIR maxf"
  assertTrue (text.containsSubstr "cuda_tile.minf ")
    "ct.minimum should lower to TileIR minf"

@[test]
def testCutileStyleMatmulRendering : IO Unit := do
  let text := renderModule surfaceMatmulDemo16
  assertTrue (text.containsSubstr "cuda_tile.mmaf ")
    "ct.matmul and Value.matmul should lower to TileIR mmaf"
  assertTrue (text.containsSubstr "!cuda_tile.tile<16x16xf32>")
    "Matmul demos should preserve the accumulator/output tile type"
  assertFalse (text.containsSubstr "cuda_tile.assert ")
    "Static shape metadata checks around matmul should specialize away"

@[test]
def testCutileStyleConstShapeRendering : IO Unit := do
  let text := renderModule (surfaceConstShapeDemo 8)
  assertTrue (text.containsSubstr "cuda_tile.entry @surfaceConstShapeDemo")
    "Parameterized cutile-style kernels should still synthesize a single TileIR entry after specialization"
  assertTrue (text.containsSubstr "!cuda_tile.tile<8xf32>")
    "ct.Const parameters should be available in runtime binder types during TileIR lowering"
  assertFalse (text.containsSubstr "cuda_tile.assert ")
    "ct.static_assert should specialize away instead of lowering to a runtime TileIR assert"

@[test]
def testCutileStyleStaticAssertFailure : IO Unit := do
  let script : System.FilePath := ⟨"/tmp/tileir_static_assert_failure.lean"⟩
  let scriptText := String.intercalate "\n" [
    "import Tyr.GPU.Codegen.TileIR.Examples",
    "#eval Tyr.GPU.Codegen.TileIR.renderModule (Tyr.GPU.Codegen.TileIR.surfaceConstShapeDemo 4)"
  ]
  IO.FS.writeFile script scriptText
  let result ← IO.Process.output {
    cmd := "lake"
    args := #["env", "lean", toString script]
  }
  let output := result.stdout ++ result.stderr
  assertTrue (output.containsSubstr "TileIR static assertion failed")
    "ct.static_assert should reject invalid ct.Const specializations during frontend evaluation"

@[test]
def testCutileStyleStaticEvalRendering : IO Unit := do
  let sqrtText := renderModule (surfaceStaticEvalDemo true)
  let expText := renderModule (surfaceStaticEvalDemo false)
  assertTrue (sqrtText.containsSubstr "cuda_tile.sqrt ")
    "ct.static_eval should preserve the chosen compile-time branch when specializing a kernel"
  assertFalse (sqrtText.containsSubstr "cuda_tile.exp2 ")
    "ct.static_eval should eliminate untaken compile-time branches from specialized TileIR"
  assertTrue (expText.containsSubstr "cuda_tile.exp2 ")
    "ct.static_eval should support alternate compile-time branches across specializations"
  assertFalse (expText.containsSubstr "cuda_tile.sqrt ")
    "ct.static_eval should not leave dead specialized branches in emitted TileIR"

@[test]
def testIntegerMmaFrontendRendering : IO Unit := do
  let text := renderModule mmaiFrontendDemo
  assertTrue (text.containsSubstr "cuda_tile.mmai ")
    "The frontend should lower integer MMA surface syntax to TileIR integer MMA ops"

@[test]
def testBuilderRejectsInvalidReshape : IO Unit := do
  let src : Value := { name := "src", ty := tileTy .f32 #[2, 2] }
  let dstTy : TileType := tileTy .f32 #[3]
  match runEntryAction (Tyr.GPU.Codegen.TileIR.reshape "bad" src dstTy) with
  | .ok _ =>
      fail "TileIR reshape should reject element-count mismatches before rendering"
  | .error err =>
      assertTrue (err.containsSubstr "same element count")
        "TileIR reshape failures should explain the static shape mismatch"

@[test]
def testBuilderRejectsMismatchedBinaryTypes : IO Unit := do
  let lhs : Value := { name := "lhs", ty := tileTy .f32 #[8] }
  let rhs : Value := { name := "rhs", ty := tileTy .i32 #[8] }
  match runEntryAction (binary "bad" .addf lhs rhs) with
  | .ok _ =>
      fail "TileIR binary ops should reject mismatched operand element types"
  | .error err =>
      assertTrue (err.containsSubstr "matching operand types")
        "TileIR binary failures should explain the operand type mismatch"

@[test]
def testBuilderRejectsMismatchedCatTypes : IO Unit := do
  let lhs : Value := { name := "lhs", ty := tileTy .f32 #[4, 4] }
  let rhs : Value := { name := "rhs", ty := tileTy .i32 #[4, 4] }
  match runEntryAction (cat "bad" lhs rhs 0) with
  | .ok _ =>
      fail "TileIR cat should reject mismatched element types before rendering"
  | .error err =>
      assertTrue (err.containsSubstr "matching element types")
        "TileIR cat failures should explain the element type mismatch"

@[test]
def testBuilderRejectsInvalidPermute : IO Unit := do
  let src : Value := { name := "src", ty := tileTy .f32 #[2, 4, 8] }
  match runEntryAction (Tyr.GPU.Codegen.TileIR.permute "bad" src #[0, 0, 2]) with
  | .ok _ =>
      fail "TileIR permute should reject duplicate permutation axes before rendering"
  | .error err =>
      assertTrue (err.containsSubstr "without duplicates")
        "TileIR permute failures should explain invalid permutations"

@[test]
def testToolchainCommandConstruction : IO Unit := do
  let toolchain : Toolchain := {
    cudaTileOpt? := some ⟨"/opt/cuda/bin/cuda-tile-opt"⟩
    cudaTileTranslate? := some ⟨"/opt/cuda/bin/cuda-tile-translate"⟩
    tileiras? := some ⟨"/opt/cuda/bin/tileiras"⟩
    searchDirs := #[⟨"/opt/cuda/bin"⟩]
  }
  let paths := artifactPaths ⟨"/tmp/tileir-tests"⟩ tileAlgebraDemo
  let .ok optCmd := buildOptCommand toolchain paths.inputMlir paths.optimizedMlir
    | fail "Expected opt command construction to succeed"
  let .ok translateCmd := buildTranslateCommand toolchain paths.optimizedMlir paths.bytecode
    | fail "Expected translate command construction to succeed"
  let .ok tileirasCmd := buildTileirasCommand toolchain paths.bytecode paths.cubin
      { gpuName := "sm_100", optLevel := 2, lineInfo := true }
    | fail "Expected tileiras command construction to succeed"
  assertEqual (renderShellCommand optCmd)
    s!"/opt/cuda/bin/cuda-tile-opt -no-implicit-module {paths.inputMlir} -o {paths.optimizedMlir}"
    "TileIR tool driver should build the expected cuda-tile-opt invocation"
  assertEqual (renderShellCommand translateCmd)
    s!"/opt/cuda/bin/cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module {paths.optimizedMlir} -o {paths.bytecode}"
    "TileIR tool driver should build the expected cuda-tile-translate invocation"
  assertEqual (renderShellCommand tileirasCmd)
    s!"/opt/cuda/bin/tileiras {paths.bytecode} -o {paths.cubin} --gpu-name sm_100 -O2 --lineinfo"
    "TileIR tool driver should build the expected tileiras invocation"

@[test]
def testMissingToolError : IO Unit := do
  let toolchain : Toolchain := { searchDirs := #[⟨"/does/not/exist"⟩] }
  match buildTranslateCommand toolchain ⟨"in.mlir"⟩ ⟨"out.tilebc"⟩ with
  | .ok _ =>
      fail "Expected buildTranslateCommand to fail when the translator is unavailable"
  | .error err =>
      assertTrue ((toString err).containsSubstr "cuda-tile-translate")
        "Missing-tool diagnostics should name the missing TileIR executable"

@[test]
def testDeadCodeEliminationPass : IO Unit := do
  let optimized := optimizeModule dceDemo
  let body := optimized.entries[0]!.body
  assertEqual body.size 3
    "Optimization should remove unused pure TileIR statements"
  match body[0]!, body[1]!, body[2]! with
  | .const dst _, .broadcast out _ _, .storePtrTko _ _ _ value _ _ _ =>
      assertEqual dst.name "live" "The live constant should remain after DCE"
      assertEqual out.name "out" "The live broadcast should remain after DCE"
      assertEqual value "out" "The effectful store should still consume the broadcast result"
  | _, _, _ =>
      fail "Dead code elimination should keep only the live constant, broadcast, and store"

@[test]
def testLoopInvariantHoistingPass : IO Unit := do
  let optimized := optimizeModule hoistDemo
  let body := optimized.entries[0]!.body
  assertEqual body.size 9
    "Optimization should hoist loop-invariant pure statements out of the loop body"
  match body[4]!, body[5]!, body[6]! with
  | .const inv _, .unary next .sqrt src, .forOp _ iv _ _ _ _ loopBody =>
      assertEqual inv.name "inv" "The invariant constant should be hoisted before the loop"
      assertEqual next.name "next" "The invariant unary op should also be hoisted"
      assertEqual src "inv" "The hoisted unary op should still consume the invariant constant"
      assertEqual iv.name "iv" "The loop induction variable should remain intact"
      assertEqual loopBody.size 1 "Only the loop terminator should remain inside the hoisted loop body"
      match loopBody[0]! with
      | .continueOp values =>
          assertEqual values #[ "next" ]
            "The loop should continue with the hoisted invariant value"
      | _ =>
          fail "The optimized loop body should end in a continue terminator"
  | _, _, _ =>
      fail "Loop-invariant statements should be hoisted immediately before the loop"

@[test]
def testCommonSubexpressionEliminationPass : IO Unit := do
  let optimized := optimizeModule cseDemo
  let body := optimized.entries[0]!.body
  assertEqual body.size 4
    "Optimization should remove duplicate pure tile algebra statements"
  match body[1]!, body[2]! with
  | .broadcast tile _ _, .binary sum .addf lhs rhs =>
      assertEqual tile.name "tile0" "The first broadcast should become the canonical CSE result"
      assertEqual sum.name "sum" "The binary consumer should remain after CSE"
      assertEqual lhs "tile0" "The binary consumer should be rewritten to use the canonical broadcast"
      assertEqual rhs "tile0" "Both duplicate operands should resolve to the canonical broadcast"
  | _, _ =>
      fail "Common subexpression elimination should keep one broadcast and rewrite its users"

@[test]
def testComparisonCsePass : IO Unit := do
  let optimized := commonSubexpressionElimination comparisonCseDemo
  let body := optimized.entries[0]!.body
  assertEqual body.size 5
    "CSE should eliminate duplicate comparison ops and rewrite their users"
  match body[0]!, body[1]!, body[2]!, body[3]! with
  | .cmpf floatPred _ _ _ _ _,
    .cmpi intPred _ _ _ _ _,
    .select chosen cond _ _ _ _,
    .assertOp assertedCond _ _ =>
      assertEqual floatPred.name "predf0"
        "The first floating-point comparison should become the canonical CSE result"
      assertEqual intPred.name "predi0"
        "The first integer comparison should become the canonical CSE result"
      assertEqual chosen.name "chosen"
        "CSE should preserve downstream comparison consumers"
      assertEqual cond "predf0"
        "Floating-point comparison consumers should be rewritten to the canonical result"
      assertEqual assertedCond "predi0"
        "Integer comparison consumers should be rewritten to the canonical result"
  | _, _, _, _ =>
      fail "Comparison CSE should keep one cmpf/cmpi each and rewrite their users"

@[test]
def testBuiltinPassTrace : IO Unit := do
  let trace := optimizeModuleWithTrace cseDemo
  assertEqual (trace.map (·.name)) #[`dce, `cse, `hoistLoopInvariants, `dce]
    "The builtin TileIR pass pipeline should follow the expected cutile-style order"
  assertEqual (trace.map (·.occurrence)) #[0, 0, 0, 1]
    "Repeated passes should retain their occurrence index in the trace"

/--
Type-safe tile algebra mismatches should fail elaboration in an isolated script
without poisoning the runtime test module with `#guard_msgs`-generated `sorry`s.
-/
@[test]
def testBadTypedAddKernelElabFailure : IO Unit := do
  let result ← runLeanScriptExpectingError "tileir_bad_typed_add.lean" [
    "import Tyr.GPU.Codegen.TileIR",
    "open Tyr.GPU.Codegen.TileIR",
    "",
    "@[tileir_kernel]",
    "def badTypedAddKernel",
    "    (lhs : ct.Array ct.f32)",
    "    (rhs : ct.Array ct.i32)",
    "    (out : ct.Array ct.f32) := do",
    "  let bid := ct.bid 0",
    "  let lhsTile ← ct.load lhs, index := (bid,), shape := (8,)",
    "  let rhsTile ← ct.load rhs, index := (bid,), shape := (8,)",
    "  ct.store out, index := (bid,), tile := lhsTile + rhsTile"
  ]
  let output := result.stdout ++ result.stderr
  assertTrue (result.exitCode != 0)
    "Mismatched tile addition should fail at elaboration time"
  assertTrue (output.containsSubstr "failed to synthesize")
    "Mismatched tile addition should fail at elaboration time"
  assertTrue (output.containsSubstr "HAdd")
    "Mismatched tile addition failures should point at the overloaded tile algebra surface"

@[test]
def testBadIntegerAddKernelElabFailure : IO Unit := do
  let result ← runLeanScriptExpectingError "tileir_bad_integer_add.lean" [
    "import Tyr.GPU.Codegen.TileIR",
    "open Tyr.GPU.Codegen.TileIR",
    "",
    "@[tileir_kernel]",
    "def badIntegerAddKernel",
    "    (lhs : ct.Array ct.i32)",
    "    (rhs : ct.Array ct.i32)",
    "    (out : ct.Array ct.i32) := do",
    "  let bid := ct.bid 0",
    "  let lhsTile ← ct.load lhs, index := (bid,), shape := (8,)",
    "  let rhsTile ← ct.load rhs, index := (bid,), shape := (8,)",
    "  ct.store out, index := (bid,), tile := lhsTile + rhsTile"
  ]
  let output := result.stdout ++ result.stderr
  assertTrue (result.exitCode != 0)
    "Integer tile addition should fail during elaboration"
  assertTrue (output.containsSubstr "failed to synthesize")
    "Integer tile addition should fail before lowering floating TileIR algebra ops"
  assertTrue (output.containsSubstr "HAdd" || output.containsSubstr "FloatValueTy")
    "Integer tile addition failures should expose the floating-only typed surface restriction"

/--
`ct.where` shape mismatches should also fail in isolation so the runtime test
binary can import this module without triggering `lean_sorry`.
-/
@[test]
def testBadTypedWhereKernelElabFailure : IO Unit := do
  let result ← runLeanScriptExpectingError "tileir_bad_typed_where.lean" [
    "import Tyr.GPU.Codegen.TileIR",
    "open Tyr.GPU.Codegen.TileIR",
    "",
    "@[tileir_kernel]",
    "def badTypedWhereKernel",
    "    (conds : ct.Array ct.bool_)",
    "    (lhs : ct.Array ct.f32)",
    "    (rhs : ct.Array ct.f32)",
    "    (out : ct.Array ct.f32) := do",
    "  let bid := ct.bid 0",
    "  let condTile ← ct.load conds, index := (bid,), shape := (4,)",
    "  let lhsTile ← ct.load lhs, index := (bid,), shape := (4,)",
    "  let rhsTile ← ct.load rhs, index := (bid,), shape := (8,)",
    "  let merged := ct.where(condTile, lhsTile, rhsTile)",
    "  ct.store out, index := (bid,), tile := merged"
  ]
  let output := result.stdout ++ result.stderr
  assertTrue (result.exitCode != 0)
    "Mismatched ct.where kernels should fail during elaboration"
  assertTrue
    (!output.trim.isEmpty)
    "Mismatched ct.where failures should emit an elaboration diagnostic"

end Tests.GPUTileIR
