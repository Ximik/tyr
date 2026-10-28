import Tyr.Model.Flux
import Tyr.Checkpoint
import LeanTest
import Tests.Assertions

open torch torch.flux
namespace Tests.FluxTyped

private def cfg : FluxConfig := {
  hidden_size := 16, num_heads := 2, head_dim := 8,
  num_double_layers := 1, num_single_layers := 1,
  context_in_dim := 6, in_channels := 4, time_dim := 5,
  mlp_ratio := 2, axes_dims := #[2, 2, 2, 2] }

example (_model : FluxModel (device := .CPU) cfg .BFloat16)
    (_weight : Tensor ⟨#[cfg.hidden_size, cfg.in_channels], .Float32, .CPU⟩) : True := by
  fail_if_success
    have _ : FluxModel (device := .CPU) cfg .BFloat16 := { _model with img_in := _weight }
  trivial

private def close {σ τ : StaticSpec} (x : Tensor σ) (y : Tensor τ) (tol : Float := 1e-5) : IO Unit := do
  LeanTest.assertEqual x.runtimeShape y.runtimeShape
  let xs ← x.toFloatArray
  let ys ← y.toFloatArray
  for i in [:xs.size] do
    LeanTest.assertTrue (xs[i]!.isFinite && (xs[i]! - ys[i]!).abs <= tol)
      s!"Flux mismatch at {i}: {xs[i]!} vs {ys[i]!}"

@[test] def testFluxPositionIds : IO Unit := do
  let image ← computeImagePositionIds 2 3
  LeanTest.assertEqual image.actualSpec.dtype .Int64
  LeanTest.assertEqual (← image.toUInt64Array)
    #[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0]
  let text ← computeTextPositionIds 3
  LeanTest.assertEqual (← text.toUInt64Array) #[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2]

@[test] def testFluxRotaryReferenceAndDType : IO Unit := do
  let ids := (Tensor.fromInt64Array #[1, 0, 0, 0]).reshape #[1, 1, 4]
  let pe := ropeEmbed (head_dim := 8) ids #[2, 2, 2, 2]
  let c := Float.cos 1
  let s := Float.sin 1
  let expected := (Tensor.fromFloatArray #[c - s, s + c, 1, 1, 1, 1, 1, 1]).reshape #[1, 1, 1, 8]
  close (applyRope ((Tensor.ones #[1, 1, 1, 8])) pe) expected
  let x := (Tensor.ones #[1, 1, 1, 8]).toBFloat16
  let y := applyRope x pe
  LeanTest.assertEqual y.actualSpec.dtype .BFloat16
  close y expected.toBFloat16 0.0

@[test] def testFluxTimestepEmbeddingOddDimension : IO Unit := do
  let t := Tensor.fromFloatArray #[0, 0.001]
  let y := timestepEmbedding t 5
  let expected := Tensor.fromFloatArray #[1, 1, 0, 0, 0,
    Float.cos 1, Float.cos 0.01, Float.sin 1, Float.sin 0.01, 0]
  close y (expected.reshape #[2, 5])

@[test] def testFluxForwardAndDenoiseDTypes : IO Unit := do
  let initial ← FluxModel.init cfg
  let initial := TensorStruct.detach initial
  let imageIds ← computeImagePositionIds 2 3
  let textIds ← computeTextPositionIds 3
  let imageIds := Tensor.expand imageIds #[2, 6, 4]
  let textIds := Tensor.expand textIds #[2, 3, 4]
  let image ← randomWeight #[2, 6, 4]
  let image := image.detach
  let text ← randomWeight #[2, 3, 6]
  let text := text.detach
  for bf16 in #[false, true] do
    let selected : RealFloatingDType :=
      if bf16 then ⟨.BFloat16, by decide⟩ else ⟨.Float32, by decide⟩
    let dtype := selected.val
    let model ← TensorStruct.castTo (FluxModel (device := .CPU) cfg dtype) initial
    let image := Tensor.castLike model.img_in image
    let text := Tensor.castLike model.txt_in text
    let y := model.forward cfg image text (Tensor.full #[2] 0.5) imageIds textIds
    LeanTest.assertEqual y.runtimeShape #[2, 6, 4]
    LeanTest.assertEqual y.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    for v in (← y.toFloatArray) do LeanTest.assertTrue v.isFinite
    close (denoise cfg model image text imageIds textIds 0) image 0.0
    let out := denoise cfg model image text imageIds textIds 2
    LeanTest.assertEqual out.actualSpec.dtype y.actualSpec.dtype
    for v in (← out.toFloatArray) do LeanTest.assertTrue v.isFinite

@[test] def testFluxLatentPackRoundTrip : IO Unit := do
  let values := Array.ofFn (n := 2 * 32 * 4 * 6) fun i => i.val.toFloat
  let spatial := (Tensor.fromFloatArray values).reshapeChecked #[2, 32, 4, 6]
  let packed := packLatents spatial
  LeanTest.assertEqual packed.runtimeShape #[2, 6, 128]
  close (unpackLatents (packed_H := 2) (packed_W := 3) packed) spatial 0.0
  let actual ← packed.toFloatArray
  LeanTest.assertEqual (actual.extract 0 8) #[0, 1, 6, 7, 24, 25, 30, 31]

@[test] def testFluxPPMNativeBoundary : IO Unit := IO.FS.withTempDir fun dir => do
  let image := (Tensor.fromFloatArray #[1, -1, -1, 1, -1, -1]).reshapeChecked #[1, 3, 1, 2]
  let path := dir / "tiny.ppm"
  saveImagePPM image path.toString
  let bytes ← IO.FS.readBinFile path
  LeanTest.assertEqual bytes ("P6\n2 1\n255\n".toUTF8 ++ ByteArray.mk #[255, 0, 0, 0, 255, 0])

@[noinline, nospecialize]
private def roundTrip {α : Type} [TensorStruct α] (value : α) (dir : String) : IO α := do
  checkpoint.saveParams value dir
  checkpoint.loadParams value dir

@[test] def testFluxIndexedCheckpoint : IO Unit := IO.FS.withTempDir fun dir => do
  let model ← FluxModel.init cfg
  let restored ← roundTrip model dir.toString
  let imageIds ← computeImagePositionIds 1 2
  let textIds ← computeTextPositionIds 3
  let x := (Tensor.ones #[1, 2, 4])
  let txt := (Tensor.ones #[1, 3, 6])
  let t := Tensor.full #[1] 0.5
  close (model.forward cfg x txt t imageIds textIds) (restored.forward cfg x txt t imageIds textIds) 0.0

private def writeFluxCheckpoint (path : System.FilePath) (bf16 : Bool)
    (foreignName : String := "") : IO Unit := do
  let mut entries : Array (String × Array Nat) := #[
    ("img_in.weight", #[16, 4]), ("txt_in.weight", #[16, 6]),
    ("time_in.in_layer.weight", #[16, 5]), ("time_in.out_layer.weight", #[16, 16]),
    ("double_stream_modulation_img.lin.weight", #[96, 16]),
    ("double_stream_modulation_txt.lin.weight", #[96, 16]),
    ("single_stream_modulation.lin.weight", #[48, 16]),
    ("single_blocks.0.linear1.weight", #[112, 16]),
    ("single_blocks.0.linear2.weight", #[16, 48]),
    ("single_blocks.0.norm.query_norm.scale", #[8]),
    ("single_blocks.0.norm.key_norm.scale", #[8]),
    ("final_layer.linear.weight", #[4, 16]),
    ("final_layer.adaLN_modulation.1.weight", #[32, 16]) ]
  for stream in #["img", "txt"] do
    entries := entries ++ #[
      (s!"double_blocks.0.{stream}_attn.qkv.weight", #[48, 16]),
      (s!"double_blocks.0.{stream}_attn.proj.weight", #[16, 16]),
      (s!"double_blocks.0.{stream}_attn.norm.query_norm.scale", #[8]),
      (s!"double_blocks.0.{stream}_attn.norm.key_norm.scale", #[8]),
      (s!"double_blocks.0.{stream}_mlp.0.weight", #[64, 16]),
      (s!"double_blocks.0.{stream}_mlp.2.weight", #[16, 32]) ]
  let mut fields : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in entries do
    let bf16 := if name == foreignName then !bf16 else bf16
    let start := data.size
    for _ in [:shape.foldl (· * ·) 1] do
      data := data ++ (if bf16 then ByteArray.mk #[0, 62] else ByteArray.mk #[0, 0, 0, 62])
    fields := fields ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if bf16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
  let header := (Lean.Json.mkObj fields).compress.toUTF8
  let mut headerSize := ByteArray.empty
  for i in [:8] do headerSize := headerSize.push ((header.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile path (headerSize ++ header ++ data)

@[test] def testFluxNamedCheckpointDTypes : IO Unit := IO.FS.withTempDir fun dir => do
  for bf16 in #[false, true] do
    let path := dir / (if bf16 then "bf16.safetensors" else "fp32.safetensors")
    writeFluxCheckpoint path bf16
    let ⟨storedDType, model⟩ ← loadFluxModel path.toString cfg
    LeanTest.assertEqual model.double_blocks.size 1
    LeanTest.assertEqual model.single_blocks.size 1
    let dtype := if bf16 then DType.BFloat16 else .Float32
    LeanTest.assertEqual model.img_in.actualSpec.dtype dtype
    let some block := model.double_blocks[0]? | throw (IO.userError "missing double block")
    LeanTest.assertEqual block.img_attn.norm.query_norm.weight.actualSpec.dtype dtype
    let image := Tensor.castLike model.img_in (Tensor.ones #[1, 2, 4])
    let text := Tensor.castLike model.txt_in (Tensor.ones #[1, 3, 6])
    let imageIds ← computeImagePositionIds 1 2
    let textIds ← computeTextPositionIds 3
    let y := model.forward cfg image text (Tensor.full #[1] 0.5) imageIds textIds
    LeanTest.assertEqual y.actualSpec.dtype dtype
    for v in (← y.toFloatArray) do LeanTest.assertTrue v.isFinite
  let mixedPath := dir / "mixed.safetensors"
  writeFluxCheckpoint mixedPath false "txt_in.weight"
  Tests.assertThrows (loadFluxModel mixedPath.toString cfg)
    (some "Flux tensor 'txt_in.weight'")

end Tests.FluxTyped
