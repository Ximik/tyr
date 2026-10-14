import Tyr.Model.VAE
import LeanTest

open torch torch.vae

namespace Tests.VAETyped

private def close {σ τ : StaticSpec} (x : Tensor σ) (y : Tensor τ) (tol : Float := 1e-5) : IO Unit := do
  LeanTest.assertEqual x.runtimeShape y.runtimeShape
  let xs ← x.toFloatArray
  let ys ← y.toFloatArray
  for i in [:xs.size] do
    LeanTest.assertTrue (xs[i]!.isFinite && (xs[i]! - ys[i]!).abs <= tol)
      s!"VAE mismatch at {i}: {xs[i]!} vs {ys[i]!}"

@[test] def testVAEGroupNormAndGradient : IO Unit := do
  let x := Tensor.setRequiresGrad ((Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6, 7, 8]).reshapeChecked #[1, 2, 2, 2]).forgetDType true
  let gn := GroupNormParams.init 2 2
  let y := gn.forward x
  let expected := (Tensor.fromFloatArray #[-1.34164025, -0.44721342, 0.44721342, 1.34164025,
    -1.34164025, -0.44721342, 0.44721342, 1.34164025]).reshapeChecked #[1, 2, 2, 2]
  close y expected
  let grad := Tensor.grad y x ((Tensor.ones #[1, 2, 2, 2]).forgetDType)
  close grad (Tensor.zeros #[1, 2, 2, 2])
  let seed := (Tensor.fromFloatArray #[1, 0, 0, 0, 1, 0, 0, 0]).reshapeChecked #[1, 2, 2, 2] |>.forgetDType
  let gradient := Tensor.grad y x seed
  close gradient ((Tensor.fromFloatArray #[0.268328, -0.357771, -0.089443, 0.178885,
    0.268328, -0.357771, -0.089443, 0.178885]).reshapeChecked #[1, 2, 2, 2])

@[test] def testVAEUpsampleAndOddDownsample : IO Unit := do
  let x := (Tensor.fromFloatArray #[1, 2, 3, 4, 5, 6]).reshapeChecked #[1, 1, 2, 3] |>.forgetDType
  let y := upsample2 x
  close y ((Tensor.fromFloatArray #[1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]).reshapeChecked #[1, 1, 4, 6])
  let down : Downsample 1 := { conv := {
    weight := (Tensor.ones #[1, 1, 3, 3]).forgetDType
    bias := (Tensor.zeros #[1]).forgetDType } }
  let odd := down.forward ((Tensor.ones #[1, 1, 3, 5]).forgetDType)
  LeanTest.assertEqual odd.runtimeShape #[1, 1, 2, 3]
  close odd ((Tensor.fromFloatArray #[4, 6, 4, 4, 6, 4]).reshapeChecked #[1, 1, 2, 3])

@[test] def testVAEAttentionAndResidualDTypes : IO Unit := do
  let attn ← AttnBlock.init 32
  let block ← ResnetBlock.init 32 64
  for bf16 in #[false, true] do
    let attn ← if bf16 then TensorStruct.castDType .BFloat16 attn else pure attn
    let block ← if bf16 then TensorStruct.castDType .BFloat16 block else pure block
    let x ← Tensor.randn #[2, 32, 2, 3]
    let x := Tensor.castLike attn.q.weight x
    let y := attn.forward x
    LeanTest.assertEqual y.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    let z := block.forward y
    LeanTest.assertEqual z.runtimeShape #[2, 64, 2, 3]
    LeanTest.assertEqual z.actualSpec.dtype y.actualSpec.dtype
    for v in (← z.toFloatArray) do LeanTest.assertTrue v.isFinite

@[test] def testVAEFullDecoderIndexedSpatialShape : IO Unit := do
  let dec ← Decoder.init
  let x ← Tensor.randn #[1, 32, 2, 3]
  let output := dec.forward x.forgetDType
  LeanTest.assertEqual output.runtimeShape #[1, 3, 16, 24]
  for v in (← output.toFloatArray) do LeanTest.assertTrue v.isFinite

@[test] def testVAELatentUnpack : IO Unit := do
  let xs := Array.ofFn (n := 128 * 16 * 16) fun i => i.val.toFloat
  let input := (Tensor.fromFloatArray xs).reshapeChecked #[1, 128, 16, 16] |>.forgetDType
  let result ← (AutoEncoder.unpackLatents16x16 input).toFloatArray
  for c in [:32] do
    for h in [:32] do
      for w in [:32] do
        let source := ((c * 2 + h % 2) * 2 + w % 2) * 256 + (h / 2) * 16 + w / 2
        LeanTest.assertEqual result[c * 1024 + h * 32 + w]! xs[source]!

private def writeConvCheckpoint (path : System.FilePath) (bf16 wrongShape : Bool) : IO Unit := do
  let mut fields : List (String × Lean.Json) := []
  let mut data := ByteArray.empty
  for (name, shape) in #[("conv.weight", if wrongShape then #[1, 1, 1, 9] else #[1, 1, 3, 3]),
                         ("conv.bias", #[1])] do
    let start := data.size
    for _ in [:shape.foldl (· * ·) (1 : Nat)] do
      data := data ++ (if bf16 then ByteArray.mk #[128, 63] else ByteArray.mk #[0, 0, 128, 63])
    fields := fields ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if bf16 then "BF16" else "F32")),
      ("shape", Lean.toJson shape), ("data_offsets", Lean.toJson #[start, data.size])])]
  let header := (Lean.Json.mkObj fields).compress.toUTF8
  let mut headerSize := ByteArray.empty
  for i in [:8] do headerSize := headerSize.push ((header.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile path (headerSize ++ header ++ data)

@[test] def testVAECheckpointDTypeAndShapeValidation : IO Unit := IO.FS.withTempDir fun dir => do
  for bf16 in #[false, true] do
    let path := dir / (if bf16 then "bf16.safetensors" else "f32.safetensors")
    writeConvCheckpoint path bf16 false
    let conv ← loadConv2d path.toString "conv" 1 1 3 1
    LeanTest.assertEqual conv.weight.actualSpec.dtype (if bf16 then .BFloat16 else .Float32)
    close (conv.forward (Tensor.castLike conv.weight (Tensor.ones #[1, 1, 3, 3])))
      ((Tensor.fromFloatArray #[4, 6, 4, 6, 9, 6, 4, 6, 4]).reshapeChecked #[1, 1, 3, 3])
  let path := dir / "invalid.safetensors"
  writeConvCheckpoint path false true
  let rejected ← try
    let _ ← loadConv2d path.toString "conv" 1 1 3 1
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "must reject an equal-numel convolution weight with the wrong layout"
  IO.FS.writeFile (dir / "model.safetensors.index.json")
    "{\"weight_map\":{\"conv.weight\":\"invalid.safetensors\"}}"
  let sharded ← safetensors.tryLoadTensorSharded dir.toString "conv.weight" #[1, 1, 3, 3]
  LeanTest.assertTrue sharded.isNone "sharded loading must reject the same wrong layout"

end Tests.VAETyped
