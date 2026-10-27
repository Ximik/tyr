/- End-to-end numerical correctness harness for the TK-style decode kernel.

   Generates random Q/K/V plus a PyTorch SDPA reference for several decode
   shapes, then invokes `tyr::flash_attn` (via `Tensor.flashAttention`) and asserts
   the kernel output matches reference within BF16 tolerance.

   Shape coverage:
   - Llama-3 (head_dim=128, GQA ratio 4, kv_seq=2048)
   - Qwen3-4B (head_dim=64, GQA ratio 4, kv_seq=2048)
   - Tail-mask exercise (head_dim=128, kv_seq=2049 — not multiple of 64)
   - Single-block (head_dim=128, kv_seq=64)
   - Larger batch (head_dim=128, batch=4)

   Each shape gets its own fixture directory under `data/gpu_fixtures/`, so
   they regenerate independently and live across runs. The runner reuses the
   shared `FixtureRunner` helper used by `RunFlashAttn3.lean` etc. -/
import Tyr.Torch
import Tyr.Typed.Neural
import Examples.GPU.DecodeReference
import Tyr.Inference.KVCache
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch

variable {device : Device}
open torch.Generator

/-- One decode test case. -/
structure DecodeShape where
  name : String
  batch : UInt64
  qHeads : UInt64
  kvHeads : UInt64
  kvSeq : UInt64
  headDim : UInt64
  /-- Max-abs tolerance for the BF16 output comparison. -/
  atol : Float := 2.5e-2
  /-- Relative tolerance. -/
  rtol : Float := 2.5e-2

/-- Decode-shape coverage exercised by this harness. -/
def decodeShapes : Array DecodeShape := #[
  { name := "llama3_b1", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { name := "llama3_b4", batch := 4, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 128 },
  { name := "llama3_tail", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2049, headDim := 128 },
  { name := "llama3_one_block", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 64, headDim := 128 },
  { name := "qwen3_4b_d64", batch := 1, qHeads := 32, kvHeads := 8,
    kvSeq := 2048, headDim := 64 },
  -- Qwen 3.5/3.6 family (35B-A3B and friends): head_dim=256, GQA ratio 8
  -- (16 q heads, 2 kv heads). Larger atol because the per-warp 64×256
  -- accumulator likely spills to local memory under launch_bounds(128, 1)
  -- — see dev/issues.md D01b.
  { name := "qwen36_35B", batch := 1, qHeads := 16, kvHeads := 2,
    kvSeq := 2048, headDim := 256, atol := 5.0e-2, rtol := 5.0e-2 }
]

def decodeFixtureDir (shape : DecodeShape) : System.FilePath :=
  ⟨s!"data/gpu_fixtures/decode_{shape.name}"⟩

def decodeFixtureSpec (shape : DecodeShape) : FixtureSpec := {
  dir := decodeFixtureDir shape
  names := #["q", "k", "v", "expected_o"]
}

def decodeFixtureFile (shape : DecodeShape) (name : String) : System.FilePath :=
  fixturePath (decodeFixtureSpec shape) name

private def shapeQ (s : DecodeShape) : Shape :=
  #[s.batch, s.qHeads, 1, s.headDim]

private def shapeKV (s : DecodeShape) : Shape :=
  #[s.batch, s.kvHeads, s.kvSeq, s.headDim]

/-- Generate a fixture set for one decode shape: random Q/K/V (BF16) plus the
    SDPA reference output computed in FP32 and quantized back to BF16. -/
def generateDecodeFixturesFor (shape : DecodeShape) : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate decode fixtures."
  IO.FS.createDirAll (decodeFixtureSpec shape).dir
  let device ← Device.cuda 0

  let qf ← Tensor.randn (shapeQ shape) false device
  let kf ← Tensor.randn (shapeKV shape) false device
  let vf ← Tensor.randn (shapeKV shape) false device

  let q := qf.toBFloat16
  let k := kf.toBFloat16
  let v := vf.toBFloat16

  -- Reference: PyTorch SDPA in FP32, with GQA enabled (broadcasts kv heads to q heads).
  let q32 := q.toFloat32
  let k32 := k.toFloat32
  let v32 := v.toFloat32
  let expected32 := Tensor.attentionGQAQKV
                      q32 k32 v32 false true
  let expected := expected32.toBFloat16

  Tensor.saveTensor q (decodeFixtureFile shape "q").toString
  Tensor.saveTensor k (decodeFixtureFile shape "k").toString
  Tensor.saveTensor v (decodeFixtureFile shape "v").toString
  Tensor.saveTensor expected (decodeFixtureFile shape "expected_o").toString

  let mae := Tensor.item (Tensor.meanAll (Tensor.abs expected))
  IO.println s!"  generated decode fixture {shape.name} kvSeq={shape.kvSeq} d={shape.headDim} ref_mean_abs={mae}"

/-- Run the kernel on one shape's saved fixtures and assert parity. -/
def runDecodeOnceFor (shape : DecodeShape) : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  let device ← Device.cuda 0
  if !(← fixturesPresent (decodeFixtureSpec shape)) then
    generateDecodeFixturesFor shape

  let q ← Tensor.loadTensorExact { device := device, shape := shapeQ shape, dtype := .BFloat16 } (decodeFixtureFile shape "q").toString
  let k ← Tensor.loadTensorExact { device := device, shape := shapeKV shape, dtype := .BFloat16 } (decodeFixtureFile shape "k").toString
  let v ← Tensor.loadTensorExact { device := device, shape := shapeKV shape, dtype := .BFloat16 } (decodeFixtureFile shape "v").toString
  let expected ← Tensor.loadTensorExact { device := device, shape := shapeQ shape, dtype := .BFloat16 } (decodeFixtureFile shape "expected_o").toString

  -- Route through the production op. select_route picks tkMhaH100Decode[64]
  -- when shape is eligible; else falls back to SDPA.
  let actual : Tensor { shape := (shapeQ shape), dtype := .BFloat16, device := device } :=
    Tensor.flashAttention q k v true
  let _ ← torch.cuda_synchronize

  let ok := Tensor.allclose expected actual shape.atol shape.rtol
  let diff := Tensor.abs (actual - expected)
  let mae := Tensor.item (Tensor.meanAll diff)
  let maxAbs := Tensor.item (Tensor.maxAllChecked diff)
  let actMaxAbs := Tensor.item (Tensor.maxAllChecked (Tensor.abs actual))
  let actMeanAbs := Tensor.item (Tensor.meanAll (Tensor.abs actual))
  IO.println s!"  decode {shape.name}: ok={ok} mae={mae} max_abs={maxAbs} act_mean_abs={actMeanAbs} act_max_abs={actMaxAbs}"
  -- Per-(batch, q_head) row diagnostics: which heads have correct vs zero output?
  -- mae over the head_dim axis for each (b, h). Expected is non-trivial; if our
  -- kernel only fills some heads correctly, we'll see a binary distribution.
  let perHeadAct := Tensor.maxAllChecked (Tensor.abs actual)
  let perHeadExp := Tensor.maxAllChecked (Tensor.abs expected)
  IO.println s!"    per-shape actMax={Tensor.item perHeadAct} expMax={Tensor.item perHeadExp}"
  pure ok

/-- Cache-vs-no-cache parity: generate K, V for a sequence of N tokens, then
    run the same attention two ways and compare:

    1. Single dense forward over the full K, V (`Tensor.attentionGQAQKV`).
    2. Token-by-token with `Cache.attendLayer`, accumulating per-step outputs.

    Both should produce the same per-token output up to BF16 tolerance.
    Catches off-by-one in append position, stale slice bounds, and any
    softmax-rescale issues across the kernel's tail-mask path. -/
def runCacheParityOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false
  runDecodeCacheParity (← Device.cuda 0)

/-- Combined runner: every shape in `decodeShapes` must pass, then the cache
    parity test must pass. Returns `0` only when all passed. -/
def runDecodeAll (regen : Bool) : IO UInt32 := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1
  let mut allOk := true
  for shape in decodeShapes do
    if regen then generateDecodeFixturesFor shape
    let ok ← runDecodeOnceFor shape
    if !ok then allOk := false
  let cacheOk ← runCacheParityOnce
  if !cacheOk then allOk := false
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  let regen := args.contains "--regen"
  let genOnly := args.contains "--gen-only"
  if regen || genOnly then
    if !(← torch.cuda_is_available) then
      IO.eprintln "CUDA is not available; cannot regenerate decode fixtures."
      return 1
    for shape in decodeShapes do
      generateDecodeFixturesFor shape
  if genOnly then return 0
  runDecodeAll regen

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
