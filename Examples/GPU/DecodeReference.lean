import Tyr.Typed.Neural
import Tyr.Inference.KVCache

namespace Examples.GPU
open torch

variable {device : Device}

/-- Compare token-by-token cached GQA against a full causal attention pass. -/
def runDecodeCacheParity (device : Device) : IO Bool := do
  -- Small Llama-3 shape: kvSeq=10 short enough to invoke the tail mask path on
  -- every step; head_dim=128.
  let batch : UInt64 := 1
  let qHeads : UInt64 := 32
  let kvHeads : UInt64 := 8
  let totalLen : UInt64 := 10
  let maxLen : UInt64 := 128
  let headDim : UInt64 := 128

  -- Random "all-positions" Q, K, V (FP32 → BF16) representing N tokens.
  let qf ← Tensor.randn #[batch, qHeads, totalLen, headDim] false device
  let kf ← Tensor.randn #[batch, kvHeads, totalLen, headDim] false device
  let vf ← Tensor.randn #[batch, kvHeads, totalLen, headDim] false device
  let qAll := qf.toBFloat16
  let kAll := kf.toBFloat16
  let vAll := vf.toBFloat16

  -- Reference: do attention over the full sequence (causal=true, q_seq=kv_seq).
  let refAll : Tensor { shape := #[batch, qHeads, totalLen, headDim], dtype := .BFloat16, device := device } :=
    Tensor.attentionGQAQKV qAll kAll vAll true true

  -- Build a 1-layer cache, append tokens one at a time, run cached attention.
  let mut cache ← Generator.KVCache.Cache.init 1 batch maxLen kvHeads headDim (device := device)
  -- Accumulate per-step kernel outputs into a list, then stack.
  let mut acc : Array (Tensor { shape := #[batch, qHeads, 1, headDim], dtype := .BFloat16, device := device }) := #[]
  for i in [:totalLen.toNat] do
    let pos : UInt64 := i.toUInt64
    let qStep : Tensor { shape := #[batch, qHeads, 1, headDim], dtype := .BFloat16, device := device } :=
      Tensor.sliceChecked qAll 2 pos 1 #[batch, qHeads, 1, headDim]
    let kStep : Tensor { shape := #[batch, kvHeads, 1, headDim], dtype := .BFloat16, device := device } :=
      Tensor.sliceChecked kAll 2 pos 1 #[batch, kvHeads, 1, headDim]
    let vStep : Tensor { shape := #[batch, kvHeads, 1, headDim], dtype := .BFloat16, device := device } :=
      Tensor.sliceChecked vAll 2 pos 1 #[batch, kvHeads, 1, headDim]
    let (cache', out) ← cache.attendLayer (numQHeads := qHeads)
        0 qStep kStep vStep (enableGqa := true)
    cache ← cache'.incrementSeqLens
    acc := acc.push out

  if device != .CPU then
    let _ ← torch.cuda_synchronize
    pure ()

  -- Compare each step's cached output to the corresponding slice of refAll.
  let mut allOk := true
  let mut maxAbsAll : Float := 0.0
  for i in [:totalLen.toNat] do
    let pos : UInt64 := i.toUInt64
    let refStep : Tensor { shape := #[batch, qHeads, 1, headDim], dtype := .BFloat16, device := device } :=
      Tensor.sliceChecked refAll 2 pos 1 #[batch, qHeads, 1, headDim]
    let some cachedStep := acc[i]? |
      throw <| IO.userError s!"Missing cached decode output at step {i}"
    let diff := Tensor.abs (cachedStep - refStep)
    let maxAbs := Tensor.item (Tensor.maxAll diff)
    if maxAbs > maxAbsAll then maxAbsAll := maxAbs
    let ok := Tensor.allclose refStep cachedStep 3e-2 3e-2
    if !ok then
      allOk := false
      IO.println s!"  cache_parity step {i}: max_abs={maxAbs} (FAIL)"

  IO.println s!"cache_parity: ok={allOk} max_abs_overall={maxAbsAll}"
  pure allOk

end Examples.GPU
