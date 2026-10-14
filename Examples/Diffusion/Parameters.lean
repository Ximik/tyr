import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Derive

/-! Parameter and rotary types owned by the discrete diffusion model.

These retain the shapes and zero-initialized residual projections previously
shared with the removed NanoProof example.
-/
namespace torch.diffusion

abbrev Activation (s : Shape) := Tensor { shape := s, dtype := .Float32 }
abbrev Tokens (s : Shape) := Tensor { shape := s, dtype := .Int64 }
abbrev Mask (s : Shape) := Tensor { shape := s, dtype := .Bool }

structure RotaryCache (seq_len head_dim : UInt64) where
  cos : Activation #[seq_len, head_dim / 2]
  sin : Activation #[seq_len, head_dim / 2]

def RotaryCache.init (seq_len head_dim : UInt64) (base : Float := 10000.0)
    : IO (RotaryCache seq_len head_dim) := do
  let (cos, sin) := Tensor.rotaryFreqs seq_len head_dim base
  return { cos, sin }

structure AttentionParams (n_embd n_head n_kv_head : UInt64) where
  c_q : Activation #[n_head * (n_embd / n_head), n_embd]
  c_k : Activation #[n_kv_head * (n_embd / n_head), n_embd]
  c_v : Activation #[n_kv_head * (n_embd / n_head), n_embd]
  c_proj : Activation #[n_embd, n_embd]
  deriving TensorStruct

structure MLPParams (n_embd : UInt64) where
  c_fc : Activation #[4 * n_embd, n_embd]
  c_proj : Activation #[n_embd, 4 * n_embd]
  deriving TensorStruct

structure BlockParams (n_embd n_head n_kv_head : UInt64) where
  attn : AttentionParams n_embd n_head n_kv_head
  mlp : MLPParams n_embd
  deriving TensorStruct

def makeLeafParam {s : Shape} (t : Activation s) : Activation s :=
  t.detach.setRequiresGrad true

def AttentionParams.init (n_embd n_head n_kv_head : UInt64)
    : IO (AttentionParams n_embd n_head n_kv_head) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_q ← Tensor.randn #[n_head * (n_embd / n_head), n_embd] false
  let c_k ← Tensor.randn #[n_kv_head * (n_embd / n_head), n_embd] false
  let c_v ← Tensor.randn #[n_kv_head * (n_embd / n_head), n_embd] false
  return {
    c_q := makeLeafParam ((c_q * scale))
    c_k := makeLeafParam ((c_k * scale))
    c_v := makeLeafParam ((c_v * scale))
    c_proj := makeLeafParam (Tensor.zeros #[n_embd, n_embd])
  }

def MLPParams.init (n_embd : UInt64) : IO (MLPParams n_embd) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_fc ← Tensor.randn #[4 * n_embd, n_embd] false
  return {
    c_fc := makeLeafParam ((c_fc * scale))
    c_proj := makeLeafParam (Tensor.zeros #[n_embd, 4 * n_embd])
  }

def BlockParams.init (n_embd n_head n_kv_head : UInt64)
    : IO (BlockParams n_embd n_head n_kv_head) := do
  return { attn := ← AttentionParams.init n_embd n_head n_kv_head,
           mlp := ← MLPParams.init n_embd }

end torch.diffusion
