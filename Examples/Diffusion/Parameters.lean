import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Derive

/-! Parameter and rotary types owned by the discrete diffusion model.

These retain the shapes and zero-initialized residual projections previously
shared with the removed NanoProof example.
-/
namespace torch.diffusion

abbrev Activation (s : Shape) := Tensor { shape := s, dtype := .Float32, device := .CPU }
abbrev Tokens (s : Shape) := Tensor { shape := s, dtype := .Int64, device := .CPU }
abbrev Mask (s : Shape) := Tensor { shape := s, dtype := .Bool, device := .CPU }

structure RotaryCache (seq_len half : Nat) where
  cos : Activation #[seq_len, half]
  sin : Activation #[seq_len, half]

def RotaryCache.init (seq_len half : Nat) (base : Float := 10000.0)
    : IO (RotaryCache seq_len half) := do
  let (cos, sin) := Tensor.rotaryFreqs seq_len half base
  return { cos, sin }

structure AttentionParams (n_head half n_kv_head : Nat) where
  c_q : Activation #[n_head * (2 * half), n_head * (2 * half)]
  c_k : Activation #[n_kv_head * (2 * half), n_head * (2 * half)]
  c_v : Activation #[n_kv_head * (2 * half), n_head * (2 * half)]
  c_proj : Activation #[n_head * (2 * half), n_head * (2 * half)]
  deriving TensorStruct

structure MLPParams (n_embd : Nat) where
  c_fc : Activation #[4 * n_embd, n_embd]
  c_proj : Activation #[n_embd, 4 * n_embd]
  deriving TensorStruct

structure BlockParams (n_head half n_kv_head : Nat) where
  attn : AttentionParams n_head half n_kv_head
  mlp : MLPParams (n_head * (2 * half))
  deriving TensorStruct

def makeLeafParam {s : Shape} (t : Activation s) : IO (Activation s) :=
  Tensor.parameter t

def AttentionParams.init (n_head half n_kv_head : Nat)
    : IO (AttentionParams n_head half n_kv_head) := do
  let scale := 1.0 / Float.sqrt (n_head * (2 * half)).toFloat
  let c_q ← Tensor.randn #[n_head * (2 * half), n_head * (2 * half)] false
  let c_k ← Tensor.randn #[n_kv_head * (2 * half), n_head * (2 * half)] false
  let c_v ← Tensor.randn #[n_kv_head * (2 * half), n_head * (2 * half)] false
  return {
    c_q := ← makeLeafParam ((c_q * scale))
    c_k := ← makeLeafParam ((c_k * scale))
    c_v := ← makeLeafParam ((c_v * scale))
    c_proj := ← makeLeafParam (Tensor.zeros #[n_head * (2 * half), n_head * (2 * half)])
  }

def MLPParams.init (n_embd : Nat) : IO (MLPParams n_embd) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_fc ← Tensor.randn #[4 * n_embd, n_embd] false
  return {
    c_fc := ← makeLeafParam ((c_fc * scale))
    c_proj := ← makeLeafParam (Tensor.zeros #[n_embd, 4 * n_embd])
  }

def BlockParams.init (n_head half n_kv_head : Nat)
    : IO (BlockParams n_head half n_kv_head) := do
  return { attn := ← AttentionParams.init n_head half n_kv_head,
           mlp := ← MLPParams.init (n_head * (2 * half)) }

end torch.diffusion
