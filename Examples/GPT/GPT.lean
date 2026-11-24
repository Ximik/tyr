/-
  GPT-2 Model Implementation

  A dependently-typed GPT model for Tyr, enabling type-safe transformer training.
  Shapes are tracked at compile time to catch dimension mismatches.
-/
import Examples.GPT.TensorOps
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.TensorStruct.DeriveCast
import Tyr.Optim

namespace torch.gpt

variable {device : Device}

/-- GPT model configuration -/
structure Config where
  vocab_size : Nat
  block_size : Nat  -- max sequence length
  n_embd : Nat      -- embedding dimension
  n_head : Nat      -- number of attention heads
  n_layer : Nat     -- number of transformer blocks
  dropout : Float := 0.0
  deriving Repr

/-- Tiny config for quick testing on Shakespeare (65 char vocab) -/
def Config.tiny_shakespeare : Config :=
  { vocab_size := 65, block_size := 64, n_embd := 64, n_head := 4, n_layer := 2, dropout := 0.0 }

/-- nanoGPT CPU config for Shakespeare: 4 layers, 4 heads, 128 embd, no dropout -/
def Config.nanogpt_cpu_shakespeare : Config :=
  { vocab_size := 65, block_size := 64, n_embd := 128, n_head := 4, n_layer := 4, dropout := 0.0 }

/-- Parameters for a single transformer block -/
structure BlockParams (n_embd : Nat) where
  -- Layer norm 1 (pre-attention)
  ln1_weight : Activation (device := device) #[n_embd]
  ln1_bias : Activation (device := device) #[n_embd]
  -- Attention: separate Q, K, V projections (avoids tensor split)
  -- Tensor.linear3d expects weight [out_dim, in_dim], computes x @ W.T
  q_proj : Activation (device := device) #[n_embd, n_embd]
  k_proj : Activation (device := device) #[n_embd, n_embd]
  v_proj : Activation (device := device) #[n_embd, n_embd]
  -- Attention output projection
  c_proj : Activation (device := device) #[n_embd, n_embd]
  c_proj_bias : Activation (device := device) #[n_embd]
  -- Layer norm 2 (pre-MLP)
  ln2_weight : Activation (device := device) #[n_embd]
  ln2_bias : Activation (device := device) #[n_embd]
  -- MLP: expand then project back
  -- mlp_fc: n_embd -> 4*n_embd, weight shape [out, in] = [4*n_embd, n_embd]
  mlp_fc : Activation (device := device) #[4 * n_embd, n_embd]
  mlp_fc_bias : Activation (device := device) #[4 * n_embd]
  -- mlp_proj: 4*n_embd -> n_embd, weight shape [out, in] = [n_embd, 4*n_embd]
  mlp_proj : Activation (device := device) #[n_embd, 4 * n_embd]
  mlp_proj_bias : Activation (device := device) #[n_embd]
  deriving TensorStruct

/-- Full GPT model parameters -/
structure GPTParams (cfg : Config) where
  -- Embeddings
  wte : Activation (device := device) #[cfg.vocab_size, cfg.n_embd]    -- token embeddings
  wpe : Activation (device := device) #[cfg.block_size, cfg.n_embd]   -- position embeddings
  -- Transformer blocks
  blocks : Vector (BlockParams (device := device) cfg.n_embd) cfg.n_layer
  -- Final layer norm
  ln_f_weight : Activation (device := device) #[cfg.n_embd]
  ln_f_bias : Activation (device := device) #[cfg.n_embd]
  -- Note: lm_head shares weights with wte (weight tying)
  deriving TensorStruct

/-- Initialize block parameters with scaled random weights -/
def BlockParams.init (n_embd : Nat) (scale : Float := 0.02) (device : Device := Device.CPU) : IO (BlockParams (device := device) n_embd) := do
  -- Create scaled random tensors as leaf tensors on target device
  -- randn * scale creates a non-leaf, so we detach and set requires_grad
  let q_proj ← Tensor.randn #[n_embd, n_embd] false device
  let k_proj ← Tensor.randn #[n_embd, n_embd] false device
  let v_proj ← Tensor.randn #[n_embd, n_embd] false device
  let c_proj ← Tensor.randn #[n_embd, n_embd] false device
  let mlp_fc ← Tensor.randn #[4*n_embd, n_embd] false device
  let mlp_proj ← Tensor.randn #[n_embd, 4*n_embd] false device
  return {
    ln1_weight := ← Tensor.parameter (Tensor.ones #[n_embd] false device)
    ln1_bias := ← Tensor.parameter (Tensor.zeros #[n_embd] false device)
    q_proj := ← Tensor.parameter (q_proj * scale)
    k_proj := ← Tensor.parameter (k_proj * scale)
    v_proj := ← Tensor.parameter (v_proj * scale)
    c_proj := ← Tensor.parameter (c_proj * scale)
    c_proj_bias := ← Tensor.parameter (Tensor.zeros #[n_embd] false device)
    ln2_weight := ← Tensor.parameter (Tensor.ones #[n_embd] false device)
    ln2_bias := ← Tensor.parameter (Tensor.zeros #[n_embd] false device)
    mlp_fc := ← Tensor.parameter (mlp_fc * scale)
    mlp_fc_bias := ← Tensor.parameter (Tensor.zeros #[4*n_embd] false device)
    mlp_proj := ← Tensor.parameter (mlp_proj * scale)
    mlp_proj_bias := ← Tensor.parameter (Tensor.zeros #[n_embd] false device)
  }

/-- Initialize full GPT model -/
def GPTParams.init (cfg : Config) (device : Device := Device.CPU) : IO (GPTParams (device := device) cfg) := do
  let wte ← Tensor.randn #[cfg.vocab_size, cfg.n_embd] false device
  let wpe ← Tensor.randn #[cfg.block_size, cfg.n_embd] false device
  let blocks ← Vector.ofFnM fun (_ : Fin cfg.n_layer) => BlockParams.init cfg.n_embd 0.02 device
  return {
    wte := ← Tensor.parameter (Tensor.mulScalar wte 0.02)
    wpe := ← Tensor.parameter (Tensor.mulScalar wpe 0.02)
    blocks := blocks
    ln_f_weight := ← Tensor.parameter (Tensor.ones #[cfg.n_embd] false device)
    ln_f_bias := ← Tensor.parameter (Tensor.zeros #[cfg.n_embd] false device)
  }

/-- Forward pass through a single transformer block with multi-head attention and dropout -/
def blockForward {n_embd n_head batch seq : Nat}
    (params : BlockParams (device := device) n_embd)
    (x : Activation (device := device) #[batch, seq, n_embd])
    (dropout_p : Float := 0.0)
    (training : Bool := true) : IO (Activation (device := device) #[batch, seq, n_embd]) := do
  -- Pre-norm attention
  let h := Tensor.layerNorm x params.ln1_weight params.ln1_bias
  -- Q, K, V projections: [batch, seq, n_embd] @ [n_embd, n_embd] -> [batch, seq, n_embd]
  let q := Tensor.linear3d h params.q_proj
  let k := Tensor.linear3d h params.k_proj
  let v := Tensor.linear3d h params.v_proj
  -- Reshape for multi-head attention:
  -- [batch, seq, n_embd] -> [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  if n_head == 0 then
    throw (IO.userError "GPT embedding dimension must be divisible by a nonzero head count")
  let ⟨hDiv⟩ ← if h : n_embd % n_head = 0 then pure (PLift.up h) else
    throw (IO.userError "GPT embedding dimension must be divisible by a nonzero head count")
  let head_dim := n_embd / n_head
  have hHeads : batch * seq * n_embd = batch * seq * n_head * head_dim := by
    rw [Nat.mul_assoc _ n_head, Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hDiv)]
  let q := Tensor.splitHeads (q.reshape #[batch, seq, n_head, head_dim] (by simpa using hHeads))
  let k := Tensor.splitHeads (k.reshape #[batch, seq, n_head, head_dim] (by simpa using hHeads))
  let v := Tensor.splitHeads (v.reshape #[batch, seq, n_head, head_dim] (by simpa using hHeads))
  -- Scaled dot-product attention with causal masking (with dropout built-in)
  -- Input: [batch, n_head, seq, head_dim], Output: [batch, n_head, seq, head_dim]
  let attn ← Tensor.attentionIO q k v dropout_p training true
  -- Reshape back: [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim] -> [batch, seq, n_embd]
  let attn := (Tensor.joinHeads attn).reshape #[batch, seq, n_embd] (by simpa using hHeads.symm)
  -- Output projection + dropout
  let attn := Tensor.affine3d attn params.c_proj params.c_proj_bias
  let attn ← Tensor.dropout attn dropout_p training
  -- Residual connection
  let x := x + attn
  -- Pre-norm MLP
  let h := Tensor.layerNorm x params.ln2_weight params.ln2_bias
  let h := Tensor.affine3d h params.mlp_fc params.mlp_fc_bias
  let h := Tensor.gelu h (by rfl)
  let h := Tensor.affine3d h params.mlp_proj params.mlp_proj_bias
  -- MLP dropout
  let h ← Tensor.dropout h dropout_p training
  -- Residual connection
  return x + h

/-- Full GPT forward pass with dropout -/
def forward {cfg : Config} {batch seq : Nat}
    (params : GPTParams (device := device) cfg)
    (idx : Tokens (device := device) #[batch, seq])
    (training : Bool := true) : IO (Activation (device := device) #[batch, seq, cfg.vocab_size]) := do
  -- Token embeddings: [batch, seq] -> [batch, seq, n_embd]
  let tok_emb := Tensor.embedding idx params.wte
  -- Position embeddings (create position indices): [seq] -> [seq, n_embd]
  -- Move positions to the same device as the embedding weights
  let positions := (Tensor.arange seq).to params.wpe.device
  let pos_emb := Tensor.embedding1d positions params.wpe
  -- Expand pos_emb to match batch: [seq, n_embd] -> [batch, seq, n_embd]
  let pos_emb_expanded := Tensor.broadcastTo pos_emb #[batch, seq, cfg.n_embd]
  -- Combine embeddings + dropout: [batch, seq, n_embd]
  let x : Activation (device := device) #[batch, seq, cfg.n_embd] := Tensor.add tok_emb pos_emb_expanded
  let x ← Tensor.dropout x cfg.dropout training
  -- Apply transformer blocks with multi-head attention and dropout
  let mut x := x
  for block in params.blocks do
    x ← blockForward (n_head := cfg.n_head) block x cfg.dropout training
  -- Final layer norm
  let xNorm := Tensor.layerNorm x params.ln_f_weight params.ln_f_bias
  -- Output projection (weight-tied with token embeddings): [batch, seq, n_embd] @ [n_embd, vocab] -> [batch, seq, vocab]
  let wte_t := Tensor.transpose2d params.wte  -- [vocab, n_embd] -> [n_embd, vocab]
  return Tensor.matmul3d xNorm wte_t

/-- Compute cross-entropy loss (training=true enables dropout) -/
def loss {cfg : Config} {batch seq : Nat}
    (params : GPTParams (device := device) cfg)
    (idx : Tokens (device := device) #[batch, seq])
    (targets : Tokens (device := device) #[batch, seq])
    (training : Bool := true) : IO (Activation (device := device) #[]) := do
  let logits ← forward params idx training
  -- Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
  let logits_flat := logits.reshape #[batch * seq, cfg.vocab_size]
  let targets_flat := targets.reshape #[batch * seq]
  pure (Tensor.crossEntropy! logits_flat targets_flat)

/-- Generate tokens autoregressively from the model
    Takes initial context and generates maxNewTokens more tokens -/
def generate {cfg : Config}
    (params : GPTParams (device := device) cfg)
    (context : Array Int64)  -- Initial token IDs
    (maxNewTokens : Nat)
    (temperature : Float := 1.0)
    : IO (Array Int64) := do
  if maxNewTokens > 0 && (context.isEmpty || cfg.block_size == 0) then
    throw <| IO.userError "GPT generation requires nonempty context and positive block size"
  if temperature <= 0 || temperature.isNaN || temperature.isInf then
    throw <| IO.userError "GPT generation requires finite positive temperature"
  let mut tokens := context

  for _ in [:maxNewTokens] do
    -- Crop context to block_size if needed
    let ctxLen := min tokens.size cfg.block_size
    let startIdx := tokens.size - ctxLen
    let ctx := tokens[startIdx:].toArray

    -- Create input tensor [1, seq] from token array and move to model's device
    let seq : Nat := ctxLen
    let inputTensor := (← IO.ofExcept (Tensor.fromInt64ArrayAs? ctx #[1, seq])).to params.wte.device

    -- Forward pass to get logits [1, seq, vocab] (training=false disables dropout)
    let logits ← forward params inputTensor false

    let ⟨hSeq⟩ ← if h : 0 < seq then pure (PLift.up h) else
      throw <| IO.userError "GPT generation requires a non-empty context"
    let lastLogits := (logits.slice 1 (seq - 1) 1).reshape #[1, cfg.vocab_size]

    -- Apply temperature and convert to probabilities
    let scaledLogits := lastLogits / temperature
    let probs := Tensor.softmax scaledLogits

    -- Sample from the distribution using multinomial
    let sampled ← Tensor.multinomial probs 1 false
    let nextTokenId := (sampled.reshape #[]).itemInt

    tokens := tokens.push nextTokenId

  return tokens


derive_tensor_tree_cast {source target : Device} {n_embd : Nat} :
  BlockParams (device := source) n_embd => BlockParams (device := target) n_embd

derive_tensor_tree_cast {source target : Device} {cfg : Config} :
  GPTParams (device := source) cfg => GPTParams (device := target) cfg

end torch.gpt
