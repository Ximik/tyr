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
  vocab_size : UInt64
  block_size : UInt64  -- max sequence length
  n_embd : UInt64      -- embedding dimension
  n_head : UInt64      -- number of attention heads
  n_layer : UInt64     -- number of transformer blocks
  dropout : Float := 0.0
  deriving Repr, Inhabited

/-- GPT-2 small configuration (124M parameters) -/
def Config.gpt2_small : Config :=
  { vocab_size := 50257, block_size := 1024, n_embd := 768, n_head := 12, n_layer := 12 }

/-- GPT-2 mini for testing (smaller) -/
def Config.gpt2_mini : Config :=
  { vocab_size := 50257, block_size := 256, n_embd := 384, n_head := 6, n_layer := 6 }

/-- GPT-2 micro for quick iteration -/
def Config.gpt2_micro : Config :=
  { vocab_size := 50257, block_size := 128, n_embd := 128, n_head := 4, n_layer := 4 }

/-- Tiny config for quick testing on Shakespeare (65 char vocab) -/
def Config.tiny_shakespeare : Config :=
  { vocab_size := 65, block_size := 64, n_embd := 64, n_head := 4, n_layer := 2, dropout := 0.0 }

/-- nanoGPT CPU config for Shakespeare: 4 layers, 4 heads, 128 embd, no dropout -/
def Config.nanogpt_cpu_shakespeare : Config :=
  { vocab_size := 65, block_size := 64, n_embd := 128, n_head := 4, n_layer := 4, dropout := 0.0 }

/-- nanoGPT GPU config for Shakespeare: 6 layers, 6 heads, 384 embd, dropout 0.2 -/
def Config.nanogpt_gpu_shakespeare : Config :=
  { vocab_size := 65, block_size := 256, n_embd := 384, n_head := 6, n_layer := 6, dropout := 0.2 }

/-- Parameters for a single transformer block -/
structure BlockParams (n_embd : UInt64) where
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
  blocks : Array (BlockParams (device := device) cfg.n_embd)
  -- Final layer norm
  ln_f_weight : Activation (device := device) #[cfg.n_embd]
  ln_f_bias : Activation (device := device) #[cfg.n_embd]
  -- Note: lm_head shares weights with wte (weight tying)
  deriving TensorStruct

/-- Helper to create a leaf parameter tensor: detach then set requires_grad -/
def makeLeafParam {s : Shape} (t : Activation (device := device) s) : IO (Activation (device := device) s) :=
  TensorOps.leaf t

/-- Initialize block parameters with scaled random weights -/
def BlockParams.init (n_embd : UInt64) (scale : Float := 0.02) (device : Device := Device.CPU) : IO (BlockParams (device := device) n_embd) := do
  -- Create scaled random tensors as leaf tensors on target device
  -- randn * scale creates a non-leaf, so we detach and set requires_grad
  let q_proj ← Tensor.randn #[n_embd, n_embd] false device
  let k_proj ← Tensor.randn #[n_embd, n_embd] false device
  let v_proj ← Tensor.randn #[n_embd, n_embd] false device
  let c_proj ← Tensor.randn #[n_embd, n_embd] false device
  let mlp_fc ← Tensor.randn #[4*n_embd, n_embd] false device
  let mlp_proj ← Tensor.randn #[n_embd, 4*n_embd] false device
  return {
    ln1_weight := ← makeLeafParam (Tensor.ones #[n_embd] false device)
    ln1_bias := ← makeLeafParam (Tensor.zeros #[n_embd] false device)
    q_proj := ← makeLeafParam (q_proj * scale)
    k_proj := ← makeLeafParam (k_proj * scale)
    v_proj := ← makeLeafParam (v_proj * scale)
    c_proj := ← makeLeafParam (c_proj * scale)
    c_proj_bias := ← makeLeafParam (Tensor.zeros #[n_embd] false device)
    ln2_weight := ← makeLeafParam (Tensor.ones #[n_embd] false device)
    ln2_bias := ← makeLeafParam (Tensor.zeros #[n_embd] false device)
    mlp_fc := ← makeLeafParam (mlp_fc * scale)
    mlp_fc_bias := ← makeLeafParam (Tensor.zeros #[4*n_embd] false device)
    mlp_proj := ← makeLeafParam (mlp_proj * scale)
    mlp_proj_bias := ← makeLeafParam (Tensor.zeros #[n_embd] false device)
  }

/-- Initialize full GPT model -/
def GPTParams.init (cfg : Config) (device : Device := Device.CPU) : IO (GPTParams (device := device) cfg) := do
  let wte ← Tensor.randn #[cfg.vocab_size, cfg.n_embd] false device
  let wpe ← Tensor.randn #[cfg.block_size, cfg.n_embd] false device
  let mut blocks := #[]
  for _ in [:cfg.n_layer.toNat] do
    let block ← BlockParams.init cfg.n_embd 0.02 device
    blocks := blocks.push block
  return {
    wte := ← makeLeafParam (Tensor.mulScalar wte 0.02)
    wpe := ← makeLeafParam (Tensor.mulScalar wpe 0.02)
    blocks := blocks
    ln_f_weight := ← makeLeafParam (Tensor.ones #[cfg.n_embd] false device)
    ln_f_bias := ← makeLeafParam (Tensor.zeros #[cfg.n_embd] false device)
  }

/-- Forward pass through a single transformer block with multi-head attention and dropout -/
def blockForward {n_embd n_head batch seq : UInt64}
    (params : BlockParams (device := device) n_embd)
    (x : Activation (device := device) #[batch, seq, n_embd])
    (dropout_p : Float := 0.0)
    (training : Bool := true) : IO (Activation (device := device) #[batch, seq, n_embd]) := do
  -- Pre-norm attention
  let h := TensorOps.layerNorm x params.ln1_weight params.ln1_bias
  -- Q, K, V projections: [batch, seq, n_embd] @ [n_embd, n_embd] -> [batch, seq, n_embd]
  let q := Tensor.linear3d h params.q_proj
  let k := Tensor.linear3d h params.k_proj
  let v := Tensor.linear3d h params.v_proj
  -- Reshape for multi-head attention:
  -- [batch, seq, n_embd] -> [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  if n_head == 0 || n_embd % n_head != 0 then
    throw (IO.userError "GPT embedding dimension must be divisible by a nonzero head count")
  let head_dim := n_embd / n_head
  let q ← TensorOps.reshape q #[batch, seq, n_head, head_dim]
  let q := TensorOps.splitHeads q
  let k ← TensorOps.reshape k #[batch, seq, n_head, head_dim]
  let k := TensorOps.splitHeads k
  let v ← TensorOps.reshape v #[batch, seq, n_head, head_dim]
  let v := TensorOps.splitHeads v
  -- Scaled dot-product attention with causal masking (with dropout built-in)
  -- Input: [batch, n_head, seq, head_dim], Output: [batch, n_head, seq, head_dim]
  let attn := TensorOps.attention q k v dropout_p
  -- Reshape back: [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim] -> [batch, seq, n_embd]
  let attn ← TensorOps.reshape (TensorOps.joinHeads attn) #[batch, seq, n_embd]
  -- Output projection + dropout
  let attn := Tensor.affine3d attn params.c_proj params.c_proj_bias
  let attn ← TensorOps.dropout attn dropout_p training
  -- Residual connection
  let x := x + attn
  -- Pre-norm MLP
  let h := TensorOps.layerNorm x params.ln2_weight params.ln2_bias
  let h := Tensor.affine3d h params.mlp_fc params.mlp_fc_bias
  let h := Tensor.gelu h (by rfl)
  let h := Tensor.affine3d h params.mlp_proj params.mlp_proj_bias
  -- MLP dropout
  let h ← TensorOps.dropout h dropout_p training
  -- Residual connection
  return x + h

/-- Full GPT forward pass with dropout -/
def forward {cfg : Config} {batch seq : UInt64}
    (params : GPTParams (device := device) cfg)
    (idx : Tokens (device := device) #[batch, seq])
    (training : Bool := true) : IO (Activation (device := device) #[batch, seq, cfg.vocab_size]) := do
  -- Token embeddings: [batch, seq] -> [batch, seq, n_embd]
  let tok_emb := TensorOps.embedding idx params.wte
  -- Position embeddings (create position indices): [seq] -> [seq, n_embd]
  -- Move positions to the same device as the embedding weights
  let positions := (Tensor.arange seq).to params.wpe.device
  let pos_emb := TensorOps.embedding1d positions params.wpe
  -- Expand pos_emb to match batch: [seq, n_embd] -> [batch, seq, n_embd]
  let pos_emb_expanded := Tensor.expand pos_emb #[batch, seq, cfg.n_embd]
  -- Combine embeddings + dropout: [batch, seq, n_embd]
  let x : Activation (device := device) #[batch, seq, cfg.n_embd] := Tensor.add tok_emb pos_emb_expanded
  let x ← TensorOps.dropout x cfg.dropout training
  -- Apply transformer blocks with multi-head attention and dropout
  let mut x := x
  for block in params.blocks do
    x ← blockForward (n_head := cfg.n_head) block x cfg.dropout training
  -- Final layer norm
  let xNorm := TensorOps.layerNorm x params.ln_f_weight params.ln_f_bias
  -- Output projection (weight-tied with token embeddings): [batch, seq, n_embd] @ [n_embd, vocab] -> [batch, seq, vocab]
  let wte_t := Tensor.transpose2d params.wte  -- [vocab, n_embd] -> [n_embd, vocab]
  return Tensor.matmul3d xNorm wte_t

/-- Compute cross-entropy loss (training=true enables dropout) -/
def loss {cfg : Config} {batch seq : UInt64}
    (params : GPTParams (device := device) cfg)
    (idx : Tokens (device := device) #[batch, seq])
    (targets : Tokens (device := device) #[batch, seq])
    (training : Bool := true) : IO (Activation (device := device) #[]) := do
  let logits ← forward params idx training
  -- Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
  let logits_flat ← TensorOps.reshape logits #[batch * seq, cfg.vocab_size]
  let targets_flat ← TensorOps.reshape targets #[batch * seq]
  IO.ofExcept (Tensor.crossEntropy? logits_flat targets_flat)

/-- Training state using Optax-style optimizer -/
structure TrainState (cfg : Config) where
  params : GPTParams (device := device) cfg
  optState : Optim.AdamWState (GPTParams (device := device) cfg)
  step : Nat

/-- Initialize training state -/
def TrainState.init (cfg : Config) (device : Device := .CPU) : IO (TrainState (device := device) cfg) := do
  let params ← GPTParams.init cfg device
  let opt := Optim.adamw (lr := 1e-3)
  return {
    params := params
    optState := opt.init params
    step := 0
  }

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
    let ctxLen := min tokens.size cfg.block_size.toNat
    let startIdx := tokens.size - ctxLen
    let ctx := tokens[startIdx:].toArray

    -- Create input tensor [1, seq] from token array and move to model's device
    let seq : UInt64 := ctxLen.toUInt64
    let inputTensor := ((Tensor.fromInt64Array ctx).reshapeChecked #[1, seq]).to params.wte.device

    -- Forward pass to get logits [1, seq, vocab] (training=false disables dropout)
    let logits ← forward params inputTensor false

    -- Get logits for last position: logits[:, -1, :] -> [1, vocab]
    -- Reshape to [1 * seq * vocab] then slice last vocab elements
    let flat ← TensorOps.reshape logits #[seq * cfg.vocab_size]
    let lastLogits := (flat.sliceChecked 0 ((seq - 1) * cfg.vocab_size) cfg.vocab_size #[cfg.vocab_size]).reshapeChecked #[1, cfg.vocab_size]

    -- Apply temperature and convert to probabilities
    let scaledLogits := lastLogits / temperature
    let probs := Tensor.softmax scaledLogits

    -- Sample from the distribution using multinomial
    let sampled ← Tensor.multinomial probs 1 false
    let nextTokenId := (sampled.reshapeChecked #[]).itemInt

    tokens := tokens.push nextTokenId

  return tokens


derive_tensor_tree_cast {source target : Device} {n_embd : UInt64} :
  BlockParams (device := source) n_embd => BlockParams (device := target) n_embd

derive_tensor_tree_cast {source target : Device} {cfg : Config} :
  GPTParams (device := source) cfg => GPTParams (device := target) cfg

end torch.gpt
