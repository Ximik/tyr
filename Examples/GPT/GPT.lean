/-
  GPT-2 Model Implementation

  A dependently-typed GPT model for Tyr, enabling type-safe transformer training.
  Shapes are tracked at compile time to catch dimension mismatches.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Optim

namespace torch.gpt

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
  ln1_weight : T #[n_embd]
  ln1_bias : T #[n_embd]
  -- Attention: separate Q, K, V projections (avoids tensor split)
  -- linear3d expects weight [out_dim, in_dim], computes x @ W.T
  q_proj : T #[n_embd, n_embd]
  k_proj : T #[n_embd, n_embd]
  v_proj : T #[n_embd, n_embd]
  -- Attention output projection
  c_proj : T #[n_embd, n_embd]
  c_proj_bias : T #[n_embd]
  -- Layer norm 2 (pre-MLP)
  ln2_weight : T #[n_embd]
  ln2_bias : T #[n_embd]
  -- MLP: expand then project back
  -- mlp_fc: n_embd -> 4*n_embd, weight shape [out, in] = [4*n_embd, n_embd]
  mlp_fc : T #[4 * n_embd, n_embd]
  mlp_fc_bias : T #[4 * n_embd]
  -- mlp_proj: 4*n_embd -> n_embd, weight shape [out, in] = [n_embd, 4*n_embd]
  mlp_proj : T #[n_embd, 4 * n_embd]
  mlp_proj_bias : T #[n_embd]
  deriving TensorStruct

/-- Full GPT model parameters -/
structure GPTParams (cfg : Config) where
  -- Embeddings
  wte : T #[cfg.vocab_size, cfg.n_embd]    -- token embeddings
  wpe : T #[cfg.block_size, cfg.n_embd]   -- position embeddings
  -- Transformer blocks
  blocks : Array (BlockParams cfg.n_embd)
  -- Final layer norm
  ln_f_weight : T #[cfg.n_embd]
  ln_f_bias : T #[cfg.n_embd]
  -- Note: lm_head shares weights with wte (weight tying)
  deriving TensorStruct

/-- Helper to create a leaf parameter tensor: detach then set requires_grad -/
def makeLeafParam {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (autograd.detach t) true

/-- Initialize block parameters with scaled random weights -/
def BlockParams.init (n_embd : UInt64) (scale : Float := 0.02) (device : Device := Device.CPU) : IO (BlockParams n_embd) := do
  -- Create scaled random tensors as leaf tensors on target device
  -- randn * scale creates a non-leaf, so we detach and set requires_grad
  let q_proj ← randn #[n_embd, n_embd] false device
  let k_proj ← randn #[n_embd, n_embd] false device
  let v_proj ← randn #[n_embd, n_embd] false device
  let c_proj ← randn #[n_embd, n_embd] false device
  let mlp_fc ← randn #[4*n_embd, n_embd] false device
  let mlp_proj ← randn #[n_embd, 4*n_embd + 1] false device
  return {
    ln1_weight := makeLeafParam (ones #[n_embd] false device)
    ln1_bias := makeLeafParam (zeros #[n_embd] false device)
    q_proj := makeLeafParam (q_proj * scale)
    k_proj := makeLeafParam (k_proj * scale)
    v_proj := makeLeafParam (v_proj * scale)
    c_proj := makeLeafParam (c_proj * scale)
    c_proj_bias := makeLeafParam (zeros #[n_embd] false device)
    ln2_weight := makeLeafParam (ones #[n_embd] false device)
    ln2_bias := makeLeafParam (zeros #[n_embd] false device)
    mlp_fc := makeLeafParam (mlp_fc * scale)
    mlp_fc_bias := makeLeafParam (zeros #[4*n_embd] false device)
    mlp_proj := makeLeafParam (mlp_proj * scale)
    mlp_proj_bias := makeLeafParam (zeros #[n_embd] false device)
  }

/-- Initialize full GPT model -/
def GPTParams.init (cfg : Config) (device : Device := Device.CPU) : IO (GPTParams cfg) := do
  let wte ← randn #[cfg.vocab_size, cfg.n_embd] false device
  let wpe ← randn #[cfg.block_size, cfg.n_embd] false device
  let mut blocks := #[]
  for _ in [:cfg.n_layer.toNat] do
    let block ← BlockParams.init cfg.n_embd 0.02 device
    blocks := blocks.push block
  return {
    wte := makeLeafParam (mul_scalar wte 0.02)
    wpe := makeLeafParam (mul_scalar wpe 0.02)
    blocks := blocks
    ln_f_weight := makeLeafParam (ones #[cfg.n_embd] false device)
    ln_f_bias := makeLeafParam (zeros #[cfg.n_embd] false device)
  }

/-- Forward pass through a single transformer block with multi-head attention and dropout -/
def blockForward {n_embd n_head batch seq : UInt64}
    (params : BlockParams n_embd)
    (x : T #[batch, seq, n_embd])
    (dropout_p : Float := 0.0)
    (training : Bool := true) : IO (T #[batch, seq, n_embd]) := do
  -- Pre-norm attention
  let h := nn.layer_norm x params.ln1_weight params.ln1_bias
  -- Q, K, V projections: [batch, seq, n_embd] @ [n_embd, n_embd] -> [batch, seq, n_embd]
  let q := linear3d h params.q_proj
  let k := linear3d h params.k_proj
  let v := linear3d h params.v_proj
  -- Reshape for multi-head attention:
  -- [batch, seq, n_embd] -> [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  let head_dim := n_embd / n_head
  let q := nn.transpose_for_attention (reshape q #[batch, seq, n_head, head_dim])
  let k := nn.transpose_for_attention (reshape k #[batch, seq, n_head, head_dim])
  let v := nn.transpose_for_attention (reshape v #[batch, seq, n_head, head_dim])
  -- Scaled dot-product attention with causal masking (with dropout built-in)
  -- Input: [batch, n_head, seq, head_dim], Output: [batch, n_head, seq, head_dim]
  let attn := nn.scaled_dot_product_attention q k v dropout_p true
  -- Reshape back: [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim] -> [batch, seq, n_embd]
  let attn := reshape (nn.transpose_from_attention attn) #[batch, seq, n_embd]
  -- Output projection + dropout
  let attn := affine3d attn params.c_proj params.c_proj_bias
  let attn ← nn.dropout attn dropout_p training
  -- Residual connection
  let x := x + attn
  -- Pre-norm MLP
  let h := nn.layer_norm x params.ln2_weight params.ln2_bias
  let h := affine3d h params.mlp_fc params.mlp_fc_bias
  let h := nn.gelu h
  let h := affine3d h params.mlp_proj params.mlp_proj_bias
  -- MLP dropout
  let h ← nn.dropout h dropout_p training
  -- Residual connection
  return x + h

/-- Full GPT forward pass with dropout -/
def forward {cfg : Config} {batch seq : UInt64}
    (params : GPTParams cfg)
    (idx : T #[batch, seq])
    (training : Bool := true) : IO (T #[batch, seq, cfg.vocab_size]) := do
  -- Token embeddings: [batch, seq] -> [batch, seq, n_embd]
  let tok_emb := nn.embedding idx params.wte
  -- Position embeddings (create position indices): [seq] -> [seq, n_embd]
  -- Move positions to the same device as the embedding weights
  let positions := (arange 0 seq 1).to params.wpe.device
  let pos_emb := nn.embedding1d positions params.wpe
  -- Expand pos_emb to match batch: [seq, n_embd] -> [batch, seq, n_embd]
  let pos_emb_expanded := nn.expand pos_emb #[batch, seq, cfg.n_embd]
  -- Combine embeddings + dropout: [batch, seq, n_embd]
  let x : T #[batch, seq, cfg.n_embd] := add tok_emb pos_emb_expanded
  let x ← nn.dropout x cfg.dropout training
  -- Apply transformer blocks with multi-head attention and dropout
  let mut x := x
  for block in params.blocks do
    x ← blockForward (n_head := cfg.n_head) block x cfg.dropout training
  -- Final layer norm
  let xNorm := nn.layer_norm x params.ln_f_weight params.ln_f_bias
  -- Output projection (weight-tied with token embeddings): [batch, seq, n_embd] @ [n_embd, vocab] -> [batch, seq, vocab]
  let wte_t := nn.transpose2d params.wte  -- [vocab, n_embd] -> [n_embd, vocab]
  return nn.matmul3d xNorm wte_t

/-- Compute cross-entropy loss (training=true enables dropout) -/
def loss {cfg : Config} {batch seq : UInt64}
    (params : GPTParams cfg)
    (idx : T #[batch, seq])
    (targets : T #[batch, seq])
    (training : Bool := true) : IO (T #[]) := do
  let logits ← forward params idx training
  -- Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
  let logits_flat := reshape logits #[batch * seq, cfg.vocab_size]
  let targets_flat := reshape targets #[batch * seq]
  return nn.cross_entropy logits_flat targets_flat

/-- Training state using Optax-style optimizer -/
structure TrainState (cfg : Config) where
  params : GPTParams cfg
  optState : Optim.AdamWState (GPTParams cfg)
  step : Nat

/-- Initialize training state -/
def TrainState.init (cfg : Config) : IO (TrainState cfg) := do
  let params ← GPTParams.init cfg
  let opt := Optim.adamw (lr := 1e-3)
  return {
    params := params
    optState := opt.init params
    step := 0
  }

/-- Generate tokens autoregressively from the model
    Takes initial context and generates maxNewTokens more tokens -/
def generate {cfg : Config}
    (params : GPTParams cfg)
    (context : Array Int64)  -- Initial token IDs
    (maxNewTokens : Nat)
    (temperature : Float := 1.0)
    : IO (Array Int64) := do
  let mut tokens := context

  for _ in [:maxNewTokens] do
    -- Crop context to block_size if needed
    let ctxLen := min tokens.size cfg.block_size.toNat
    let startIdx := tokens.size - ctxLen
    let ctx := tokens[startIdx:].toArray

    -- Create input tensor [1, seq] from token array and move to model's device
    let seq : UInt64 := ctxLen.toUInt64
    let inputFlat := torch.data.fromInt64Array ctx
    let inputTensor := (reshape inputFlat #[1, seq]).to params.wte.device

    -- Forward pass to get logits [1, seq, vocab] (training=false disables dropout)
    let logits ← forward params inputTensor false

    -- Get logits for last position: logits[:, -1, :] -> [1, vocab]
    -- Reshape to [1 * seq * vocab] then slice last vocab elements
    let flat := reshape logits #[seq * cfg.vocab_size]
    let lastLogits := torch.data.slice1d' flat ((seq - 1) * cfg.vocab_size).toInt64 (seq * cfg.vocab_size).toInt64

    -- Apply temperature and convert to probabilities
    let scaledLogits := div_scalar lastLogits temperature
    let probs := nn.softmax scaledLogits

    -- Sample from the distribution using multinomial
    let sampled ← nn.multinomial probs 1 false
    let nextTokenId := nn.itemInt sampled

    tokens := tokens.push nextTokenId

  return tokens

end torch.gpt
