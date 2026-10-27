/-
  Checkpoint System for Diffusion Model Training

  Provides save/load functionality for diffusion model parameters and optimizer state.
  Follows the same pattern as Tyr/Checkpoint.lean for GPT models.
-/
import Examples.Diffusion.Diffusion
import Tyr.TensorStruct
import Tyr.Optim

namespace torch.diffusion.checkpoint

open torch
open torch.diffusion
open torch.Optim

/-- Checkpoint metadata for diffusion training -/
structure DiffusionCheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0
  deriving Repr, Inhabited

/-- Save NanoProof-style AttentionParams -/
def saveAttentionParams {n_embd n_head n_kv_head : UInt64}
    (params : AttentionParams n_embd n_head n_kv_head)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/block_{blockIdx}_attn_"
  Tensor.saveTensor params.c_q (pfx ++ "c_q.pt")
  Tensor.saveTensor params.c_k (pfx ++ "c_k.pt")
  Tensor.saveTensor params.c_v (pfx ++ "c_v.pt")
  Tensor.saveTensor params.c_proj (pfx ++ "c_proj.pt")

/-- Save NanoProof-style MLPParams -/
def saveMLPParams {n_embd : UInt64}
    (params : MLPParams n_embd)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/block_{blockIdx}_mlp_"
  Tensor.saveTensor params.c_fc (pfx ++ "c_fc.pt")
  Tensor.saveTensor params.c_proj (pfx ++ "c_proj.pt")

/-- Save NanoProof-style BlockParams (attention + MLP) -/
def saveDiffusionBlockParams {n_embd n_head n_kv_head : UInt64}
    (params : BlockParams n_embd n_head n_kv_head)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  saveAttentionParams params.attn dir blockIdx
  saveMLPParams params.mlp dir blockIdx

/-- Load AttentionParams from directory -/
private def loadAttentionParams (n_embd n_head n_kv_head : UInt64)
    (dir : String) (blockIdx : Nat) : IO (AttentionParams n_embd n_head n_kv_head) := do
  let pfx := s!"{dir}/block_{blockIdx}_attn_"
  let head_dim := n_embd / n_head
  let q_dim := n_head * head_dim
  let kv_dim := n_kv_head * head_dim
  return {
    c_q := ← Tensor.loadTensorExact { device := .CPU, shape := #[q_dim, n_embd], dtype := .Float32 } (pfx ++ "c_q.pt")
    c_k := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "c_k.pt")
    c_v := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "c_v.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, n_embd], dtype := .Float32 } (pfx ++ "c_proj.pt")
  }

/-- Load MLPParams from directory -/
private def loadMLPParams (n_embd : UInt64)
    (dir : String) (blockIdx : Nat) : IO (MLPParams n_embd) := do
  let pfx := s!"{dir}/block_{blockIdx}_mlp_"
  return {
    c_fc := ← Tensor.loadTensorExact { device := .CPU, shape := #[4 * n_embd, n_embd], dtype := .Float32 } (pfx ++ "c_fc.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, 4 * n_embd], dtype := .Float32 } (pfx ++ "c_proj.pt")
  }

/-- Load BlockParams from directory -/
private def loadDiffusionBlockParams (n_embd n_head n_kv_head : UInt64)
    (dir : String) (blockIdx : Nat) : IO (BlockParams n_embd n_head n_kv_head) := do
  let attn ← loadAttentionParams n_embd n_head n_kv_head dir blockIdx
  let mlp ← loadMLPParams n_embd dir blockIdx
  return { attn, mlp }

/-- Save full DiffusionParams to a directory -/
def saveDiffusionParams {cfg : Config} (params : DiffusionParams cfg) (dir : String) : IO Unit := do
  -- Save embeddings
  Tensor.saveTensor params.token_emb (dir ++ "/token_emb.pt")
  Tensor.saveTensor params.time_emb (dir ++ "/time_emb.pt")
  Tensor.saveTensor params.output_head (dir ++ "/output_head.pt")
  -- Save blocks
  let mut idx := 0
  for block in params.blocks do
    saveDiffusionBlockParams block dir idx
    idx := idx + 1
  IO.println s!"Diffusion model saved to {dir}"

/-- Load DiffusionParams from a directory -/
def loadDiffusionParams (cfg : Config) (dir : String) : IO (DiffusionParams cfg) := do
  -- Load embeddings
  let token_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (dir ++ "/token_emb.pt")
  let time_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.diffusion_steps, cfg.n_embd], dtype := .Float32 } (dir ++ "/time_emb.pt")
  let output_head ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (dir ++ "/output_head.pt")
  -- Load blocks
  let mut blocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_head) := #[]
  for i in [:cfg.n_layer.toNat] do
    let block ← loadDiffusionBlockParams cfg.n_embd cfg.n_head cfg.n_head dir i
    blocks := blocks.push block
  IO.println s!"Diffusion model loaded from {dir}"
  -- Apply makeLeafParams
  let model : DiffusionParams cfg := {
    token_emb := token_emb
    time_emb := time_emb
    blocks := blocks
    output_head := output_head
  }
  TensorStruct.makeLeafParams model

/-- Save checkpoint metadata -/
def saveCheckpointMeta (m : DiffusionCheckpointMeta) (path : String) : IO Unit := do
  let content := s!"iteration={m.iteration}\nbestValLoss={m.bestValLoss}\ntrainLoss={m.trainLoss}\noptimCount={m.optimCount}"
  IO.FS.writeFile path content

/-- Load checkpoint metadata -/
def loadCheckpointMeta (path : String) : IO DiffusionCheckpointMeta := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n"
  let mut iteration : Nat := 0
  let mut bestValLoss : Float := 1e10
  let mut trainLoss : Float := 0.0
  let mut optimCount : Nat := 0
  for line in lines do
    if line.startsWith "iteration=" then
      match (line.drop 10).toNat? with
      | some n => iteration := n
      | none => pure ()
    else if line.startsWith "bestValLoss=" then
      match (line.drop 12).toNat? with
      | some n => bestValLoss := n.toFloat
      | none => pure ()
    else if line.startsWith "trainLoss=" then
      match (line.drop 10).toNat? with
      | some n => trainLoss := n.toFloat
      | none => pure ()
    else if line.startsWith "optimCount=" then
      match (line.drop 11).toNat? with
      | some n => optimCount := n
      | none => pure ()
  return { iteration, bestValLoss, trainLoss, optimCount }

/-- Save full checkpoint (model + metadata) -/
def saveCheckpoint {cfg : Config}
    (params : DiffusionParams cfg)
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String) : IO Unit := do
  saveDiffusionParams params dir
  saveCheckpointMeta { iteration, bestValLoss, trainLoss } (dir ++ "/meta.txt")
  IO.println s!"Diffusion checkpoint saved at iteration {iteration}"

/-- Load checkpoint (model + metadata) -/
def loadCheckpoint (cfg : Config) (dir : String)
    : IO (DiffusionParams cfg × DiffusionCheckpointMeta) := do
  let params ← loadDiffusionParams cfg dir
  let m ← loadCheckpointMeta (dir ++ "/meta.txt")
  IO.println s!"Diffusion checkpoint loaded from iteration {m.iteration}"
  return (params, m)

/-- Check if a checkpoint exists -/
def checkpointExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/meta.txt")

/-! ## Optimizer State Checkpointing -/

/-- Helper: Save Adam state for AttentionParams -/
private def saveAttentionAdamState {n_embd n_head n_kv_head : UInt64}
    (mu nu : AttentionParams n_embd n_head n_kv_head)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_attn_"
  Tensor.saveTensor mu.c_q (pfx ++ "mu_c_q.pt")
  Tensor.saveTensor mu.c_k (pfx ++ "mu_c_k.pt")
  Tensor.saveTensor mu.c_v (pfx ++ "mu_c_v.pt")
  Tensor.saveTensor mu.c_proj (pfx ++ "mu_c_proj.pt")
  Tensor.saveTensor nu.c_q (pfx ++ "nu_c_q.pt")
  Tensor.saveTensor nu.c_k (pfx ++ "nu_c_k.pt")
  Tensor.saveTensor nu.c_v (pfx ++ "nu_c_v.pt")
  Tensor.saveTensor nu.c_proj (pfx ++ "nu_c_proj.pt")

/-- Helper: Save Adam state for MLPParams -/
private def saveMLPAdamState {n_embd : UInt64}
    (mu nu : MLPParams n_embd)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_mlp_"
  Tensor.saveTensor mu.c_fc (pfx ++ "mu_c_fc.pt")
  Tensor.saveTensor mu.c_proj (pfx ++ "mu_c_proj.pt")
  Tensor.saveTensor nu.c_fc (pfx ++ "nu_c_fc.pt")
  Tensor.saveTensor nu.c_proj (pfx ++ "nu_c_proj.pt")

/-- Helper: Save Adam state for BlockParams -/
private def saveDiffusionBlockAdamState {n_embd n_head n_kv_head : UInt64}
    (mu nu : BlockParams n_embd n_head n_kv_head)
    (dir : String) (blockIdx : Nat) : IO Unit := do
  saveAttentionAdamState mu.attn nu.attn dir blockIdx
  saveMLPAdamState mu.mlp nu.mlp dir blockIdx

/-- Helper: Load Adam state for AttentionParams -/
private def loadAttentionAdamState (n_embd n_head n_kv_head : UInt64)
    (dir : String) (blockIdx : Nat)
    : IO (AttentionParams n_embd n_head n_kv_head × AttentionParams n_embd n_head n_kv_head) := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_attn_"
  let head_dim := n_embd / n_head
  let q_dim := n_head * head_dim
  let kv_dim := n_kv_head * head_dim
  let mu : AttentionParams n_embd n_head n_kv_head := {
    c_q := ← Tensor.loadTensorExact { device := .CPU, shape := #[q_dim, n_embd], dtype := .Float32 } (pfx ++ "mu_c_q.pt")
    c_k := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "mu_c_k.pt")
    c_v := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "mu_c_v.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, n_embd], dtype := .Float32 } (pfx ++ "mu_c_proj.pt")
  }
  let nu : AttentionParams n_embd n_head n_kv_head := {
    c_q := ← Tensor.loadTensorExact { device := .CPU, shape := #[q_dim, n_embd], dtype := .Float32 } (pfx ++ "nu_c_q.pt")
    c_k := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "nu_c_k.pt")
    c_v := ← Tensor.loadTensorExact { device := .CPU, shape := #[kv_dim, n_embd], dtype := .Float32 } (pfx ++ "nu_c_v.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, n_embd], dtype := .Float32 } (pfx ++ "nu_c_proj.pt")
  }
  return (mu, nu)

/-- Helper: Load Adam state for MLPParams -/
private def loadMLPAdamState (n_embd : UInt64) (dir : String) (blockIdx : Nat)
    : IO (MLPParams n_embd × MLPParams n_embd) := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_mlp_"
  let mu : MLPParams n_embd := {
    c_fc := ← Tensor.loadTensorExact { device := .CPU, shape := #[4 * n_embd, n_embd], dtype := .Float32 } (pfx ++ "mu_c_fc.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, 4 * n_embd], dtype := .Float32 } (pfx ++ "mu_c_proj.pt")
  }
  let nu : MLPParams n_embd := {
    c_fc := ← Tensor.loadTensorExact { device := .CPU, shape := #[4 * n_embd, n_embd], dtype := .Float32 } (pfx ++ "nu_c_fc.pt")
    c_proj := ← Tensor.loadTensorExact { device := .CPU, shape := #[n_embd, 4 * n_embd], dtype := .Float32 } (pfx ++ "nu_c_proj.pt")
  }
  return (mu, nu)

/-- Helper: Load Adam state for BlockParams -/
private def loadDiffusionBlockAdamState (n_embd n_head n_kv_head : UInt64)
    (dir : String) (blockIdx : Nat)
    : IO (BlockParams n_embd n_head n_kv_head × BlockParams n_embd n_head n_kv_head) := do
  let (muAttn, nuAttn) ← loadAttentionAdamState n_embd n_head n_kv_head dir blockIdx
  let (muMLP, nuMLP) ← loadMLPAdamState n_embd dir blockIdx
  return ({ attn := muAttn, mlp := muMLP }, { attn := nuAttn, mlp := nuMLP })

/-- Save AdamW optimizer state for DiffusionParams -/
def saveDiffusionAdamWState {cfg : Config}
    (state : AdamWState (DiffusionParams cfg))
    (dir : String) : IO Unit := do
  let adamState := state.fst
  let pfx := dir ++ "/optim_"
  -- Save count
  IO.FS.writeFile (pfx ++ "count.txt") (toString adamState.count)
  -- Save embedding mu/nu
  Tensor.saveTensor adamState.mu.token_emb (pfx ++ "mu_token_emb.pt")
  Tensor.saveTensor adamState.mu.time_emb (pfx ++ "mu_time_emb.pt")
  Tensor.saveTensor adamState.mu.output_head (pfx ++ "mu_output_head.pt")
  Tensor.saveTensor adamState.nu.token_emb (pfx ++ "nu_token_emb.pt")
  Tensor.saveTensor adamState.nu.time_emb (pfx ++ "nu_time_emb.pt")
  Tensor.saveTensor adamState.nu.output_head (pfx ++ "nu_output_head.pt")
  -- Save block states
  let mut idx := 0
  for (muBlock, nuBlock) in adamState.mu.blocks.zip adamState.nu.blocks do
    saveDiffusionBlockAdamState muBlock nuBlock dir idx
    idx := idx + 1
  IO.println s!"Diffusion optimizer state saved to {dir}"

/-- Load AdamW optimizer state for DiffusionParams -/
def loadDiffusionAdamWState (cfg : Config) (dir : String)
    : IO (AdamWState (DiffusionParams cfg)) := do
  let pfx := dir ++ "/optim_"
  -- Load count
  let countStr ← IO.FS.readFile (pfx ++ "count.txt")
  let count := match countStr.trimAscii.toString.toNat? with
    | some n => n
    | none => 0
  -- Load embedding mu/nu
  let mu_token_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (pfx ++ "mu_token_emb.pt")
  let mu_time_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.diffusion_steps, cfg.n_embd], dtype := .Float32 } (pfx ++ "mu_time_emb.pt")
  let mu_output_head ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (pfx ++ "mu_output_head.pt")
  let nu_token_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (pfx ++ "nu_token_emb.pt")
  let nu_time_emb ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.diffusion_steps, cfg.n_embd], dtype := .Float32 } (pfx ++ "nu_time_emb.pt")
  let nu_output_head ← Tensor.loadTensorExact { device := .CPU, shape := #[cfg.vocab_size, cfg.n_embd], dtype := .Float32 } (pfx ++ "nu_output_head.pt")
  -- Load block states
  let mut muBlocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_head) := #[]
  let mut nuBlocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_head) := #[]
  for i in [:cfg.n_layer.toNat] do
    let (muBlock, nuBlock) ← loadDiffusionBlockAdamState cfg.n_embd cfg.n_head cfg.n_head dir i
    muBlocks := muBlocks.push muBlock
    nuBlocks := nuBlocks.push nuBlock
  let mu : DiffusionParams cfg := {
    token_emb := mu_token_emb
    time_emb := mu_time_emb
    blocks := muBlocks
    output_head := mu_output_head
  }
  let nu : DiffusionParams cfg := {
    token_emb := nu_token_emb
    time_emb := nu_time_emb
    blocks := nuBlocks
    output_head := nu_output_head
  }
  IO.println s!"Diffusion optimizer state loaded from {dir} (count={count})"
  return { fst := { count, mu, nu }, snd := { fst := {}, snd := {} } }

/-- Save full checkpoint with optimizer state -/
def saveFullCheckpoint {cfg : Config}
    (params : DiffusionParams cfg)
    (optState : AdamWState (DiffusionParams cfg))
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String) : IO Unit := do
  saveDiffusionParams params dir
  saveDiffusionAdamWState optState dir
  let ckptMeta : DiffusionCheckpointMeta := {
    iteration, bestValLoss, trainLoss
    optimCount := optState.fst.count
  }
  saveCheckpointMeta ckptMeta (dir ++ "/meta.txt")
  IO.println s!"Full diffusion checkpoint saved at iteration {iteration}"

/-- Load full checkpoint with optimizer state -/
def loadFullCheckpoint (cfg : Config) (dir : String)
    : IO (DiffusionParams cfg × AdamWState (DiffusionParams cfg) × DiffusionCheckpointMeta) := do
  let params ← loadDiffusionParams cfg dir
  let optState ← loadDiffusionAdamWState cfg dir
  let ckptMeta ← loadCheckpointMeta (dir ++ "/meta.txt")
  IO.println s!"Full diffusion checkpoint loaded from iteration {ckptMeta.iteration}"
  return (params, optState, ckptMeta)

/-- Check if optimizer state exists in checkpoint -/
def optimStateExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/optim_count.txt")

end torch.diffusion.checkpoint
