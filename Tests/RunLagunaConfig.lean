/-
  Tests/RunLagunaConfig.lean

  Validates Laguna config.json parsing against the real
  poolside/Laguna-S-2.1-NVFP4 config snapshot in dev/laguna_reference/.
-/
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.ConfigIO

open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

def main : IO Unit := do
  let candidates := #[
    "dev/laguna_reference/config.json",
    "../dev/laguna_reference/config.json"
  ]
  let mut pathOpt : Option String := none
  for p in candidates do
    if ← System.FilePath.pathExists p then
      pathOpt := some p
      break
  let path ← match pathOpt with
    | some p => pure p
    | none => throw (IO.userError "config.json reference not found (run from repo root)")

  let cfg ← Config.loadFromFile path

  check (cfg.vocab_size == 100352) s!"vocab_size={cfg.vocab_size}"
  check (cfg.hidden_size == 3072) s!"hidden_size={cfg.hidden_size}"
  check (cfg.intermediate_size == 12288) s!"intermediate_size={cfg.intermediate_size}"
  check (cfg.num_hidden_layers == 48) s!"num_hidden_layers={cfg.num_hidden_layers}"
  check (cfg.num_attention_heads == 48) s!"num_attention_heads={cfg.num_attention_heads}"
  check (cfg.num_attention_heads_sliding == 72) s!"num_attention_heads_sliding={cfg.num_attention_heads_sliding}"
  check (cfg.num_key_value_heads == 8) s!"num_key_value_heads={cfg.num_key_value_heads}"
  check (cfg.head_dim == 128) s!"head_dim={cfg.head_dim}"
  check (cfg.num_experts == 256) s!"num_experts={cfg.num_experts}"
  check (cfg.num_experts_per_tok == 10) s!"num_experts_per_tok={cfg.num_experts_per_tok}"
  check (cfg.moe_intermediate_size == 1024) s!"moe_intermediate_size={cfg.moe_intermediate_size}"
  check (cfg.shared_expert_intermediate_size == 1024) s!"shared_expert_intermediate_size={cfg.shared_expert_intermediate_size}"
  check cfg.norm_topk_prob "norm_topk_prob=true"
  check (cfg.moe_routed_scaling_factor == 2.5) s!"moe_routed_scaling_factor={cfg.moe_routed_scaling_factor}"
  check (cfg.sliding_window == 512) s!"sliding_window={cfg.sliding_window}"
  check (cfg.mlp_only_layers == #[0]) s!"mlp_only_layers={cfg.mlp_only_layers}"
  check (cfg.eos_token_ids == #[2, 24]) s!"eos_token_ids={cfg.eos_token_ids}"

  -- YaRN (full-attention) rope parameters from the NVFP4 checkpoint.
  check (cfg.rope_theta_full == 500000.0) s!"rope_theta_full={cfg.rope_theta_full}"
  check (cfg.half_rotary_dim_full == 32) s!"half_rotary_dim_full={cfg.half_rotary_dim_full}"
  check (cfg.yarn_factor == 32.0) s!"yarn_factor={cfg.yarn_factor}"
  check (cfg.yarn_original_max_position_embeddings == 8192) s!"yarn_orig={cfg.yarn_original_max_position_embeddings}"
  check (cfg.yarn_beta_fast == 32.0) s!"yarn_beta_fast={cfg.yarn_beta_fast}"
  check (cfg.yarn_beta_slow == 1.0) s!"yarn_beta_slow={cfg.yarn_beta_slow}"
  check ((cfg.yarn_attention_factor - 1.3465735902799727).abs < 1e-9) s!"yarn_attention_factor={cfg.yarn_attention_factor}"
  check (cfg.rope_theta_sliding == 10000.0) s!"rope_theta_sliding={cfg.rope_theta_sliding}"
  check (cfg.half_rotary_dim_sliding == 64) s!"half_rotary_dim_sliding={cfg.half_rotary_dim_sliding}"

  -- Layer schedule: full at i%4==0 (12 layers), sliding elsewhere (36).
  let lts := cfg.layer_types
  check (lts.size == 48) s!"layer_types.size={lts.size}"
  check ((lts.filter (· == .fullAttention)).size == 12) "12 full-attention layers"
  check ((lts.filter (· == .slidingAttention)).size == 36) "36 sliding-attention layers"
  check (cfg.layerType 0 == .fullAttention) "layer 0 is full attention"
  check (cfg.layerType 1 == .slidingAttention) "layer 1 is sliding attention"
  check (cfg.layerType 44 == .fullAttention) "layer 44 is full attention"
  check (cfg.numHeadsForLayer 0 == 48) "layer 0 has 48 heads"
  check (cfg.numHeadsForLayer 1 == 72) "layer 1 has 72 heads"
  check (cfg.isDenseMlpLayer 0) "layer 0 is dense MLP"
  check (!(cfg.isDenseMlpLayer 1)) "layer 1 is MoE"
  check cfg.isMoE "config is MoE"
  check (cfg.rotaryDimFull == 64) s!"rotaryDimFull={cfg.rotaryDimFull}"
  check (cfg.rotaryDimSliding == 128) s!"rotaryDimSliding={cfg.rotaryDimSliding}"
  check (cfg.numHeadsPerKVGroupFull == 6) s!"kv group full={cfg.numHeadsPerKVGroupFull}"
  check (cfg.numHeadsPerKVGroupSliding == 9) s!"kv group sliding={cfg.numHeadsPerKVGroupSliding}"

  IO.println "All Laguna config tests passed."
