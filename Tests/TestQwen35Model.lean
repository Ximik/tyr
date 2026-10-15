import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Weights
import LeanTest

open torch
open torch.Model
open torch.qwen35

private def floatArrayDynamic (values : Array Float) := (Tensor.fromFloatArray values).forgetDType

private def tinyDenseCfg : Config := {
  vocab_size := 64
  hidden_size := 32
  intermediate_size := 64
  num_hidden_layers := 3
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 8
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 128
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 4
  linear_value_head_dim := 4
  linear_num_key_heads := 2
  linear_num_value_heads := 4
  layer_types := #[.linearAttention, .fullAttention, .linearAttention]
  full_attention_interval := 2
  moe_intermediate_size := 16
  shared_expert_intermediate_size := 16
  num_experts_per_tok := 2
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def tinyMoeCfg : Config := {
  tinyDenseCfg with
  intermediate_size := 0
  moe_intermediate_size := 8
  shared_expert_intermediate_size := 16
  num_experts_per_tok := 2
  num_experts := 4
}

private def tinyBatchedEosCfg : Config := {
  vocab_size := 4
  hidden_size := 4
  intermediate_size := 4
  num_hidden_layers := 0
  num_attention_heads := 1
  num_key_value_heads := 1
  head_dim := 4
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 16
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 1
  linear_key_head_dim := 1
  linear_value_head_dim := 1
  linear_num_key_heads := 1
  linear_num_value_heads := 1
  layer_types := #[]
  full_attention_interval := 1
  moe_intermediate_size := 4
  shared_expert_intermediate_size := 4
  num_experts_per_tok := 1
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def tinyAttentionCfg : Config := {
  vocab_size := 8
  hidden_size := 4
  intermediate_size := 8
  num_hidden_layers := 1
  num_attention_heads := 2
  num_key_value_heads := 2
  head_dim := 2
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 16
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 1
  linear_key_head_dim := 1
  linear_value_head_dim := 1
  linear_num_key_heads := 1
  linear_num_value_heads := 1
  layer_types := #[.fullAttention]
  full_attention_interval := 1
  moe_intermediate_size := 4
  shared_expert_intermediate_size := 4
  num_experts_per_tok := 1
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def approxEq (a b : Float) (tol : Float := 1e-4) : Bool :=
  Float.abs (a - b) <= tol

private def batchedEosDeterministicModel : IO (Qwen35ForCausalLM tinyBatchedEosCfg) := do
  let base ← Qwen35ForCausalLM.init tinyBatchedEosCfg false
  let embedVals : Array Int64 := #[
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]
  let lmHeadVals : Array Int64 := #[
    0, 0, 0, 0,
    0, 1, 1, 1,
    1, 0, 0, 0,
    0, 0, 0, 0
  ]
  let embed : Tensor { shape := #[4, 4] } := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array embedVals) #[4, 4])
  let lmHead : Tensor { shape := #[4, 4] } := floatDynamic (Tensor.reshapeChecked (Tensor.fromInt64Array lmHeadVals) #[4, 4])
  pure {
    base with
    model := { base.model with embed_tokens := embed }
    lmHead := lmHead
  }

private def rowTokenIds {seq : UInt64}
    (tokens : Tensor { shape := #[2, seq], dtype := .Int64 })
    (row : UInt64)
    : IO (Array UInt64) := do
  let row2d : Tensor { shape := #[1, seq], dtype := .Int64 } := typedSlice tokens 0 row 1
  let flat : Tensor { shape := #[seq], dtype := .Int64 } := Tensor.reshapeChecked (id row2d) #[seq]
  Tensor.toUInt64Array flat

@[test]
def testQwen35DenseInitAndForward : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : Tensor { shape := #[1, 5], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array #[1, 2, 3, 4, 5]) #[1, 5]
  let logits := model.forward tinyDenseCfg ids
  LeanTest.assertEqual logits.runtimeShape #[1, 5, tinyDenseCfg.vocab_size]
    "dense forward should return [batch, seq, vocab]"

@[test]
def testQwen35DenseGreedyCachedUncachedParity : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : Tensor { shape := #[1, 4], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array #[3, 1, 2, 4]) #[1, 4]

  let ⟨seqCached, outCached⟩ ← model.generateGreedy tinyDenseCfg ids 4 #[]
  let ⟨seqUncached, outUncached⟩ ←
    model.generateUncached tinyDenseCfg ids 4 .greedy #[]

  LeanTest.assertEqual seqCached (4 + 4)
    "cached greedy should append maxNewTokens"
  LeanTest.assertEqual seqUncached (4 + 4)
    "uncached greedy should append maxNewTokens"

  let cachedFlat : Tensor { shape := #[seqCached], dtype := .Int64 } := Tensor.reshapeChecked (id outCached) #[seqCached]
  let uncachedFlat : Tensor { shape := #[seqUncached], dtype := .Int64 } := Tensor.reshapeChecked (id outUncached) #[seqUncached]
  let cachedIds ← Tensor.toUInt64Array cachedFlat
  let uncachedIds ← Tensor.toUInt64Array uncachedFlat

  LeanTest.assertEqual cachedIds uncachedIds
    "cached and uncached greedy should produce the same tokens"

@[test]
def testQwen35AttentionUsesPerHeadQGateLayout : IO Unit := do
  let cfg := tinyAttentionCfg
  let qProj : Tensor { shape := #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] } :=
    Tensor.reshapeChecked (floatArrayDynamic #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      10.0, 0.0, 0.0, 0.0,
      0.0, 10.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, -10.0, 0.0,
      0.0, 0.0, 0.0, -10.0
    ]) #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size]
  let ident : Tensor { shape := #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size] } :=
    Tensor.reshapeChecked (floatArrayDynamic #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]) #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size]
  let oProj : Tensor { shape := #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] } :=
    Tensor.reshapeChecked (floatArrayDynamic #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]) #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]
  let attn : Qwen35Attention cfg := {
    q_proj := qProj
    k_proj := ident
    v_proj := ident
    o_proj := oProj
    q_norm := ← Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
    k_norm := ← Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
  }
  let x : Tensor { shape := #[1, 1, cfg.hidden_size] } :=
    Tensor.reshapeChecked (floatArrayDynamic #[1.0, 2.0, 3.0, 4.0]) #[1, 1, cfg.hidden_size]
  let (cos, sin) := Tensor.rotaryFreqs 1 (Config.rotaryDim cfg) cfg.rope_theta x.device
  let out : Tensor { shape := #[1, 1, cfg.hidden_size] } := attn.forward cfg x cos sin true
  let vals ← Tensor.toFloatArray (id out)
  LeanTest.assertTrue (approxEq vals[0]! (1.0 / (1.0 + Float.exp (-10.0))) 1e-4)
    "head 0 dim 0 should use the head-local gate rows"
  LeanTest.assertTrue (approxEq vals[1]! (2.0 / (1.0 + Float.exp (-20.0))) 1e-4)
    "head 0 dim 1 should use the head-local gate rows"
  LeanTest.assertTrue (Float.abs vals[2]! < 1e-4)
    "head 1 dim 0 should be suppressed by its negative gate rows"
  LeanTest.assertTrue (Float.abs vals[3]! < 1e-4)
    "head 1 dim 1 should be suppressed by its negative gate rows"

@[test]
def testQwen35RMSNormCheckpointWeightIsZeroCentered : IO Unit := do
  let raw : Tensor { shape := #[2] } := floatArrayDynamic #[0.25, -0.5]
  let norm := Qwen35RMSNorm.fromCheckpointWeight raw
  let x : Tensor { shape := #[1, 2] } := Tensor.reshapeChecked (floatArrayDynamic #[1.0, 1.0]) #[1, 2]
  let out := norm.forward2d x
  let vals ← Tensor.toFloatArray (id out)
  LeanTest.assertTrue (approxEq vals[0]! 1.25 1e-5)
    "checkpoint RMSNorm weight should be preserved directly"
  LeanTest.assertTrue (approxEq vals[1]! 0.5 1e-5)
    "Qwen3.5 RMSNorm checkpoint weights are already zero-centered"

@[test]
def testQwen35DenseGreedyStreamParity : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : Tensor { shape := #[1, 4], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array #[7, 2, 5, 1]) #[1, 4]

  let streamedRef ← IO.mkRef (#[] : Array UInt64)
  let onStep : StreamCallback 1 := fun _ nextTok => do
    let flat : Tensor { shape := #[1], dtype := .Int64 } := Tensor.reshapeChecked (id nextTok) #[1]
    let vals ← Tensor.toUInt64Array flat
    streamedRef.modify (fun xs => xs ++ vals)

  let ⟨seqStream, outStream⟩ ←
    model.generateStream tinyDenseCfg ids onStep 4 .greedy #[]
  let ⟨seqPlain, outPlain⟩ ←
    model.generate tinyDenseCfg ids 4 .greedy #[]

  LeanTest.assertEqual seqStream seqPlain
    "streaming and non-streaming cached generation should have equal output length"

  let streamFlat : Tensor { shape := #[seqStream], dtype := .Int64 } := Tensor.reshapeChecked (id outStream) #[seqStream]
  let plainFlat : Tensor { shape := #[seqPlain], dtype := .Int64 } := Tensor.reshapeChecked (id outPlain) #[seqPlain]
  let streamIds ← Tensor.toUInt64Array streamFlat
  let plainIds ← Tensor.toUInt64Array plainFlat
  LeanTest.assertEqual streamIds plainIds
    "streaming and non-streaming cached generation should produce identical token ids"

  let streamed ← streamedRef.get
  LeanTest.assertEqual streamed.size 4
    "stream callback should be invoked once per generated token"

@[test]
def testQwen35BatchedEosStopsPerRow : IO Unit := do
  let model ← batchedEosDeterministicModel
  let ids : Tensor { shape := #[2, 1], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array #[0, 3]) #[2, 1]
  let eos := #[2]

  let ⟨seqCached, outCached⟩ ← model.generate tinyBatchedEosCfg ids 2 .greedy eos
  let ⟨seqUncached, outUncached⟩ ← model.generateUncached tinyBatchedEosCfg ids 2 .greedy eos

  LeanTest.assertEqual seqCached 3
    "cached generation should append maxNewTokens when at least one row is unfinished"
  LeanTest.assertEqual seqUncached 3
    "uncached generation should append maxNewTokens when at least one row is unfinished"

  let row0Cached ← rowTokenIds outCached 0
  let row1Cached ← rowTokenIds outCached 1
  let row0Uncached ← rowTokenIds outUncached 0
  let row1Uncached ← rowTokenIds outUncached 1

  LeanTest.assertEqual row0Cached #[0, 2, 2]
    "cached path should keep finished rows at EOS while other rows continue"
  LeanTest.assertEqual row1Cached #[3, 1, 1]
    "cached path should continue decoding unfinished rows"
  LeanTest.assertEqual row0Uncached #[0, 2, 2]
    "uncached path should keep finished rows at EOS while other rows continue"
  LeanTest.assertEqual row1Uncached #[3, 1, 1]
    "uncached path should continue decoding unfinished rows"

  LeanTest.assertEqual row0Cached row0Uncached
    "cached and uncached row 0 outputs should match in batched EOS handling"
  LeanTest.assertEqual row1Cached row1Uncached
    "cached and uncached row 1 outputs should match in batched EOS handling"

@[test]
def testQwen35MoeInitAndForward : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyMoeCfg
  let ids : Tensor { shape := #[1, 3], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array #[5, 6, 7]) #[1, 3]
  let logits := model.forward tinyMoeCfg ids
  LeanTest.assertEqual logits.runtimeShape #[1, 3, tinyMoeCfg.vocab_size]
    "moe forward should return [batch, seq, vocab]"

@[test]
def testQwen35BFloat16ExpertsFloat32Routing : IO Unit := do
  let cfg : Config := { tinyMoeCfg with
    hidden_size := 2
    num_experts := 2
    num_experts_per_tok := 2
    moe_intermediate_size := 1
  }
  -- Both experts compute SiLU(1). Float32 routing weights sum to one,
  -- so weighting must preserve that value rather than sum unweighted experts.
  let experts : Qwen35MoeExperts cfg := {
    gate_up_proj := (Tensor.full #[2, 2, 2] 0.5).toBFloat16.forgetDType
    down_proj := (Tensor.ones #[2, 2, 1]).toBFloat16.forgetDType
  }
  let hidden := (Tensor.ones #[1, 2]).toBFloat16.forgetDType
  let weights := (Tensor.fromFloatArray #[0.25, 0.75]).reshapeChecked #[1, 2] |>.forgetDType
  let indices := (Tensor.fromInt64Array #[0, 1]).reshapeChecked #[1, 2]
  let actual := experts.forward2d cfg hidden weights indices
  LeanTest.assertEqual actual.actualSpec.dtype .BFloat16
  for value in (← actual.toFloatArray) do
    LeanTest.assertTrue (approxEq value 0.73046875 0.005)
      s!"Float32 routing weights must scale BF16 expert outputs: got {value}"

@[test]
def testQwen35BFloat16IntegerPaddingMask : IO Unit := do
  let initialized ← Qwen35GatedDeltaNet.init tinyDenseCfg
  let net ← TensorStruct.castDType .BFloat16 initialized
  let hidden := (Tensor.ones #[2, 2, tinyDenseCfg.hidden_size]).toBFloat16
  let mask := (Tensor.fromInt64Array #[1, 0, 0, 1]).reshapeChecked #[2, 2]
  let expanded := Tensor.expand (mask.reshapeChecked #[2, 2, 1]) #[2, 2, tinyDenseCfg.hidden_size]
  let masked := Tensor.mul hidden expanded
  let (expected, _, _) := net.forward tinyDenseCfg masked.forgetDType
  let (actual, _, _) := net.forward tinyDenseCfg hidden.forgetDType (some mask)
  LeanTest.assertEqual actual.actualSpec.dtype .BFloat16
  let expectedValues ← expected.toFloatArray
  let actualValues ← actual.toFloatArray
  LeanTest.assertEqual actualValues.size expectedValues.size
  for i in [:actualValues.size] do
    LeanTest.assertTrue (approxEq actualValues[i]! expectedValues[i]! 1e-6)
      "integer padding masks must use native promotion without widening BF16 activations"

@[test]
def testQwen35ConfigLoad : IO Unit := do
  let path := "/tmp/qwen35_config_test.json"
  let json :=
    "{\"vocab_size\":128,\"hidden_size\":48,\"intermediate_size\":96," ++
    "\"num_hidden_layers\":4,\"num_attention_heads\":6,\"num_key_value_heads\":2," ++
    "\"head_dim\":8,\"linear_num_key_heads\":3,\"linear_num_value_heads\":6," ++
    "\"linear_key_head_dim\":4,\"linear_value_head_dim\":4," ++
    "\"attn_output_gate\":true,\"mamba_ssm_dtype\":\"float32\"," ++
    "\"mlp_only_layers\":[1,3]," ++
    "\"moe_intermediate_size\":12,\"shared_expert_intermediate_size\":24," ++
    "\"num_experts\":8,\"num_experts_per_tok\":2," ++
    "\"layer_types\":[\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\"]," ++
    "\"rope_parameters\":{\"rope_theta\":500000.0,\"partial_rotary_factor\":0.5," ++
    "\"mrope_interleaved\":true,\"mrope_section\":[2,2,2]}," ++
    "\"mtp_num_hidden_layers\":1,\"mtp_use_dedicated_embeddings\":false}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 128 "vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 48 "hidden_size should load"
  LeanTest.assertEqual cfg.num_experts 8 "num_experts should load"
  LeanTest.assertEqual cfg.layer_types.size 4 "layer_types should parse"
  LeanTest.assertEqual cfg.rope_theta 500000.0 "rope theta should load from rope_parameters"
  LeanTest.assertEqual cfg.attn_output_gate true "attention output gate flag should load"
  LeanTest.assertEqual cfg.mamba_ssm_dtype "float32" "mamba ssm dtype should load"
  LeanTest.assertEqual cfg.mlp_only_layers #[1, 3] "mlp-only layers should load"
  LeanTest.assertEqual cfg.mrope_interleaved true "mrope interleaving flag should load"
  LeanTest.assertEqual cfg.mrope_section #[2, 2, 2] "mrope section should load"
  LeanTest.assertEqual cfg.mtp_num_hidden_layers 1 "mtp hidden layer count should load"
  LeanTest.assertEqual cfg.mtp_use_dedicated_embeddings false "mtp embedding flag should load"

@[test]
def testQwen35ConfigLoadNestedTextConfig : IO Unit := do
  let path := "/tmp/qwen35_config_test_nested.json"
  let json :=
    "{\"tie_word_embeddings\":true,\"text_config\":{" ++
    "\"vocab_size\":32000,\"hidden_size\":64,\"intermediate_size\":128," ++
    "\"num_hidden_layers\":6,\"num_attention_heads\":8,\"num_key_value_heads\":2," ++
    "\"head_dim\":8,\"linear_num_key_heads\":4,\"linear_num_value_heads\":8," ++
    "\"linear_key_head_dim\":4,\"linear_value_head_dim\":4," ++
    "\"moe_intermediate_size\":16,\"shared_expert_intermediate_size\":32," ++
    "\"num_experts\":16,\"num_experts_per_tok\":4," ++
    "\"attn_output_gate\":true," ++
    "\"eos_token_id\":151645," ++
    "\"rope_parameters\":{\"rope_theta\":10000000.0,\"partial_rotary_factor\":0.5," ++
    "\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]}," ++
    "\"mtp_num_hidden_layers\":1,\"mtp_use_dedicated_embeddings\":false}}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 32000 "nested text_config vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 64 "nested text_config hidden_size should load"
  LeanTest.assertEqual cfg.tie_word_embeddings true "tie_word_embeddings should load from top level"
  LeanTest.assertEqual cfg.num_experts 16 "nested text_config num_experts should load"
  LeanTest.assertEqual cfg.eos_token_id (some 151645) "nested text_config eos token should load"
  LeanTest.assertEqual cfg.attn_output_gate true "nested text_config attention gate should load"
  LeanTest.assertEqual cfg.mrope_interleaved true "nested mrope interleaving flag should load"
  LeanTest.assertEqual cfg.mrope_section #[11, 11, 10] "nested mrope section should load"
  LeanTest.assertEqual cfg.mtp_num_hidden_layers 1 "nested mtp hidden layer count should load"
  LeanTest.assertEqual cfg.mtp_use_dedicated_embeddings false "nested mtp embedding flag should load"

@[test] def testQwen35BFloat16HybridCache : IO Unit := do
  let initialized ← Qwen35ForCausalLM.init tinyDenseCfg false
  let model ← TensorStruct.castDType .BFloat16 initialized
  let ids := (Tensor.fromInt64Array #[1, 2, 3, 4]).reshapeChecked #[1, 4]
  let embeds := model.embedTokens ids
  let (cos, sin) := Tensor.rotaryFreqs 4 (Config.rotaryDim tinyDenseCfg) tinyDenseCfg.rope_theta
  let cache := Qwen35Model.initCache tinyDenseCfg model.model 8 .CPU
  let (hidden, cache) ← Qwen35Model.forwardWithCache tinyDenseCfg model.model embeds cos sin cache
  LeanTest.assertEqual hidden.actualSpec.dtype DType.BFloat16
  LeanTest.assertTrue ((← hidden.toFloatArray).all Float.isFinite)
  for state in cache.recurrentStates do
    if let some tensor := state then
      LeanTest.assertEqual tensor.actualSpec.dtype DType.Float32
  let ⟨_, cached⟩ ← model.generate tinyDenseCfg ids 3 .greedy #[]
  let ⟨_, uncached⟩ ← model.generateUncached tinyDenseCfg ids 3 .greedy #[]
  LeanTest.assertEqual (← cached.toUInt64Array) (← uncached.toUInt64Array)

private def qwen35CheckpointEntries (cfg : Config) (optionalWeights : Bool) : Array (String × Shape) := Id.run do
  let mut entries := #[ ("model.embed_tokens.weight", #[cfg.vocab_size, cfg.hidden_size]),
    ("model.norm.weight", #[cfg.hidden_size]) ]
  if optionalWeights then
    entries := entries.push ("lm_head.weight", #[cfg.vocab_size, cfg.hidden_size])
  for i in [:cfg.num_hidden_layers.toNat] do
    let p := s!"model.layers.{i}."
    let mut localEntries := #[ ("input_layernorm.weight", #[cfg.hidden_size]),
      ("post_attention_layernorm.weight", #[cfg.hidden_size]) ]
    match (Config.normalizedLayerTypes cfg).getD i .linearAttention with
    | .fullAttention =>
      localEntries := localEntries ++ #[
        ("self_attn.q_proj.weight", #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size]),
        ("self_attn.k_proj.weight", #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]),
        ("self_attn.v_proj.weight", #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]),
        ("self_attn.o_proj.weight", #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]),
        ("self_attn.q_norm.weight", #[cfg.head_dim]), ("self_attn.k_norm.weight", #[cfg.head_dim]) ]
    | .linearAttention =>
      localEntries := localEntries ++ #[
        ("linear_attn.in_proj_qkv.weight", #[Config.linearConvDim cfg, cfg.hidden_size]),
        ("linear_attn.in_proj_z.weight", #[Config.linearValueDim cfg, cfg.hidden_size]),
        ("linear_attn.in_proj_b.weight", #[cfg.linear_num_value_heads, cfg.hidden_size]),
        ("linear_attn.in_proj_a.weight", #[cfg.linear_num_value_heads, cfg.hidden_size]),
        ("linear_attn.conv1d.weight", #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim]),
        ("linear_attn.dt_bias", #[cfg.linear_num_value_heads]),
        ("linear_attn.A_log", #[cfg.linear_num_value_heads]),
        ("linear_attn.norm.weight", #[cfg.linear_value_head_dim]),
        ("linear_attn.out_proj.weight", #[cfg.hidden_size, Config.linearValueDim cfg]) ]
      if optionalWeights then
        localEntries := localEntries.push ("linear_attn.conv1d.bias", #[Config.linearConvDim cfg])
    if Config.isMoE cfg then
      localEntries := localEntries ++ #[
        ("mlp.gate.weight", #[cfg.num_experts, cfg.hidden_size]),
        ("mlp.experts.gate_up_proj", #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]),
        ("mlp.experts.down_proj", #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]),
        ("mlp.shared_expert.gate_proj.weight", #[cfg.shared_expert_intermediate_size, cfg.hidden_size]),
        ("mlp.shared_expert.up_proj.weight", #[cfg.shared_expert_intermediate_size, cfg.hidden_size]),
        ("mlp.shared_expert.down_proj.weight", #[cfg.hidden_size, cfg.shared_expert_intermediate_size]),
        ("mlp.shared_expert_gate.weight", #[1, cfg.hidden_size]) ]
    else
      localEntries := localEntries ++ #[
        ("mlp.gate_proj.weight", #[cfg.intermediate_size, cfg.hidden_size]),
        ("mlp.up_proj.weight", #[cfg.intermediate_size, cfg.hidden_size]),
        ("mlp.down_proj.weight", #[cfg.hidden_size, cfg.intermediate_size]) ]
    entries := entries ++ localEntries.map fun (name, shape) => (p ++ name, shape)
  return entries

private def writeQwen35Fixture (dir : System.FilePath) (cfg : Config)
    (bf16 sharded optionalWeights : Bool) : IO Unit := do
  let mut header : List (String × Lean.Json) := []
  let mut index : List (String × Lean.Json) := []
  let mut bytes := ByteArray.empty
  for (name, shape) in qwen35CheckpointEntries cfg optionalWeights do
    let start := bytes.size
    for _ in [:shape.foldl (fun n d => n * d.toNat) 1] do
      -- Exactly representable 1/8 in both storage dtypes.
      bytes := bytes ++ (if bf16 then ByteArray.mk #[0, 62] else ByteArray.mk #[0, 0, 0, 62])
    header := header ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if bf16 then "BF16" else "F32")),
      ("shape", Lean.toJson (shape.map UInt64.toNat)),
      ("data_offsets", Lean.toJson #[start, bytes.size])])]
    index := index ++ [(name, Lean.toJson "model.safetensors")]
  let json := (Lean.Json.mkObj header).compress.toUTF8
  let mut length := ByteArray.empty
  for i in [:8] do
    length := length.push ((json.size.toUInt64 >>> (i * 8).toUInt64).toUInt8)
  IO.FS.writeBinFile (dir / "model.safetensors") (length ++ json ++ bytes)
  if sharded then
    IO.FS.writeFile (dir / "model.safetensors.index.json")
      (Lean.Json.mkObj [("weight_map", Lean.Json.mkObj index)]).compress

@[test] def testQwen35TypedCheckpointLoading : IO Unit := do
  for cfg in #[tinyDenseCfg, tinyMoeCfg] do
    for bf16 in #[false, true] do
      for sharded in #[false, true] do
        for optionalWeights in #[false, true] do
          IO.FS.withTempDir fun dir => do
            writeQwen35Fixture dir cfg bf16 sharded optionalWeights
            let model ← if sharded then Qwen35ForCausalLM.loadSharded dir.toString cfg
              else Qwen35ForCausalLM.load (dir / "model.safetensors").toString cfg
            let model := TensorStruct.detach model
            let dtype := if bf16 then DType.BFloat16 else .Float32
            LeanTest.assertEqual model.model.embed_tokens.actualSpec.dtype dtype
            LeanTest.assertEqual model.lmHead.actualSpec.dtype dtype
            LeanTest.assertEqual model.tieWordEmbeddings (!optionalWeights)
            for layer in model.model.layers do
              if let some linear := layer.linear_attn then
                LeanTest.assertEqual linear.conv1d_bias.actualSpec.dtype dtype
            let ids := (Tensor.fromInt64Array #[1, 2, 3]).reshapeChecked #[1, 3]
            let logits := model.forward cfg ids
            LeanTest.assertEqual logits.actualSpec.dtype dtype
            LeanTest.assertTrue ((← logits.toFloatArray).all Float.isFinite)
            let ⟨length, tokens⟩ ← model.generate cfg ids 2 .greedy #[]
            LeanTest.assertEqual length 5
            LeanTest.assertEqual tokens.actualSpec.dtype DType.Int64
