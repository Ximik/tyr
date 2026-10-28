import Tyr.Typed
import Tyr.Model.Qwen3
import LeanTest

open torch
open torch.qwen3

private def tinyCfg : Config := {
  vocab_size := 64
  hidden_size := 32
  intermediate_size := 64
  num_hidden_layers := 2
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 8
  rope_theta := 10000.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 128
}

private def flattenIds {batch seq : Nat} (ids : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU }) : IO (Array UInt64) := do
  let flat : Tensor { shape := #[batch * seq], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.toInt64 ids) #[batch * seq]
  Tensor.toUInt64Array flat

private def rowToken (vals : Array UInt64) (seq : Nat) (row col : Nat) : UInt64 :=
  vals.getD (row * seq + col) 0

@[test]
def testQwen3InitAndForward : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : Tensor { shape := #[1, 4], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[1, 2, 3, 4]) #[1, 4]
  let logits := model.forward ids
  LeanTest.assertEqual logits.runtimeShape #[1, 4, tinyCfg.vocab_size]
    "forward should return [batch, seq, vocab]"

@[test]
def testQwen3GreedyCachedUncachedParity : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : Tensor { shape := #[1, 3], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[3, 2, 1]) #[1, 3]

  let ⟨outSeqCached, outCached⟩ ← model.generateGreedy ids 4 #[]
  let ⟨outSeqUncached, outUncached⟩ ← model.generateGreedyUncached ids 4 #[]

  LeanTest.assertEqual outSeqCached (3 + 4)
    "cached generation should append maxNewTokens"
  LeanTest.assertEqual outSeqUncached (3 + 4)
    "uncached generation should append maxNewTokens"

  let cachedFlat : Tensor { shape := #[outSeqCached], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.toInt64 outCached) #[outSeqCached]
  let uncachedFlat : Tensor { shape := #[outSeqUncached], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.toInt64 outUncached) #[outSeqUncached]
  let cachedIds ← Tensor.toUInt64Array cachedFlat
  let uncachedIds ← Tensor.toUInt64Array uncachedFlat

  LeanTest.assertEqual cachedIds uncachedIds
    "cached and uncached greedy decoding should produce identical tokens"

@[test]
def testQwen3ConfigLoad : IO Unit := do
  let path := "/tmp/qwen3_config_test.json"
  let json :=
    "{\"vocab_size\":128,\"hidden_size\":48,\"intermediate_size\":96," ++
    "\"num_hidden_layers\":3,\"num_attention_heads\":6,\"num_key_value_heads\":2," ++
    "\"rope_theta\":500000.0,\"max_position_embeddings\":4096}"
  IO.FS.writeFile path json
  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 128 "vocab_size should load from json"
  LeanTest.assertEqual cfg.hidden_size 48 "hidden_size should load from json"
  LeanTest.assertEqual cfg.num_attention_heads 6 "num_attention_heads should load from json"
  LeanTest.assertEqual cfg.head_dim 8 "head_dim should default to hidden_size / num_attention_heads"

@[test]
def testQwen3ConfigLoadRejectsInvalidInvariants : IO Unit := do
  let path := "/tmp/qwen3_config_invalid_test.json"
  let json :=
    "{\"vocab_size\":128,\"hidden_size\":49,\"intermediate_size\":96," ++
    "\"num_hidden_layers\":3,\"num_attention_heads\":6,\"num_key_value_heads\":2}"
  IO.FS.writeFile path json
  let threw ←
    try
      let _ ← Config.loadFromFile path
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw "config loader should reject invalid hidden/heads divisibility"

@[test]
def testQwen3GreedyUncachedEmptyPromptThrows : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : Tensor { shape := #[1, 0], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[]) #[1, 0]
  let threw ←
    try
      let _ ← model.generateGreedyUncached ids 1 #[]
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw "generateGreedyUncached should reject empty prompt"

@[test]
def testQwen3GreedyZeroNewTokensIdentity : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : Tensor { shape := #[1, 3], dtype := .Int64, device := .CPU } := Tensor.reshape (Tensor.fromInt64Array #[4, 5, 6]) #[1, 3]
  let ⟨seqCached, outCached⟩ ← model.generateGreedy ids 0 #[]
  let ⟨seqUncached, outUncached⟩ ← model.generateGreedyUncached ids 0 #[]
  LeanTest.assertEqual seqCached 3 "cached generation with maxNewTokens=0 should preserve prompt length"
  LeanTest.assertEqual seqUncached 3 "uncached generation with maxNewTokens=0 should preserve prompt length"
  let flatCached : Tensor { shape := #[seqCached], dtype := .Int64, device := .CPU } := Tensor.reshape outCached #[seqCached]
  let flatUncached : Tensor { shape := #[seqUncached], dtype := .Int64, device := .CPU } := Tensor.reshape outUncached #[seqUncached]
  let gotCached ← Tensor.toUInt64Array flatCached
  let gotUncached ← Tensor.toUInt64Array flatUncached
  LeanTest.assertEqual gotCached #[4, 5, 6] "cached generation should return identity sequence when no new tokens requested"
  LeanTest.assertEqual gotUncached #[4, 5, 6] "uncached generation should return identity sequence when no new tokens requested"

@[test]
def testQwen3BatchedEosIsRowAware : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let promptRows : Array (Array Int64) := #[
    #[1, 2, 3],
    #[3, 2, 1],
    #[4, 5, 6],
    #[6, 5, 4],
    #[7, 8, 9],
    #[9, 8, 7]
  ]

  let mut selected : Option (Tensor { shape := #[2, 3], dtype := .Int64, device := .CPU } × UInt64) := none
  for i in [:promptRows.size] do
    if selected.isNone then
      for j in [:promptRows.size] do
        if selected.isNone then
          let row0 := promptRows.getD i #[1, 2, 3]
          let row1 := promptRows.getD j #[3, 2, 1]
          let ids : Tensor { shape := #[2, 3], dtype := .Int64, device := .CPU } := (← IO.ofExcept (Tensor.fromInt64ArrayAs? (row0 ++ row1) #[2, 3]))
          let ⟨seqBase, outBase⟩ ← model.generateGreedy ids 2 #[]
          let baseVals ← flattenIds outBase
          let row0Step1 := rowToken baseVals seqBase 0 3
          let row0Step2 := rowToken baseVals seqBase 0 4
          let row1Step1 := rowToken baseVals seqBase 1 3
          if row0Step1 != row1Step1 && row0Step2 != row0Step1 then
            selected := some (ids, row0Step1)

  match selected with
  | none =>
    LeanTest.assertTrue false
      "failed to find a deterministic prompt pair for batched row-aware EOS validation"
  | some (ids, eosTok) =>
    let ⟨seqCached, outCached⟩ ← model.generateGreedy ids 4 #[eosTok]
    let ⟨seqUncached, outUncached⟩ ← model.generateGreedyUncached ids 4 #[eosTok]
    LeanTest.assertEqual seqCached seqUncached
      "cached and uncached EOS-aware decode should stop at same sequence length"

    let cachedVals ← flattenIds outCached
    let uncachedVals ← flattenIds outUncached
    LeanTest.assertEqual cachedVals uncachedVals
      "cached and uncached EOS-aware decode should match token-by-token"

    let row0First := rowToken cachedVals seqCached 0 3
    let row1First := rowToken cachedVals seqCached 1 3
    LeanTest.assertEqual row0First eosTok
      "row 0 should emit EOS at its finishing step"
    LeanTest.assertTrue (row1First != eosTok)
      "row 1 should still be active when row 0 first reaches EOS"

    for t in [3:seqCached] do
      let tok := rowToken cachedVals seqCached 0 t
      LeanTest.assertEqual tok eosTok
        "once finished, a row should stay on EOS for subsequent decode steps"
