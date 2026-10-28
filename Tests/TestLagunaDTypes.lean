import Tests.Assertions
import Tyr.Model.Laguna.Weights
import LeanTest

open torch torch.laguna

namespace Tests.LagunaDTypes

example {device : Device} {cfg : Config} {batch seq : Nat}
    (model : LagunaForCausalLM (device := device) cfg .BFloat16)
    (ids : Tensor ⟨#[batch, seq], .Int64, device⟩) :
    IO (Tensor ⟨#[batch, seq, cfg.vocab_size], .BFloat16, device⟩) :=
  model.forward cfg ids

private def cfg : Config := { LagunaConfig.laguna_s_2_1 with
  vocab_size := 8, hidden_size := 4, intermediate_size := 8
  num_hidden_layers := 0, num_attention_heads := 1, num_attention_heads_sliding := 1
  num_key_value_heads := 1, head_dim := 4, num_experts := 0 }

/-- One checkpoint per case: native checkpoint metadata is cached by path. -/
private def writeFixture (directory : System.FilePath) (mode : Nat)
    (computeDType : String := "BF16") : IO Unit := do
  let headShape := if mode == 2 then #[cfg.vocab_size, cfg.hidden_size + 1]
    else #[cfg.vocab_size, cfg.hidden_size]
  let entries := #[
    ("model.embed_tokens.weight", #[cfg.vocab_size, cfg.hidden_size], true),
    ("model.norm.weight", #[cfg.hidden_size], false)] ++
    (if mode == 3 then #[] else #[("lm_head.weight", headShape, mode != 1)])
  let mut header : List (String × Lean.Json) := []
  let mut bytes := ByteArray.empty
  for (name, shape, bf16) in entries do
    let start := bytes.size
    for _ in [:shape.foldl (fun n d => n * d) 1] do
      bytes := bytes ++ (if bf16 && computeDType == "BF16" then ByteArray.mk #[0, 62]
        else ByteArray.mk #[0, 0, 0, 62])
    header := header ++ [(name, Lean.Json.mkObj [
      ("dtype", Lean.toJson (if bf16 then computeDType else "F32")),
      ("shape", Lean.toJson shape),
      ("data_offsets", Lean.toJson #[start, bytes.size])])]
  let json := (Lean.Json.mkObj header).compress.toUTF8
  let mut length := ByteArray.empty
  for i in [:8] do
    length := length.push ((json.size >>> (i * 8)).toUInt8)
  IO.FS.writeBinFile (directory / "model.safetensors") (length ++ json ++ bytes)

@[test] def testLagunaLoaderPreservesDTypeRoles : IO Unit := do
  for sharded in #[false, true] do
    for mode in #[0, 3] do
      IO.FS.withTempDir fun directory => do
        writeFixture directory mode
        let ⟨selected, model⟩ ← if sharded then LagunaForCausalLM.loadSharded directory.toString cfg
          else LagunaForCausalLM.load (directory / "model.safetensors").toString cfg
        LeanTest.assertEqual selected.val .BFloat16
        LeanTest.assertEqual model.model.norm.1 .Float32
        LeanTest.assertEqual (← model.model.norm.2.toFloatArray) #[0.125, 0.125, 0.125, 0.125]
        LeanTest.assertEqual model.tieWordEmbeddings (mode == 3)
        let ids := (Tensor.fromInt64Array #[0, 1]).reshapeChecked #[1, 2]
        let logits ← model.forward cfg ids
        LeanTest.assertEqual logits.actualSpec.dtype .BFloat16
        LeanTest.assertTrue ((← logits.toFloatArray).all Float.isFinite)

@[test] def testLagunaLoaderRejectsNonFloatingCompute : IO Unit := do
  for sharded in #[false, true] do
    IO.FS.withTempDir fun directory => do
      writeFixture directory 0 "I32"
      Tests.assertThrows (do
        if sharded then discard <| LagunaForCausalLM.loadSharded directory.toString cfg
        else discard <| LagunaForCausalLM.load (directory / "model.safetensors").toString cfg)
        (some "Expected a standard real floating dtype")

@[test] def testLagunaLoaderRejectsIncompatibleHead : IO Unit := do
  for sharded in #[false, true] do
    for mode in #[1, 2] do
      IO.FS.withTempDir fun directory => do
        writeFixture directory mode
        Tests.assertThrows (do
          if sharded then discard <| LagunaForCausalLM.loadSharded directory.toString cfg
          else discard <| LagunaForCausalLM.load (directory / "model.safetensors").toString cfg)
          (if mode == 1 then some "Expected dtype" else some "shape")

end Tests.LagunaDTypes
