/-
  Tests/RunLagunaParity.lean

  End-to-end numerical parity of the tyr Laguna implementation against the
  HuggingFace reference (`dev/laguna_reference/modeling_laguna.py`) on the
  tiny BF16 fixture in `Tests/fixtures/laguna/tiny/` (generator:
  `scripts/laguna/make_tiny_fixture.py`; references computed in fp32 on CPU
  with eager attention from the BF16-rounded checkpoint weights).

  Gates (CPU first; repeated on CUDA when available):
  1. `Config.loadFromPretrainedDir` on the fixture dir yields the expected
     tiny config (4 layers, heads 4 full / 6 sliding, kv 2, 8 experts,
     window 8, dense layer 0).
  2. `LagunaForCausalLM.load` on the fixture safetensors (dense-fused expert
     layout). The loaded BF16 weights are then upcast and all parity gates run
     in FP32: the reference tie margins (min top-5/top-6 logit gap ≈ 3e-4) are
     far below bf16 rounding, so no bf16 engine can meet the top-5-set and
     exact-gen gates — HF's own bf16 run flips top-5 sets on logits_b and
     diverges from gen_ids at step 4 (measured; an informational BF16 pass at
     the end of the fixture run reports tyr's bf16 diffs without gating).
  3. forward logits for inputs A (7 tokens) and B (20 tokens) vs the fp32
     references: max abs diff ≤ 0.05 AND top-5 id sets match at every
     position. Hidden-state diagnostics (`hidden_a_l0`, `hidden_a_final`)
     localize layer-0-vs-rest divergence when a gate fails.
  4. 16 greedy generate steps from A reproduce the reference `gen_ids`
     EXACTLY (budget termination, EOS disabled).
  5. Logits producing the final generated token match `gen_logits_last`.

  Real-checkpoint smoke mode: `LagunaParityTest <modelDir> [numLayers] [cpu|cuda]`
  loads the config and the first `numLayers` (default 2) layers of a sharded
  NVFP4 checkpoint through the packed-expert path, verifies bank shapes, and
  runs a forward (truncated model ⇒ numerics are meaningless; this proves
  the loading path and reports timing).

  Build: lake build Tests.RunLagunaParity
  Run:   lake exe LagunaParityTest
-/
import Tyr.Torch
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.ConfigIO
import Tyr.Model.Laguna.Rope
import Tyr.Model.Laguna.Model
import Tyr.Model.Laguna.Weights
import Tyr.Model.Laguna.Pretrained

open torch
open torch.Model
open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

private def deviceLabel : Device → String
  | .CPU => "cpu"
  | .CUDA i => s!"cuda:{i}"
  | .MPS => "mps"

/-- Deterministic int64 token-id tensor `[1, seq]` on `device`. -/
private def mkIds (vals : Array Int64) (device : Device) : T #[1, vals.size.toUInt64] :=
  reshape ((data.fromInt64Array vals).to device) #[1, vals.size.toUInt64]

/-- Max abs / max rel (denominator clamped by 1e-3) error in FP32. -/
private def maxErrors (got expected : T #[]) : IO (Float × Float) := do
  let ad : T #[] := nn.abs (sub (toFloat' got) (toFloat' expected))
  let maxAbs := nn.item (nn.maxAll ad)
  let denom : T #[] := add_scalar (nn.abs (toFloat' expected)) 1e-3
  let maxRel := nn.item (nn.maxAll (nn.div ad denom))
  pure (maxAbs, maxRel)

/-- Sorted top-k id set per row of an erased logits tensor (`[n, vocab]`, or
    `[vocab]` which is treated as one row). -/
private def topkIdSets (logits : T #[]) (k : UInt64) : IO (Array (Array UInt64)) := do
  let sh := T.runtimeShape logits
  let (n, v) :=
    if sh.size == 1 then (1, sh.getD 0 0)
    else (sh.getD 0 0, sh.getD 1 0)
  let l2 : T #[n, v] := reshape logits #[n, v]
  let (_, idx) := torch.topk_2d l2 k 1
  let flat ← data.tensorToUInt64Array' (nn.eraseShape idx)
  pure (Array.ofFn (n := n.toNat) fun r =>
    (flat.extract (r.val * k.toNat) (r.val * k.toNat + k.toNat)).qsort (fun a b => a < b))

/-- Abs-diff gate + top-5-set gate for one logits tensor. -/
private def checkLogits (lbl name : String) (got ref : T #[]) (tol : Float) : IO Unit := do
  let (maxAbs, maxRel) ← maxErrors got ref
  IO.println s!"  [{lbl}] {name}: maxAbs={maxAbs} maxRel={maxRel} (tol={tol})"
  check (maxAbs ≤ tol) s!"[{lbl}] {name} maxAbs={maxAbs} ≤ {tol}"
  let gotSets ← topkIdSets got 5
  let refSets ← topkIdSets ref 5
  check (gotSets == refSets) s!"[{lbl}] {name} top-5 id sets match at every position"

/-- Abs-diff report + gate for hidden-state diagnostics. -/
private def checkClose (lbl name : String) (got ref : T #[]) (tol : Float) : IO Unit := do
  let (maxAbs, maxRel) ← maxErrors got ref
  IO.println s!"  [{lbl}] {name}: maxAbs={maxAbs} maxRel={maxRel} (tol={tol})"
  check (maxAbs ≤ tol) s!"[{lbl}] {name} maxAbs={maxAbs} ≤ {tol}"

private def fixtureDir : String := "Tests/fixtures/laguna/tiny"

/-- The fixture's reference ids (see reference.json). -/
private def promptA : Array Int64 := #[2, 37, 101, 456, 1000, 5, 9]
private def promptB : Array Int64 :=
  #[116, 151, 318, 914, 319, 188, 397, 253, 548, 955,
    78, 159, 531, 507, 590, 668, 350, 783, 901, 786]
private def genRef : Array UInt64 :=
  #[969, 969, 969, 969, 317, 340, 346, 352, 689, 559, 803, 424, 424, 424, 424, 842]

private def runFixture (device : Device) : IO Unit := do
  let lbl := deviceLabel device
  IO.println s!"-- tiny-fixture HF parity [{lbl}]"

  -- 1. Config.
  let cfg ← Config.loadFromPretrainedDir fixtureDir
  check (cfg.num_hidden_layers == 4) s!"[{lbl}] cfg num_hidden_layers == 4"
  check (cfg.num_attention_heads == 4) s!"[{lbl}] cfg num_attention_heads == 4"
  check (cfg.num_attention_heads_sliding == 6) s!"[{lbl}] cfg num_attention_heads_sliding == 6"
  check (cfg.num_key_value_heads == 2) s!"[{lbl}] cfg num_key_value_heads == 2"
  check (cfg.num_experts == 8) s!"[{lbl}] cfg num_experts == 8"
  check (cfg.num_experts_per_tok == 2) s!"[{lbl}] cfg num_experts_per_tok == 2"
  check (cfg.sliding_window == 8) s!"[{lbl}] cfg sliding_window == 8"
  check (cfg.layerType 0 == .fullAttention) s!"[{lbl}] cfg layer 0 full attention"
  check (cfg.layerType 1 == .slidingAttention) s!"[{lbl}] cfg layer 1 sliding attention"
  check (cfg.isDenseMlpLayer 0) s!"[{lbl}] cfg layer 0 dense MLP"

  -- 2. Checkpoint load (dense-fused expert layout; weights are BF16 as stored).
  let log : torch.Log.Handlers := { onInfo := fun m => IO.println s!"  {m}" }
  let modelB16 ← LagunaForCausalLM.load s!"{fixtureDir}/model.safetensors" cfg device log
  let layer0b ←
    match modelB16.model.layers[0]? with
    | some l => pure l
    | none => throw (IO.userError s!"[{lbl}] missing layer 0")
  let layer1b ←
    match modelB16.model.layers[1]? with
    | some l => pure l
    | none => throw (IO.userError s!"[{lbl}] missing layer 1")
  check layer0b.denseMlp.isSome s!"[{lbl}] layer 0 loaded as dense MLP"
  let moe1b ←
    match layer1b.sparseMoe with
    | some m => pure m
    | none => throw (IO.userError s!"[{lbl}] layer 1 missing sparse MoE block")
  check moe1b.denseExperts.isSome s!"[{lbl}] layer 1 uses dense BF16 expert banks"
  check moe1b.router.eScoreCorrectionBias.isSome
    s!"[{lbl}] layer 1 router e_score_correction_bias loaded"

  -- Gated parity runs in FP32 (all weights upcast). The reference tie margins
  -- are far below bf16 rounding (min top-5/top-6 logit margin ≈ 3e-4; bf16
  -- logits noise ≈ 1e-2..1e-1), so NO bf16 engine can meet the top-5-set and
  -- exact-gen gates against the fp32 reference — HF's own bf16 run flips
  -- top-5 sets at 4 positions of logits_b and diverges from gen_ids at step 4.
  -- FP32 compute tests the math itself (which is what parity means here);
  -- the BF16 pass below is informational.
  let model : LagunaForCausalLM cfg := TensorStruct.map (fun t => toFloat' t) modelB16
  let layer0 ←
    match model.model.layers[0]? with
    | some l => pure l
    | none => throw (IO.userError s!"[{lbl}] missing layer 0 (fp32)")

  -- References (fp32, CPU) moved to the comparison device.
  let refPath := s!"{fixtureDir}/reference.safetensors"
  let logitsARef ← torch.safetensors.loadTensor refPath "logits_a" #[7, 1024]
  let logitsBRef ← torch.safetensors.loadTensor refPath "logits_b" #[20, 1024]
  let genLastRef ← torch.safetensors.loadTensor refPath "gen_logits_last" #[1024]
  let hiddenL0Ref ← torch.safetensors.loadTensor refPath "hidden_a_l0" #[7, 256]
  let hiddenFinRef ← torch.safetensors.loadTensor refPath "hidden_a_final" #[7, 256]

  let idsA : T #[1, 7] := mkIds promptA device
  let idsB : T #[1, 20] := mkIds promptB device

  -- Hidden-state diagnostics for input A (layer-0 output, final hidden).
  let x0 : T #[1, 7, cfg.hidden_size] := nn.embedding idsA model.model.embed_tokens
  let tables ← precomputeRotaryTables cfg 7 x0.device
  let h0 ← layer0.forward cfg x0 tables
  checkClose lbl "hidden_a_l0" (nn.eraseShape h0)
    (nn.eraseShape (hiddenL0Ref.to device)) 0.05
  let hFin ← model.model.forward cfg idsA
  checkClose lbl "hidden_a_final" (nn.eraseShape hFin)
    (nn.eraseShape (hiddenFinRef.to device)) 0.05

  -- 3. Full-forward logits vs fp32 references.
  let logitsA ← model.forward cfg idsA
  checkLogits lbl "logits_a" (nn.eraseShape (reshape logitsA #[7, cfg.vocab_size]))
    (nn.eraseShape (logitsARef.to device)) 0.05
  let logitsB ← model.forward cfg idsB
  checkLogits lbl "logits_b" (nn.eraseShape (reshape logitsB #[20, cfg.vocab_size]))
    (nn.eraseShape (logitsBRef.to device)) 0.05

  -- 4. Greedy generation: 16 tokens from A, exact id match (EOS disabled,
  --    budget termination; the HF reference used its KV-cache path).
  let r ← model.generate cfg idsA 16 .greedy #[]
  check (r.1 == 23) s!"[{lbl}] generate produced 7 + 16 tokens (outSeq={r.1})"
  let genGot ← data.tensorToUInt64Array' (nn.eraseShape (data.slice r.2 1 7 16))
  check (genGot == genRef) s!"[{lbl}] greedy gen ids EXACTLY match reference gen_ids"

  -- 5. Final-step logits: the logits that produced gen_ids[15] come from the
  --    forward whose last input token is gen_ids[14] (position 21 of
  --    A ++ gen_ids[0..15)).
  let idsFull := mkIds (promptA ++ (genRef.extract 0 15).map (fun t => Int64.ofNat t.toNat)) device
  let logitsFull ← model.forward cfg idsFull
  let lastLogits : T #[1024] := reshape (data.slice logitsFull 1 21 1) #[1024]
  checkLogits lbl "gen_logits_last" (nn.eraseShape lastLogits)
    (nn.eraseShape (genLastRef.to device)) 0.05

  -- Informational BF16 pass (the checkpoint's storage dtype): report diffs
  -- only. Bf16 rounding alone (HF bf16 vs HF fp32: logits_b maxAbs ≈ 0.102)
  -- exceeds the 0.05 gate, so no PASS/FAIL here.
  IO.println (s!"  [{lbl}] -- informational BF16 pass (no gates; HF bf16 self-noise: " ++
    "logits_a 0.0137, logits_b 0.1021, gen diverges at step 4)")
  let logitsAb ← modelB16.forward cfg idsA
  let (abAbs, abRel) ← maxErrors (nn.eraseShape (reshape logitsAb #[7, cfg.vocab_size]))
    (nn.eraseShape (logitsARef.to device))
  IO.println s!"  [{lbl}] bf16 logits_a: maxAbs={abAbs} maxRel={abRel}"
  let logitsBb ← modelB16.forward cfg idsB
  let (bbAbs, bbRel) ← maxErrors (nn.eraseShape (reshape logitsBb #[20, cfg.vocab_size]))
    (nn.eraseShape (logitsBRef.to device))
  IO.println s!"  [{lbl}] bf16 logits_b: maxAbs={bbAbs} maxRel={bbRel}"
  let rb ← modelB16.generate cfg idsA 16 .greedy #[]
  let genB ← data.tensorToUInt64Array' (nn.eraseShape (data.slice rb.2 1 7 16))
  IO.println s!"  [{lbl}] bf16 greedy ids: {genB} (exact match vs fp32 ref: {genB == genRef})"

/-- Real NVFP4-checkpoint smoke load: config + first `numLayers` layers via
    the sharded packed-expert path, bank-shape checks, one forward. -/
private def runRealSmoke (modelDir : String) (numLayers : UInt64) (device : Device) : IO Unit := do
  let lbl := deviceLabel device
  IO.println s!"-- real-checkpoint smoke [{lbl}] dir={modelDir} layers={numLayers}"
  let cfg0 ← Config.loadFromPretrainedDir modelDir
  IO.println (s!"  full config: {cfg0.num_hidden_layers} layers, hidden {cfg0.hidden_size}, " ++
    s!"{cfg0.num_experts} experts top-{cfg0.num_experts_per_tok}")
  if numLayers > cfg0.num_hidden_layers then
    throw (IO.userError s!"numLayers={numLayers} > num_hidden_layers={cfg0.num_hidden_layers}")
  -- Truncate the layer schedule so only the first `numLayers` layers load.
  let cfg : Config := { cfg0 with
    num_hidden_layers := numLayers
    layer_types := (Config.normalizedLayerTypes cfg0).extract 0 numLayers.toNat }
  let log : torch.Log.Handlers := { onInfo := fun m => IO.println s!"  {m}" }
  let t0 ← IO.monoMsNow
  let model ← LagunaForCausalLM.loadSharded modelDir cfg device log
  let t1 ← IO.monoMsNow
  IO.println s!"  loaded {numLayers}/{cfg0.num_hidden_layers} layers in {t1 - t0} ms"

  let layer0 ←
    match model.model.layers[0]? with
    | some l => pure l
    | none => throw (IO.userError "missing layer 0")
  check layer0.denseMlp.isSome "layer 0 loaded as dense MLP"
  check layer0.attnFull.isSome "layer 0 loaded as full attention"
  if numLayers ≥ 2 then
    let layer1 ←
      match model.model.layers[1]? with
      | some l => pure l
      | none => throw (IO.userError "missing layer 1")
    check layer1.attnSliding.isSome "layer 1 loaded as sliding attention"
    match layer1.sparseMoe with
    | some moe =>
      check moe.denseExperts.isNone "layer 1 uses NVFP4-packed expert banks"
      check (moe.router.eScoreCorrectionBias.isSome)
        "layer 1 router e_score_correction_bias loaded (experts.* key)"
      let sh := T.runtimeShape moe.experts.gatePacked
      check (sh == #[cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size / 2])
        s!"layer 1 gatePacked bank shape {sh} == [{cfg.num_experts}, {cfg.moe_intermediate_size}, {cfg.hidden_size / 2}]"
      let shD := T.runtimeShape moe.experts.downPacked
      check (shD == #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size / 2])
        s!"layer 1 downPacked bank shape {shD}"
      let shG := T.runtimeShape moe.experts.gateGlobal
      check (shG == #[cfg.num_experts]) s!"layer 1 gateGlobal bank shape {shG}"
    | none => throw (IO.userError "layer 1 missing sparse MoE block")

  -- Forward smoke on the truncated model (numerics meaningless by design).
  let ids : T #[1, 8] := mkIds #[2, 100, 200, 300, 400, 500, 600, 700] device
  let logits ← model.forward cfg ids
  let shL := T.runtimeShape (nn.eraseShape logits)
  check (shL == #[1, 8, cfg.vocab_size]) s!"forward ok: logits shape {shL}"
  IO.println s!"Real-checkpoint smoke [{lbl}] passed."

def main (args : List String) : IO Unit := do
  match args with
  | [] =>
    runFixture Device.CPU
    if ← torch.cuda_is_available then
      runFixture (Device.CUDA 0)
      torch.cuda_synchronize
    else
      IO.println "CUDA not available; skipped CUDA parity."
    IO.println "All Laguna parity tests passed."
  | dir :: rest =>
    let numLayers := ((rest.getD 0 "2").toNat?.getD 2).toUInt64
    let device := if rest.getD 1 "cpu" == "cuda" then Device.CUDA 0 else Device.CPU
    runRealSmoke dir numLayers device
