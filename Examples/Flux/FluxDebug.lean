/-
  Examples/Flux/FluxDebug.lean

  Debug harness for Flux Klein 4B.
  Runs the same pipeline as FluxDemo but saves intermediate tensors to disk.
-/
import Tyr.Torch
import Tyr.Model.Qwen
import Tyr.Model.Qwen.Weights
import Tyr.Model.VAE
import Tyr.Model.VAE.Weights
import Tyr.Model.Flux
import Tyr.Model.Flux.Weights
import Tyr.Tokenizer.Qwen3

namespace Examples.Flux

open torch
open torch.qwen
open torch.vae
open torch.flux

private def toFloat32 {α : Type} [TensorStruct α] (x : α) : α :=
  TensorStruct.map (fun t => t.checkedReplacement t.toFloat32.raw) x

private def toFloat32Embedder {cfg : QwenConfig} {max_seq : UInt64}
    (e : QwenFluxEmbedder cfg max_seq) : QwenFluxEmbedder cfg max_seq :=
  { e with model := toFloat32 e.model }

/-- Configuration for the debug run -/
structure DebugConfig where
  qwenDir : String := "weights/flux-klein-4b/text_encoder"
  fluxPath : String := "weights/flux.safetensors"
  vaePath : String := "weights/ae.safetensors"
  outputPath : String := "output.ppm"
  useTokenFiles : Bool := false
  tokensPath : String := "weights/flux-klein-4b/tokenizer/tokens.txt"
  attnMaskPath : String := "weights/flux-klein-4b/tokenizer/attention_mask.txt"
  numSteps : Nat := 4
  latentH : UInt64 := 16
  latentW : UInt64 := 16
  debugDir : String := "debug"
  deriving Repr, Inhabited

private def qwenPadId : Int64 := 151643
private def maxSeqLen : UInt64 := 512

private def parseTokenIds (content : String) : Array Int64 := Id.run do
  let mut out : Array Int64 := #[]
  let mut cur : String := ""
  for c in content.toList do
    if c.isDigit then
      cur := cur.push c
    else
      if !cur.isEmpty then
        match cur.toNat? with
        | some n => out := out.push n.toInt64
        | none => pure ()
        cur := ""
  if !cur.isEmpty then
    match cur.toNat? with
    | some n => out := out.push n.toInt64
    | none => pure ()
  out

private def padTokens (ids : Array Int64) (maxLen : UInt64) (padId : Int64)
    : Array Int64 × Array Int64 := Id.run do
  let max := maxLen.toNat
  let mut tokens : Array Int64 := Array.mkEmpty max
  let mut mask : Array Int64 := Array.mkEmpty max
  for i in [:max] do
    if h : i < ids.size then
      tokens := tokens.push (ids[i]'h)
      mask := mask.push (1 : Int64)
    else
      tokens := tokens.push padId
      mask := mask.push (0 : Int64)
  (tokens, mask)

private def loadTokenIds (path : String) (maxLen : UInt64) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  let mut ids := parseTokenIds contents
  if ids.isEmpty then
    throw (IO.userError s!"No tokens found in {path}")
  if ids.size.toUInt64 > maxLen then
    ids := ids.extract 0 maxLen.toNat
  pure ids

private def loadMaskIds (path : String) (maxLen : UInt64) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  let mut ids := parseTokenIds contents
  if ids.isEmpty then
    throw (IO.userError s!"No mask values found in {path}")
  if ids.size.toUInt64 > maxLen then
    ids := ids.extract 0 maxLen.toNat
  if ids.size.toUInt64 < maxLen then
    let mut padded : Array Int64 := Array.mkEmpty maxLen.toNat
    for i in [:maxLen.toNat] do
      if h : i < ids.size then
        padded := padded.push (ids[i]'h)
      else
        padded := padded.push (0 : Int64)
    ids := padded
  pure ids

private def saveDebug {σ : StaticSpec} (dir : String) (name : String) (t : Tensor σ) : IO Unit := do
  let path := s!"{dir}/{name}.pt"
  Tensor.saveTensor t path

/-- Construct indexed token IDs and attention masks from a prompt or token files. -/
private def promptTokens (cfg : DebugConfig) (prompt : String) : IO (DTensor #[1, maxSeqLen] .Int64 × DTensor #[1, maxSeqLen] .Int64) := do
  if cfg.useTokenFiles then
    let ids ← loadTokenIds cfg.tokensPath maxSeqLen
    let (padded, defaultMask) := padTokens ids maxSeqLen qwenPadId
    let mask ←
      if (← data.fileExists cfg.attnMaskPath) then
        loadMaskIds cfg.attnMaskPath maxSeqLen
      else
        pure defaultMask
    let tokens := Tensor.reshapeChecked (Tensor.fromInt64Array padded) #[1, maxSeqLen]
    let attnMask := Tensor.reshapeChecked (Tensor.fromInt64Array mask) #[1, maxSeqLen]
    pure (tokens, attnMask)
  else
    let tok ← tokenizer.qwen3.loadTokenizer cfg.qwenDir
    let (tokIds, maskIds) := tokenizer.qwen3.encodePrompt tok prompt maxSeqLen.toNat
    let tokens := Tensor.reshapeChecked (Tensor.fromInt64Array (tokenizer.qwen3.toInt64Array tokIds)) #[1, maxSeqLen]
    let attnMask := Tensor.reshapeChecked (Tensor.fromInt64Array (tokenizer.qwen3.toInt64Array maskIds)) #[1, maxSeqLen]
    pure (tokens, attnMask)
/-- Main debug run: saves intermediate tensors to disk. -/
def runDebug (cfg : DebugConfig) (prompt : String) : IO Unit := do
  IO.FS.createDirAll cfg.debugDir

  -- Load models
  let qwenCfg := QwenConfig.fluxKleinTextEncoder
  let qwen ← loadQwenFluxEmbedderSharded cfg.qwenDir qwenCfg maxSeqLen #[8, 17, 26]
  let qwen := toFloat32Embedder qwen

  let fluxCfg := FluxConfig.klein4B
  let flux ← loadFluxModel cfg.fluxPath fluxCfg
  let flux := toFloat32 flux

  let vae ← loadAutoEncoder cfg.vaePath
  let vae := toFloat32 vae

  -- Tokenize and encode text
  let (tokens, attnMask) ← promptTokens cfg prompt
  let txtEmb := qwen.encodeMasked (batch := 1) (seq := maxSeqLen) (maskDtype := .Int64) qwenCfg maxSeqLen tokens attnMask
  let txtEmb : STensor #[1, maxSeqLen, fluxCfg.context_in_dim] :=
    (txtEmb.reshapeChecked #[1, maxSeqLen, fluxCfg.context_in_dim]).forgetDType

  saveDebug cfg.debugDir "tokens" tokens
  saveDebug cfg.debugDir "attn_mask" attnMask
  saveDebug cfg.debugDir "txt_emb" txtEmb

  -- Prepare latent space
  let imgSeq := cfg.latentH * cfg.latentW
  let txtSeq := maxSeqLen
  let noiseSpatial ← randomWeight #[1, 128, cfg.latentH, cfg.latentW]
  let noiseSeq := Tensor.permute noiseSpatial #[0, 2, 3, 1]
  let noise : STensor #[1, imgSeq, 128] := Tensor.reshapeChecked noiseSeq #[1, imgSeq, 128]
  saveDebug cfg.debugDir "noise" noise

  -- Position IDs
  let imgIds ← computeImagePositionIds cfg.latentH cfg.latentW
  let txtIds ← computeTextPositionIds maxSeqLen
  saveDebug cfg.debugDir "img_ids" imgIds
  saveDebug cfg.debugDir "txt_ids" txtIds

  -- First-step prediction (and intermediate captures)
  let timesteps := computeTimesteps imgSeq cfg.numSteps
  let (t_curr, _t_next) := timesteps.getD 0 (0.0, 0.0)
  let t := Tensor.full #[1] t_curr
  saveDebug cfg.debugDir "t0" t

  -- Timestep embedding + model modulation
  let t_emb := timestepEmbedding t fluxCfg.time_dim
  let vec := flux.time_in.forward (Tensor.castLike flux.time_in.in_layer t_emb)
  saveDebug cfg.debugDir "vec" vec
  let mod_img := flux.double_stream_modulation_img.forward true vec
  let mod_txt := flux.double_stream_modulation_txt.forward true vec
  saveDebug cfg.debugDir "mod_img" mod_img
  saveDebug cfg.debugDir "mod_txt" mod_txt

  -- Input projections
  let img_proj := Tensor.linear3d noise flux.img_in
  let txt_proj := Tensor.linear3d txtEmb flux.txt_in
  saveDebug cfg.debugDir "img_proj" img_proj
  saveDebug cfg.debugDir "txt_proj" txt_proj

  -- RoPE embeddings
  let img_pe := ropeEmbed (head_dim := fluxCfg.head_dim) imgIds fluxCfg.axes_dims fluxCfg.theta
  let txt_pe := ropeEmbed (head_dim := fluxCfg.head_dim) txtIds fluxCfg.axes_dims fluxCfg.theta
  saveDebug cfg.debugDir "img_pe" img_pe
  saveDebug cfg.debugDir "txt_pe" txt_pe

  -- First double-stream block output (with internal captures)
  let mlp_hidden := FluxConfig.mlpHiddenDim fluxCfg
  let mut img_cur := img_proj
  let mut txt_cur := txt_proj
  let mut saved_first := false
  for block in flux.double_blocks do
    if !saved_first then
      -- Split modulation into two triplets
      let img_mod1 := sliceTyped mod_img 1 0 3
      let img_mod2 := sliceTyped mod_img 1 3 3
      let txt_mod1 := sliceTyped mod_txt 1 0 3
      let txt_mod2 := sliceTyped mod_txt 1 3 3

      let img_mod1_shift := Tensor.reshapeChecked (sliceTyped img_mod1 1 0 1) #[1, fluxCfg.hidden_size]
      let img_mod1_scale := Tensor.reshapeChecked (sliceTyped img_mod1 1 1 1) #[1, fluxCfg.hidden_size]
      let img_mod1_gate := Tensor.reshapeChecked (sliceTyped img_mod1 1 2 1) #[1, fluxCfg.hidden_size]
      let img_mod2_shift := Tensor.reshapeChecked (sliceTyped img_mod2 1 0 1) #[1, fluxCfg.hidden_size]
      let img_mod2_scale := Tensor.reshapeChecked (sliceTyped img_mod2 1 1 1) #[1, fluxCfg.hidden_size]
      let img_mod2_gate := Tensor.reshapeChecked (sliceTyped img_mod2 1 2 1) #[1, fluxCfg.hidden_size]

      let txt_mod1_shift := Tensor.reshapeChecked (sliceTyped txt_mod1 1 0 1) #[1, fluxCfg.hidden_size]
      let txt_mod1_scale := Tensor.reshapeChecked (sliceTyped txt_mod1 1 1 1) #[1, fluxCfg.hidden_size]
      let txt_mod1_gate := Tensor.reshapeChecked (sliceTyped txt_mod1 1 2 1) #[1, fluxCfg.hidden_size]
      let txt_mod2_shift := Tensor.reshapeChecked (sliceTyped txt_mod2 1 0 1) #[1, fluxCfg.hidden_size]
      let txt_mod2_scale := Tensor.reshapeChecked (sliceTyped txt_mod2 1 1 1) #[1, fluxCfg.hidden_size]
      let txt_mod2_gate := Tensor.reshapeChecked (sliceTyped txt_mod2 1 2 1) #[1, fluxCfg.hidden_size]

      -- Norm + modulation
      let img_norm := block.img_norm1.forward3d img_cur
      let img_mod := applyModulation img_norm img_mod1_scale img_mod1_shift
      let txt_norm := block.txt_norm1.forward3d txt_cur
      let txt_mod := applyModulation txt_norm txt_mod1_scale txt_mod1_shift

      -- QKV projections
      let img_qkv := Tensor.linear3d img_mod block.img_attn.qkv
      let txt_qkv := Tensor.linear3d txt_mod block.txt_attn.qkv

      let img_qkv := Tensor.reshapeChecked img_qkv #[1, imgSeq, 3, fluxCfg.num_heads, fluxCfg.head_dim]
      let txt_qkv := Tensor.reshapeChecked txt_qkv #[1, txtSeq, 3, fluxCfg.num_heads, fluxCfg.head_dim]

      let img_q := Tensor.reshapeChecked (sliceTyped img_qkv 2 0 1) #[1, imgSeq, fluxCfg.num_heads, fluxCfg.head_dim]
      let img_k := Tensor.reshapeChecked (sliceTyped img_qkv 2 1 1) #[1, imgSeq, fluxCfg.num_heads, fluxCfg.head_dim]
      let img_v := Tensor.reshapeChecked (sliceTyped img_qkv 2 2 1) #[1, imgSeq, fluxCfg.num_heads, fluxCfg.head_dim]

      let txt_q := Tensor.reshapeChecked (sliceTyped txt_qkv 2 0 1) #[1, txtSeq, fluxCfg.num_heads, fluxCfg.head_dim]
      let txt_k := Tensor.reshapeChecked (sliceTyped txt_qkv 2 1 1) #[1, txtSeq, fluxCfg.num_heads, fluxCfg.head_dim]
      let txt_v := Tensor.reshapeChecked (sliceTyped txt_qkv 2 2 1) #[1, txtSeq, fluxCfg.num_heads, fluxCfg.head_dim]

      -- QK normalization
      let (img_q_norm, img_k_norm) := block.img_attn.norm.forward img_q img_k
      let (txt_q_norm, txt_k_norm) := block.txt_attn.norm.forward txt_q txt_k
      saveDebug cfg.debugDir "img_q_norm" img_q_norm
      saveDebug cfg.debugDir "img_k_norm" img_k_norm
      saveDebug cfg.debugDir "txt_q_norm" txt_q_norm
      saveDebug cfg.debugDir "txt_k_norm" txt_k_norm

      -- Apply RoPE
      let img_q_rope := applyRope img_q_norm img_pe
      let img_k_rope := applyRope img_k_norm img_pe
      let txt_q_rope := applyRope txt_q_norm txt_pe
      let txt_k_rope := applyRope txt_k_norm txt_pe
      saveDebug cfg.debugDir "img_q_rope" img_q_rope
      saveDebug cfg.debugDir "img_k_rope" img_k_rope
      saveDebug cfg.debugDir "txt_q_rope" txt_q_rope
      saveDebug cfg.debugDir "txt_k_rope" txt_k_rope

      -- Joint attention
      let q : STensor #[1, txtSeq + imgSeq, fluxCfg.num_heads, fluxCfg.head_dim] := qwen35.typedCat txt_q_rope img_q_rope 1
      let k : STensor #[1, txtSeq + imgSeq, fluxCfg.num_heads, fluxCfg.head_dim] := qwen35.typedCat txt_k_rope img_k_rope 1
      let v : STensor #[1, txtSeq + imgSeq, fluxCfg.num_heads, fluxCfg.head_dim] := qwen35.typedCat txt_v img_v 1
      let q := transposeAttention q
      let k := transposeAttention k
      let v := transposeAttention v
      let attn := Tensor.attentionGQA q k v 0.0 false
      let attn := fromAttention attn
      let txt_attn := sliceTyped attn 1 0 txtSeq
      let img_attn := sliceTyped attn 1 txtSeq imgSeq
      let img_attn := Tensor.reshapeChecked img_attn #[1, imgSeq, fluxCfg.num_heads * fluxCfg.head_dim]
      let txt_attn := Tensor.reshapeChecked txt_attn #[1, txtSeq, fluxCfg.num_heads * fluxCfg.head_dim]

      let img_attn := Tensor.linear3d img_attn block.img_attn.proj
      let txt_attn := Tensor.linear3d txt_attn block.txt_attn.proj
      saveDebug cfg.debugDir "img_attn_proj" img_attn
      saveDebug cfg.debugDir "txt_attn_proj" txt_attn

      let img_mod1_gate := Tensor.unsqueeze img_mod1_gate 1
      let img_mod1_gate := Tensor.expand img_mod1_gate #[1, imgSeq, fluxCfg.hidden_size]
      let txt_mod1_gate := Tensor.unsqueeze txt_mod1_gate 1
      let txt_mod1_gate := Tensor.expand txt_mod1_gate #[1, txtSeq, fluxCfg.hidden_size]

      let img_mid := img_cur + img_mod1_gate * img_attn
      let txt_mid := txt_cur + txt_mod1_gate * txt_attn

      let img_mlp_in := block.img_norm2.forward3d img_mid
      let img_mlp_in := applyModulation img_mlp_in img_mod2_scale img_mod2_shift
      let img_mlp_proj := Tensor.linear3d img_mlp_in block.img_mlp.w1
      let img_mlp_proj := Tensor.reshapeChecked img_mlp_proj #[1, imgSeq, 2, mlp_hidden]
      let img_gate := Tensor.reshapeChecked (sliceTyped img_mlp_proj 2 0 1) #[1, imgSeq, mlp_hidden]
      let img_up := Tensor.reshapeChecked (sliceTyped img_mlp_proj 2 1 1) #[1, imgSeq, mlp_hidden]
      let img_mlp_out := Tensor.Preserving.silu img_gate * img_up
      let img_mlp_out := Tensor.linear3d img_mlp_out block.img_mlp.w2
      saveDebug cfg.debugDir "img_mlp_out" img_mlp_out

      let txt_mlp_in := block.txt_norm2.forward3d txt_mid
      let txt_mlp_in := applyModulation txt_mlp_in txt_mod2_scale txt_mod2_shift
      let txt_mlp_proj := Tensor.linear3d txt_mlp_in block.txt_mlp.w1
      let txt_mlp_proj := Tensor.reshapeChecked txt_mlp_proj #[1, txtSeq, 2, mlp_hidden]
      let txt_gate := Tensor.reshapeChecked (sliceTyped txt_mlp_proj 2 0 1) #[1, txtSeq, mlp_hidden]
      let txt_up := Tensor.reshapeChecked (sliceTyped txt_mlp_proj 2 1 1) #[1, txtSeq, mlp_hidden]
      let txt_mlp_out := Tensor.Preserving.silu txt_gate * txt_up
      let txt_mlp_out := Tensor.linear3d txt_mlp_out block.txt_mlp.w2
      saveDebug cfg.debugDir "txt_mlp_out" txt_mlp_out

      let img_mod2_gate := Tensor.unsqueeze img_mod2_gate 1
      let img_mod2_gate := Tensor.expand img_mod2_gate #[1, imgSeq, fluxCfg.hidden_size]
      let txt_mod2_gate := Tensor.unsqueeze txt_mod2_gate 1
      let txt_mod2_gate := Tensor.expand txt_mod2_gate #[1, txtSeq, fluxCfg.hidden_size]

      let img_next := img_mid + img_mod2_gate * img_mlp_out
      let txt_next := txt_mid + txt_mod2_gate * txt_mlp_out
      saveDebug cfg.debugDir "img1" img_next
      saveDebug cfg.debugDir "txt1" txt_next

      saved_first := true
      img_cur := img_next
      txt_cur := txt_next
    else
      let (img_next, txt_next) := block.forward img_cur txt_cur img_pe txt_pe mod_img mod_txt
      img_cur := img_next
      txt_cur := txt_next

  let pred_t0 := flux.forward fluxCfg noise txtEmb t imgIds txtIds
  saveDebug cfg.debugDir "pred_t0" pred_t0

  -- Full denoise
  let latents := denoise fluxCfg flux noise txtEmb imgIds txtIds cfg.numSteps
  saveDebug cfg.debugDir "latents" latents

  -- Decode with VAE
  let latentsHW : STensor #[1, cfg.latentH, cfg.latentW, 128] :=
    Tensor.reshapeChecked latents #[1, cfg.latentH, cfg.latentW, 128]
  let packedLatents : STensor #[1, 128, cfg.latentH, cfg.latentW] :=
    (Tensor.permute latentsHW #[0, 3, 1, 2]).reshapeChecked #[1, 128, cfg.latentH, cfg.latentW]
  saveDebug cfg.debugDir "packed_latents" packedLatents

  let z := vae.invNormalize (packedLatents.reshapeChecked #[1, 128, 16, 16])
  let z := torch.vae.AutoEncoder.unpackLatents16x16 z
  let image := vae.decoder.forward1 z
  saveDebug cfg.debugDir "decoded" image

  saveImagePPM image cfg.outputPath
  IO.println s!"Saved debug tensors to {cfg.debugDir}"
  IO.println s!"Image saved to {cfg.outputPath}"

/-- Main entry point -/
def _root_.main (args : List String) : IO UInt32 := do
  if args == ["--help"] then
    IO.println "Usage: FluxDebug [prompt]\nUses local Flux, Qwen, and VAE checkpoints under weights/."
    return 0
  let prompt := match args with
    | arg :: _ => arg
    | [] => "a cat sitting on a windowsill, photorealistic"

  let cfg : DebugConfig := {}
  runDebug cfg prompt
  return 0

end Examples.Flux
