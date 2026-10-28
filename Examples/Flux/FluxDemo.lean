/-
  Examples/Flux/FluxDemo.lean

  End-to-end demonstration of Flux Klein 4B image generation.
  Shows complete pipeline: text encoding → diffusion → VAE decoding.
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

/-- Configuration for the demo -/
structure DemoConfig where
  /-- Directory containing sharded Qwen text encoder weights -/
  qwenDir : String := "weights/flux-klein-4b/text_encoder"
  /-- Path to Flux Klein 4B SafeTensors weights (single file) -/
  fluxPath : String := "weights/flux.safetensors"
  /-- Path to VAE SafeTensors weights (single file) -/
  vaePath : String := "weights/ae.safetensors"
  /-- Output image path -/
  outputPath : String := "output.ppm"
  /-- Whether to load tokens from files instead of generating them -/
  useTokenFiles : Bool := false
  /-- Optional token file (whitespace-separated token IDs) -/
  tokensPath : String := "weights/flux-klein-4b/tokenizer/tokens.txt"
  /-- Optional attention mask file (whitespace-separated 0/1 IDs) -/
  attnMaskPath : String := "weights/flux-klein-4b/tokenizer/attention_mask.txt"
  /-- Number of diffusion steps -/
  numSteps : Nat := 4
  /-- Packed latent height (16 for 256x256 output with 2×2 patchification and 8× VAE upscaling) -/
  latentH : Nat := 16
  /-- Packed latent width (16 for 256x256 output with 2×2 patchification and 8× VAE upscaling) -/
  latentW : Nat := 16
  deriving Repr, Inhabited

private def qwenPadId : Int64 := 151643
private def maxSeqLen : Nat := 512

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

private def padTokens (ids : Array Int64) (maxLen : Nat) (padId : Int64)
    : Array Int64 × Array Int64 := Id.run do
  let max := maxLen
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

private def loadTokenIds (path : String) (maxLen : Nat) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  let mut ids := parseTokenIds contents
  if ids.isEmpty then
    throw (IO.userError s!"No tokens found in {path}")
  if ids.size > maxLen then
    IO.println s!"    Truncating tokens from {ids.size} to {maxLen}"
    ids := ids.extract 0 maxLen
  pure ids

private def loadMaskIds (path : String) (maxLen : Nat) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  let mut ids := parseTokenIds contents
  if ids.isEmpty then
    throw (IO.userError s!"No mask values found in {path}")
  if ids.size > maxLen then
    IO.println s!"    Truncating mask from {ids.size} to {maxLen}"
    ids := ids.extract 0 maxLen
  if ids.size < maxLen then
    let mut padded : Array Int64 := Array.mkEmpty maxLen
    for i in [:maxLen] do
      if h : i < ids.size then
        padded := padded.push (ids[i]'h)
      else
        padded := padded.push (0 : Int64)
    ids := padded
  pure ids

/-- Construct indexed token IDs and attention masks from a prompt or token files. -/
private def promptTokens (cfg : DemoConfig) (prompt : String) : IO (Tensor { shape := #[1, maxSeqLen], dtype := .Int64, device := .CPU } × Tensor { shape := #[1, maxSeqLen], dtype := .Int64, device := .CPU } × Nat) := do
  if cfg.useTokenFiles then
    if !(← data.fileExists cfg.tokensPath) then
      throw (IO.userError s!"Token file not found: {cfg.tokensPath}")
    IO.println s!"    Loading tokens from {cfg.tokensPath}"
    let ids ← loadTokenIds cfg.tokensPath maxSeqLen
    let tokenCount := ids.size
    let (padded, defaultMask) := padTokens ids maxSeqLen qwenPadId
    let mask ←
      if (← data.fileExists cfg.attnMaskPath) then
        IO.println s!"    Loading attention mask from {cfg.attnMaskPath}"
        loadMaskIds cfg.attnMaskPath maxSeqLen
      else
        pure defaultMask
    let tokens := (← IO.ofExcept (Tensor.fromInt64ArrayAs? padded #[1, maxSeqLen]))
    let attnMask := (← IO.ofExcept (Tensor.fromInt64ArrayAs? mask #[1, maxSeqLen]))
    pure (tokens, attnMask, tokenCount)
  else
    let tok ← tokenizer.qwen3.loadTokenizer cfg.qwenDir
    let (tokIds, maskIds) := tokenizer.qwen3.encodePrompt tok prompt maxSeqLen
    let tokenCount := maskIds.foldl (fun acc v => if v != 0 then acc + 1 else acc) (0 : Nat)
    let tokens := (← IO.ofExcept (Tensor.fromInt64ArrayAs? (tokenizer.qwen3.toInt64Array tokIds) #[1, maxSeqLen]))
    let attnMask := (← IO.ofExcept (Tensor.fromInt64ArrayAs? (tokenizer.qwen3.toInt64Array maskIds) #[1, maxSeqLen]))
    pure (tokens, attnMask, tokenCount)

-- The derived cast for the nested VAE tree needs a 278-node instance dictionary.
set_option synthInstance.maxSize 512 in
/-- Main demo: generate an image from a text prompt -/
def runDemo (cfg : DemoConfig) (prompt : String) : IO Unit := do
  IO.println "=== Flux Klein 4B Demo ==="
  let ⟨hLatent⟩ ← if h : cfg.latentH = 16 ∧ cfg.latentW = 16 then pure (PLift.up h) else
    throw <| IO.userError s!"the VAE decode path expects 16x16 latents, got {cfg.latentH}x{cfg.latentW}"

  IO.println s!"Prompt: {prompt}"
  IO.println ""

  -- Load models
  IO.println "Loading models..."

  -- Load Qwen text encoder (sharded, with Flux Klein config)
  let qwenCfg := QwenConfig.fluxKleinTextEncoder
  let qwen ← loadQwenFluxEmbedderSharded cfg.qwenDir qwenCfg maxSeqLen #[8, 17, 26]
  IO.println "  ✓ Qwen text encoder loaded (Flux Klein variant)"

  -- Load Flux model
  let fluxCfg := FluxConfig.klein4B
  let ⟨_, flux⟩ ← loadFluxModel cfg.fluxPath fluxCfg
  let flux ← TensorStruct.castTo (FluxModel (device := .CPU) fluxCfg .Float32) flux
  let ⟨hContext⟩ ← if h : qwen.outputLayers.size * qwenCfg.hidden_size = fluxCfg.context_in_dim then pure (PLift.up h) else
    throw <| IO.userError
      s!"text encoder emits {qwen.outputLayers.size * qwenCfg.hidden_size} features, Flux expects {fluxCfg.context_in_dim}"
  IO.println "  ✓ Flux model loaded"

  -- Load VAE AutoEncoder (includes BN stats for proper normalization)
  let ⟨_, vae⟩ ← loadAutoEncoder cfg.vaePath
  let vae ← TensorStruct.castTo (AutoEncoder (device := .CPU) float32AutoEncoderDTypes) vae
  IO.println "  ✓ VAE AutoEncoder loaded"

  IO.println ""
  IO.println "Generating image..."

  -- Step 1: Tokenize and encode text
  IO.println "  [1/4] Encoding text..."
  let (tokens, attnMask, tokenCount) ← promptTokens cfg prompt
  let txtEmb := qwen.encodeMasked (batch := 1) (seq := maxSeqLen) (maskDtype := .Int64) qwenCfg maxSeqLen tokens attnMask
  let txtEmb : Tensor { shape := #[1, maxSeqLen, fluxCfg.context_in_dim], dtype := .Float32, device := .CPU } :=
    (txtEmb.reshape #[1, maxSeqLen, fluxCfg.context_in_dim] (by simp [hContext]))
  IO.println s!"    Text token count: {tokenCount}"
  IO.println s!"    Text embedding shape: {txtEmb.runtimeShape}"
  IO.println s!"    Text embedding dtype: {txtEmb.dtype}"

  -- Step 2: Prepare latent space
  IO.println "  [2/4] Preparing latent space..."
  let imgSeq := cfg.latentH * cfg.latentW
  let noiseSpatial ← randomWeight #[1, 128, cfg.latentH, cfg.latentW]
  let noiseSeq := Tensor.permute noiseSpatial #[0, 2, 3, 1]
  let noise : Tensor { shape := #[1, imgSeq, 128], dtype := .Float32, device := .CPU } := Tensor.reshape noiseSeq #[1, imgSeq, 128]

  -- Compute position IDs for RoPE
  let imgIds ← computeImagePositionIds cfg.latentH cfg.latentW
  let txtIds ← computeTextPositionIds maxSeqLen

  -- Step 3: Run diffusion denoising
  IO.println "  [3/4] Running Flux denoising..."
  let latents := denoise fluxCfg flux noise txtEmb imgIds txtIds cfg.numSteps
  IO.println s!"    Denoised with {cfg.numSteps} steps"
  -- Transform latents for VAE (inverse of flattening)
  let latentsHW : Tensor { shape := #[1, cfg.latentH, cfg.latentW, 128], dtype := .Float32, device := .CPU } :=
    Tensor.reshape latents #[1, cfg.latentH, cfg.latentW, 128]
      (by simp [imgSeq, fluxCfg, FluxConfig.klein4B, Nat.mul_assoc])
  let packedLatents : Tensor { shape := #[1, 128, cfg.latentH, cfg.latentW], dtype := .Float32, device := .CPU } :=
    (Tensor.permute latentsHW #[0, 3, 1, 2]).reshape #[1, 128, cfg.latentH, cfg.latentW]

  -- Step 4: Decode with VAE and save
  IO.println "  [4/4] Decoding with VAE..."

  -- Run VAE decode pipeline
  let z := vae.invNormalize (packedLatents.reshape #[1, 128, 16, 16] (by simp [hLatent.1, hLatent.2]))
  let z := torch.vae.AutoEncoder.unpackLatents16x16 z
  let image := vae.decoder.forward1 z
  saveImagePPM image cfg.outputPath
  IO.println s!"  ✓ Image saved to {cfg.outputPath}"


/-- Main entry point -/
def _root_.main (args : List String) : IO UInt32 := do
  if args == ["--help"] then
    IO.println "Usage: FluxDemo [prompt]\nUses local Flux, Qwen, and VAE checkpoints under weights/."
    return 0
  let prompt := match args with
    | arg :: _ => arg
    | [] => "a cat sitting on a windowsill, photorealistic"

  let cfg : DemoConfig := {
    qwenDir := "weights/flux-klein-4b/text_encoder"
    fluxPath := "weights/flux.safetensors"
    vaePath := "weights/ae.safetensors"
    outputPath := "output.ppm"
    useTokenFiles := false
    tokensPath := "weights/flux-klein-4b/tokenizer/tokens.txt"
    attnMaskPath := "weights/flux-klein-4b/tokenizer/attention_mask.txt"
    numSteps := 4
    latentH := 16
    latentW := 16
  }

  runDemo cfg prompt
  return 0

end Examples.Flux
