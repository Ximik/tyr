/-
  Tyr/Model/Flux/ConfigIO.lean

  diffusers `transformer/config.json` loader for the Flux2 transformer.
-/
import Tyr.Model.Flux.Config
import Tyr.IO.ConfigJson

namespace torch.flux

open Lean ConfigJson

namespace FluxConfig

private def required {α : Type} [FromJson α] (j : Json) (key : String) : Except String α := do
  let some value ← field? j key | throw s!"Flux config: missing '{key}'"
  pure value

/-- Parse a diffusers `Flux2Transformer2DModel` config. Behavior the model
hard-codes (patch size 1, no guidance embedding, output channels equal to the
input channels) is required of the file. -/
def parseJson (j : Json) : Except String FluxConfig := do
  requireOneOf j "_class_name" ["Flux2Transformer2DModel"]
  requireEq j "patch_size" 1
  requireEq j "guidance_embeds" false
  let heads : Nat ← required j "num_attention_heads"
  let headDim : Nat ← required j "attention_head_dim"
  unless headDim % 2 == 0 do
    throw s!"Flux config: attention_head_dim={headDim} must be even"
  let inChannels : Nat ← required j "in_channels"
  match ← member? j "out_channels" with
  | none | some .null => pure ()
  | some value =>
    unless value == toJson inChannels do
      throw s!"Flux config: out_channels={value.compress} must be null or in_channels={inChannels}"
  let hidden := heads * headDim
  let mlpRatio : Float ← required j "mlp_ratio"
  let mlpHidden := (hidden.toFloat * mlpRatio).round.toUInt64.toNat
  unless mlpHidden.toFloat == hidden.toFloat * mlpRatio do
    throw s!"Flux config: mlp_ratio={mlpRatio} does not give a whole MLP width for hidden size {hidden}"
  let axes : Array Nat ← required j "axes_dims_rope"
  unless axes.all (· % 2 == 0) do
    throw s!"Flux config: axes_dims_rope={axes} must be even"
  let halfAxes := axes.map (· / 2)
  let ⟨hAxes⟩ ← if h : halfAxes.size = 4 ∧ halfAxes.toList.sum = headDim / 2 then pure (PLift.up h) else
    throw s!"Flux config: axes_dims_rope={axes} must be four axes summing to attention_head_dim={headDim}"
  let theta : Nat ← required j "rope_theta"
  let ⟨hTheta⟩ ← if h : 0 < theta then pure (PLift.up h) else
    throw "Flux config: rope_theta must be positive"
  pure {
    hidden_size := hidden
    num_heads := heads
    half_head_dim := headDim / 2
    num_double_layers := ← required j "num_layers"
    num_single_layers := ← required j "num_single_layers"
    context_in_dim := ← required j "joint_attention_dim"
    in_channels := inChannels
    time_dim := ← required j "timestep_guidance_channels"
    mlp_hidden_dim := mlpHidden
    theta
    eps := ← required j "eps"
    half_axes_dims := halfAxes
    axesValid := hAxes
    thetaPos := hTheta
  }

def loadFromFile (path : String) : IO FluxConfig := do
  IO.ofExcept ((parseJson (← parseFile path)).mapError (s!"{path}: " ++ ·))

/-- The transformer config of a diffusers model directory. -/
def loadFromPretrainedDir (modelDir : String) : IO FluxConfig :=
  loadFromFile s!"{modelDir}/transformer/config.json"

/-- A diffusers pipeline directory must describe the sampling `denoise`
implements: a step-distilled model (no classifier-free guidance) and a
`FlowMatchEulerDiscreteScheduler` whose sigmas are the pipeline's, shifted
exponentially by the `mu` it computes, with timesteps scaled by 1000. -/
def checkPipelineDir (modelDir : String) : IO Unit := do
  let index := s!"{modelDir}/model_index.json"
  let scheduler := s!"{modelDir}/scheduler/scheduler_config.json"
  let check (path : String) (body : Json → Except String Unit) : IO Unit := do
    IO.ofExcept ((body (← parseFile path)).mapError (s!"{path}: " ++ ·))
  let require {α : Type} [FromJson α] [BEq α] [ToString α] (j : Json) (key : String) (expected : α) :
      Except String Unit := do
    let value : α ← required j key
    unless value == expected do
      throw s!"unsupported '{key}' = {value}; tyr's Flux sampling implements {expected}"
  check index fun j => require j "is_distilled" true
  check scheduler fun j => do
    require j "_class_name" "FlowMatchEulerDiscreteScheduler"
    require j "num_train_timesteps" 1000
    require j "use_dynamic_shifting" true
    require j "time_shift_type" "exponential"
    for key in ["invert_sigmas", "stochastic_sampling", "use_karras_sigmas", "use_exponential_sigmas",
        "use_beta_sigmas"] do
      requireEq j key false
    requireEq j "use_flow_sigmas" false
    match ← member? j "shift_terminal" with
    | none | some .null => pure ()
    | some value => throw s!"unsupported 'shift_terminal' = {value.compress}; tyr's Flux sampling implements none"

end FluxConfig

end torch.flux
