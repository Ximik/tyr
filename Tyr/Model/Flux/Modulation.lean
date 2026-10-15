/-
  Tyr/Model/Flux/Modulation.lean

  Adaptive modulation for Flux diffusion model.
  Computes scale/shift/gate parameters from timestep embedding.
-/
import Tyr.Model.Flux.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

namespace torch.flux

variable {device : Device} {dtype : DType}
/-- Create sinusoidal timestep embeddings.
    This is implemented as an extern function for efficiency.
    t: [batch] timestep tensor (values typically in [0, 1])
    dim: output dimension
    Returns: [batch, dim] positional embedding -/
def timestepEmbedding {batch : UInt64}
    (t : Tensor { device := device, shape := #[batch], dtype := .Float32 }) (dim : UInt64)
    (max_period : Float := 10000.0) (time_factor : Float := 1000.0)
    : Tensor { device := device, shape := #[batch, dim], dtype := .Float32 } :=
  match Tensor.ofRaw (RawTensor.timestepEmbedding batch t.raw dim max_period time_factor) with
  | .ok value => value
  | .error message => nativeInvariantFailure s!"Flux timestep embedding: {message}"

/-- Modulation output: shift, scale, gate triplets.
    For double blocks: 6 values (2 triplets for img and txt)
    For single blocks: 3 values (1 triplet) -/
structure ModulationOutput (batch num_mods hidden_size : UInt64) (dtype : DType := .Float32) where
  values : Tensor { dtype := dtype, device := device, shape := #[batch, num_mods, hidden_size] }

/-- Modulation layer: projects timestep embedding to shift/scale/gate.
    For double blocks: outputs 6 modulation vectors
    For single blocks: outputs 3 modulation vectors -/
structure Modulation (hidden_size : UInt64) (isDouble : Bool) (dtype : DType := .Float32) where
  /-- Linear projection from hidden_size to num_mods * hidden_size -/
  lin : Tensor { dtype := dtype, device := device, shape := #[if isDouble then 6 * hidden_size else 3 * hidden_size, hidden_size] }
  deriving TensorStruct


namespace Modulation

/-- Number of modulation outputs -/
def numMods (isDouble : Bool) : UInt64 :=
  if isDouble then 6 else 3

/-- Initialize modulation layer -/
def init (hidden_size : UInt64) (isDouble : Bool := false) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Modulation (device := device) (dtype := dtype) hidden_size isDouble) := do
  let outDim := if isDouble then 6 * hidden_size else 3 * hidden_size
  let std := Float.sqrt (2.0 / hidden_size.toFloat)
  let w ← randomWeight (dtype := dtype) (device := device) #[outDim, hidden_size]
  pure { lin := (← Tensor.parameter (Tensor.Preserving.mulScalar w std)) }

/-- Initialize modulation with zero weights (for dummy/placeholder modulation).
    Used when modulation is at model level instead of per-block. -/
def initZero (hidden_size : UInt64) (isDouble : Bool := false) (device : Device := .CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : Modulation (device := device) (dtype := dtype) hidden_size isDouble :=
  let outDim := if isDouble then 6 * hidden_size else 3 * hidden_size
  { lin := Tensor.detach (zerosWeight (dtype := dtype) (device := device) #[outDim, hidden_size]) }

/-- Forward pass: compute modulation from timestep embedding.
    Input vec: [batch, hidden_size] (timestep embedding)
    Output: [batch, num_mods, hidden_size] -/
def forward {batch hidden_size : UInt64} (isDouble : Bool)
    (mod : Modulation (device := device) (dtype := dtype) hidden_size isDouble)
    (vec : Tensor { dtype := dtype, device := device, shape := #[batch, hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, if isDouble then 6 else 3, hidden_size] } :=
  -- Apply SiLU activation first
  let vec := Tensor.Preserving.silu vec
  -- Project to modulation vectors
  let out := Tensor.linear vec mod.lin  -- [batch, num_mods * hidden_size]
  -- Reshape to [batch, num_mods, hidden_size]
  let numMods := if isDouble then 6 else 3
  Tensor.reshapeChecked out #[batch, numMods, hidden_size]

end Modulation

/-- Apply modulation to a tensor.
    x: [batch, seq, hidden_size]
    scale, shift: [batch, hidden_size]
    Returns: scale * x + shift -/
def applyModulation {batch seq hidden_size : UInt64}
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, hidden_size] })
    (scale : Tensor { dtype := dtype, device := device, shape := #[batch, hidden_size] })
    (shift : Tensor { dtype := dtype, device := device, shape := #[batch, hidden_size] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, hidden_size] } :=
  -- Expand scale and shift to match sequence dimension
  let scale := Tensor.unsqueeze scale 1  -- [batch, 1, hidden_size]
  let shift := Tensor.unsqueeze shift 1  -- [batch, 1, hidden_size]
  let scale := Tensor.expand scale #[batch, seq, hidden_size]
  let shift := Tensor.expand shift #[batch, seq, hidden_size]
  -- Apply: (1 + scale) * x + shift
  let scale := Tensor.Preserving.addScalar scale 1.0
  Tensor.Preserving.add (Tensor.Preserving.mul scale x) shift

end torch.flux
