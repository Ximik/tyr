/-
  Tyr/Model/SileroVAD/Model.lean

  Lean4 port of the Silero-VAD tiny model (`tinygrad_model.py`):
  - STFT conv front-end
  - Conv stack encoder
  - LSTM cell decoder
  - Final speech-probability head
-/
import Tyr.Typed
import Tyr.SafeTensors.Load
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Utils

namespace torch.silerovad

open torch.Model

private def sampleRate16k : UInt64 := 16000
private def chunkSizeSamples : Nat := 512
private def contextSizeSamples : Nat := 64
private def combinedInputSamples : Nat := chunkSizeSamples + contextSizeSamples
private def rightReflectPad : Nat := 64
private def preparedInputSamples : Nat := combinedInputSamples + rightReflectPad

private def addBias2d {n d : UInt64}
    (x : Tensor { shape := #[n, d], dtype := .Float32 }) (b : Tensor { shape := #[d], dtype := .Float32 })
    : Tensor { shape := #[n, d], dtype := .Float32 } :=
  x + (b.reshapeChecked #[1, d]).expand #[n, d]

private def addBias3d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (b : Tensor { shape := #[channels], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  x + (b.reshapeChecked #[1, channels, 1]).expand #[batch, channels, frames]

private def linear2d {batch inDim outDim : UInt64}
    (x : Tensor { shape := #[batch, inDim], dtype := .Float32 })
    (w : Tensor { shape := #[outDim, inDim], dtype := .Float32 })
    (b : Tensor { shape := #[outDim], dtype := .Float32 }) : Tensor { shape := #[batch, outDim], dtype := .Float32 } :=
  addBias2d (Tensor.linear x w) b

private def loadWeight (path name : String) (shape : Shape) : IO (Tensor { shape := shape, dtype := .Float32 }) := do
  let some tensor ← safetensors.tryLoadTensor path name shape
    | throw <| IO.userError s!"SileroVAD: failed to load {name}"
  pure tensor.toFloat32

private partial def reflectIndex (n : Int) (i : Int) : Int :=
  if n <= 1 then
    0
  else if i < 0 then
    reflectIndex n (-i)
  else if i >= n then
    reflectIndex n ((2 * n - 2) - i)
  else
    i

/-- Right-side reflect padding equivalent to `F.pad(x, (0, pad), mode="reflect")`. -/
def reflectPadRight (xs : Array Float) (pad : Nat) : Array Float :=
  if pad == 0 then
    xs
  else
    let n := xs.size
    if n == 0 then
      Array.replicate pad 0.0
    else if n == 1 then
      xs ++ Array.replicate pad xs[0]!
    else
      Id.run do
        let mut out := xs
        let nI : Int := Int.ofNat n
        for k in [:pad] do
          let srcI : Int := Int.ofNat (n + k)
          let idxI := reflectIndex nI srcI
          out := out.push (xs[Int.toNat idxI]!)
        out

/-- LSTM state for one-stream Silero-VAD inference (`batch=1`). -/
structure VADLstmState where
  h : Tensor { shape := #[1, 128], dtype := .Float32 }
  c : Tensor { shape := #[1, 128], dtype := .Float32 }
  deriving TensorStruct

namespace VADLstmState

def zero (device : Device := Device.CPU) : VADLstmState := {
  h := Tensor.zeros #[1, 128] false device
  c := Tensor.zeros #[1, 128] false device
}

end VADLstmState

/-- Lean module matching `TinySileroVAD` weight layout. -/
structure SileroVAD where
  stftConvWeight : Tensor { shape := #[258, 1, 256], dtype := .Float32 }
  conv1Weight : Tensor { shape := #[128, 129, 3], dtype := .Float32 }
  conv1Bias : Tensor { shape := #[128], dtype := .Float32 }
  conv2Weight : Tensor { shape := #[64, 128, 3], dtype := .Float32 }
  conv2Bias : Tensor { shape := #[64], dtype := .Float32 }
  conv3Weight : Tensor { shape := #[64, 64, 3], dtype := .Float32 }
  conv3Bias : Tensor { shape := #[64], dtype := .Float32 }
  conv4Weight : Tensor { shape := #[128, 64, 3], dtype := .Float32 }
  conv4Bias : Tensor { shape := #[128], dtype := .Float32 }
  lstmWeightIH : Tensor { shape := #[512, 128], dtype := .Float32 }
  lstmWeightHH : Tensor { shape := #[512, 128], dtype := .Float32 }
  lstmBiasIH : Tensor { shape := #[512], dtype := .Float32 }
  lstmBiasHH : Tensor { shape := #[512], dtype := .Float32 }
  finalConvWeight : Tensor { shape := #[1, 128, 1], dtype := .Float32 }
  finalConvBias : Tensor { shape := #[1], dtype := .Float32 }
  deriving TensorStruct

namespace SileroVAD

def init : IO SileroVAD := do
  let stftConvWeight ← initWeight #[258, 1, 256] 256
  let conv1Weight ← initWeight #[128, 129, 3] (129 * 3)
  let conv1Bias ← initBias #[128]
  let conv2Weight ← initWeight #[64, 128, 3] (128 * 3)
  let conv2Bias ← initBias #[64]
  let conv3Weight ← initWeight #[64, 64, 3] (64 * 3)
  let conv3Bias ← initBias #[64]
  let conv4Weight ← initWeight #[128, 64, 3] (64 * 3)
  let conv4Bias ← initBias #[128]
  let lstmWeightIH ← initWeight #[512, 128] 128
  let lstmWeightHH ← initWeight #[512, 128] 128
  let lstmBiasIH ← initBias #[512]
  let lstmBiasHH ← initBias #[512]
  let finalConvWeight ← initWeight #[1, 128, 1] 128
  let finalConvBias ← initBias #[1]
  pure {
    stftConvWeight
    conv1Weight
    conv1Bias
    conv2Weight
    conv2Bias
    conv3Weight
    conv3Bias
    conv4Weight
    conv4Bias
    lstmWeightIH
    lstmWeightHH
    lstmBiasIH
    lstmBiasHH
    finalConvWeight
    finalConvBias
  }

/-- Load pretrained Silero-VAD tiny safetensors checkpoint. -/
def load (path : String) : IO SileroVAD := do
  let stftConvWeight ← loadWeight path "stft_conv.weight" #[258, 1, 256]
  let conv1Weight ← loadWeight path "conv1.weight" #[128, 129, 3]
  let conv1Bias ← loadWeight path "conv1.bias" #[128]
  let conv2Weight ← loadWeight path "conv2.weight" #[64, 128, 3]
  let conv2Bias ← loadWeight path "conv2.bias" #[64]
  let conv3Weight ← loadWeight path "conv3.weight" #[64, 64, 3]
  let conv3Bias ← loadWeight path "conv3.bias" #[64]
  let conv4Weight ← loadWeight path "conv4.weight" #[128, 64, 3]
  let conv4Bias ← loadWeight path "conv4.bias" #[128]
  let lstmWeightIH ← loadWeight path "lstm_cell.weight_ih" #[512, 128]
  let lstmWeightHH ← loadWeight path "lstm_cell.weight_hh" #[512, 128]
  let lstmBiasIH ← loadWeight path "lstm_cell.bias_ih" #[512]
  let lstmBiasHH ← loadWeight path "lstm_cell.bias_hh" #[512]
  let finalConvWeight ← loadWeight path "final_conv.weight" #[1, 128, 1]
  let finalConvBias ← loadWeight path "final_conv.bias" #[1]
  pure {
    stftConvWeight := reqGradFalse stftConvWeight
    conv1Weight := reqGradFalse conv1Weight
    conv1Bias := reqGradFalse conv1Bias
    conv2Weight := reqGradFalse conv2Weight
    conv2Bias := reqGradFalse conv2Bias
    conv3Weight := reqGradFalse conv3Weight
    conv3Bias := reqGradFalse conv3Bias
    conv4Weight := reqGradFalse conv4Weight
    conv4Bias := reqGradFalse conv4Bias
    lstmWeightIH := reqGradFalse lstmWeightIH
    lstmWeightHH := reqGradFalse lstmWeightHH
    lstmBiasIH := reqGradFalse lstmBiasIH
    lstmBiasHH := reqGradFalse lstmBiasHH
    finalConvWeight := reqGradFalse finalConvWeight
    finalConvBias := reqGradFalse finalConvBias
  }

private def lstmCellStep
    (m : SileroVAD)
    (x : Tensor { shape := #[1, 128], dtype := .Float32 })
    (st : VADLstmState)
    : VADLstmState := Id.run do
  let gatesIH : Tensor { shape := #[1, 512], dtype := .Float32 } := linear2d x m.lstmWeightIH m.lstmBiasIH
  let gatesHH : Tensor { shape := #[1, 512], dtype := .Float32 } := linear2d st.h m.lstmWeightHH m.lstmBiasHH
  let gates : Tensor { shape := #[1, 512], dtype := .Float32 } := gatesIH + gatesHH

  let iGateRaw : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.slice gates 1 0 128
  let fGateRaw : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.slice gates 1 128 128
  let gGateRaw : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.slice gates 1 256 128
  let oGateRaw : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.slice gates 1 384 128

  let iGate : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.sigmoid iGateRaw
  let fGate : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.sigmoid fGateRaw
  let gGate : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.tanh gGateRaw
  let oGate : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.sigmoid oGateRaw

  let cNext : Tensor { shape := #[1, 128], dtype := .Float32 } := fGate * st.c + iGate * gGate
  let hNext : Tensor { shape := #[1, 128], dtype := .Float32 } := oGate * Tensor.tanh cNext
  { h := hNext, c := cNext }

/-- Core forward pass.
    Input must be one prepared chunk with shape `[1, 640]`:
    (`64` context + `512` chunk) with additional right reflect padding `64`. -/
def forwardPrepared
    (m : SileroVAD)
    (xPrepared : Tensor { shape := #[1, preparedInputSamples.toUInt64], dtype := .Float32 })
    (state : VADLstmState)
    : Tensor { shape := #[1, 1], dtype := .Float32 } × VADLstmState := Id.run do
  let x0 : Tensor { shape := #[1, 1, preparedInputSamples.toUInt64], dtype := .Float32 } :=
    Tensor.reshapeChecked xPrepared #[1, 1, preparedInputSamples.toUInt64]

  let stft : Tensor { shape := #[1, 258, 4], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.conv1d x0 m.stftConvWeight 128 0 1) #[1, 258, 4]
  let re : Tensor { shape := #[1, 129, 4], dtype := .Float32 } := Tensor.slice stft 1 0 129
  let im : Tensor { shape := #[1, 129, 4], dtype := .Float32 } := Tensor.slice stft 1 129 129
  let mag : Tensor { shape := #[1, 129, 4], dtype := .Float32 } := Tensor.sqrt (re * re + im * im)

  let h1 : Tensor { shape := #[1, 128, 4], dtype := .Float32 } :=
    Tensor.relu (addBias3d (Tensor.reshapeChecked (Tensor.conv1d mag m.conv1Weight 1 1 1) #[1, 128, 4]) m.conv1Bias)
  let h2 : Tensor { shape := #[1, 64, 2], dtype := .Float32 } :=
    Tensor.relu (addBias3d (Tensor.reshapeChecked (Tensor.conv1d h1 m.conv2Weight 2 1 1) #[1, 64, 2]) m.conv2Bias)
  let h3 : Tensor { shape := #[1, 64, 1], dtype := .Float32 } :=
    Tensor.relu (addBias3d (Tensor.reshapeChecked (Tensor.conv1d h2 m.conv3Weight 2 1 1) #[1, 64, 1]) m.conv3Bias)
  let h4 : Tensor { shape := #[1, 128, 1], dtype := .Float32 } :=
    Tensor.relu (addBias3d (Tensor.reshapeChecked (Tensor.conv1d h3 m.conv4Weight 1 1 1) #[1, 128, 1]) m.conv4Bias)

  let xFlat : Tensor { shape := #[1, 128], dtype := .Float32 } := Tensor.reshapeChecked h4 #[1, 128]
  let state' := lstmCellStep m xFlat state
  let hAct : Tensor { shape := #[1, 128, 1], dtype := .Float32 } := Tensor.reshapeChecked (Tensor.relu state'.h) #[1, 128, 1]

  let y0 : Tensor { shape := #[1, 1, 1], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.conv1d hAct m.finalConvWeight 1 0 1) #[1, 1, 1]
  let y1 : Tensor { shape := #[1, 1, 1], dtype := .Float32 } := addBias3d y0 m.finalConvBias
  let y2 : Tensor { shape := #[1, 1, 1], dtype := .Float32 } := Tensor.sigmoid y1
  let y : Tensor { shape := #[1, 1], dtype := .Float32 } := Tensor.reshapeChecked y2 #[1, 1]
  (y, state')

end SileroVAD

/-- Stateful streaming wrapper mirroring Silero-VAD chunk inference (`16k`, `512` samples). -/
structure SileroVADRuntime where
  model : SileroVAD
  state : VADLstmState
  context : Array Float := Array.replicate contextSizeSamples 0.0
  lastSamplingRate : UInt64 := sampleRate16k

namespace SileroVADRuntime

def init (model : SileroVAD) (device : Device := Device.CPU) : SileroVADRuntime := {
  model
  state := VADLstmState.zero device
  context := Array.replicate contextSizeSamples 0.0
  lastSamplingRate := sampleRate16k
}

def reset (rt : SileroVADRuntime) : SileroVADRuntime := {
  rt with
  state := VADLstmState.zero rt.state.h.device
  context := Array.replicate contextSizeSamples 0.0
  lastSamplingRate := sampleRate16k
}

private def appendRightZeroPad (xs : Array Float) (target : Nat) : Array Float :=
  if xs.size >= target then
    xs
  else
    xs ++ Array.replicate (target - xs.size) 0.0

private def splitEvery (xs : Array Float) (n : Nat) : Array (Array Float) :=
  if n == 0 then
    #[xs]
  else
    Id.run do
      let mut out : Array (Array Float) := #[]
      let mut i : Nat := 0
      while i < xs.size do
        let j := Nat.min xs.size (i + n)
        out := out.push (xs.extract i j)
        i := i + n
      out

private def prepareChunkTensor (context chunk : Array Float) : IO (Tensor { shape := #[1, preparedInputSamples.toUInt64], dtype := .Float32 } × Array Float) := do
  if context.size != contextSizeSamples then
    throw <| IO.userError
      s!"SileroVAD context size mismatch: expected {contextSizeSamples}, got {context.size}"
  if chunk.size != chunkSizeSamples then
    throw <| IO.userError
      s!"SileroVAD chunk size mismatch: expected {chunkSizeSamples}, got {chunk.size}"
  let combined := context ++ chunk
  if combined.size != combinedInputSamples then
    throw <| IO.userError
      s!"SileroVAD combined chunk size mismatch: expected {combinedInputSamples}, got {combined.size}"

  let prepared := reflectPadRight combined rightReflectPad
  if prepared.size != preparedInputSamples then
    throw <| IO.userError
      s!"SileroVAD prepared input size mismatch: expected {preparedInputSamples}, got {prepared.size}"

  let x : Tensor { shape := #[1, preparedInputSamples.toUInt64], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.fromFloatArray prepared) #[1, preparedInputSamples.toUInt64]
  let nextContext := combined.extract (combined.size - contextSizeSamples) combined.size
  pure (x, nextContext)

/-- One `512`-sample chunk inference step (`16kHz` only). -/
def step (rt : SileroVADRuntime) (chunk : Array Float) (samplingRate : UInt64 := sampleRate16k)
    : IO (Float × SileroVADRuntime) := do
  if samplingRate != sampleRate16k then
    throw <| IO.userError
      s!"SileroVAD runtime currently supports only {sampleRate16k}Hz, got {samplingRate}"

  let rt0 :=
    if rt.lastSamplingRate != samplingRate then
      rt.reset
    else
      rt
  let (x, nextContext) ← prepareChunkTensor rt0.context chunk
  let (out, nextState) := rt0.model.forwardPrepared x rt0.state
  let vals ← Tensor.toFloatArray out
  let prob := vals.getD 0 0.0
  pure (prob, {
    rt0 with
    state := nextState
    context := nextContext
    lastSamplingRate := samplingRate
  })

/-- Full-audio probability forward (one probability per 512-sample chunk). -/
def audioForward (rt : SileroVADRuntime) (audio : Array Float) (samplingRate : UInt64 := sampleRate16k)
    : IO (Array Float × SileroVADRuntime) := do
  if samplingRate != sampleRate16k then
    throw <| IO.userError
      s!"SileroVAD runtime currently supports only {sampleRate16k}Hz, got {samplingRate}"
  if audio.isEmpty then
    pure (#[], rt.reset)
  else
    let padded :=
      if audio.size % chunkSizeSamples == 0 then
        audio
      else
        appendRightZeroPad audio (audio.size + (chunkSizeSamples - (audio.size % chunkSizeSamples)))
    let chunks := splitEvery padded chunkSizeSamples
    let mut cur := rt.reset
    let mut probs : Array Float := #[]
    for chunk in chunks do
      let (p, cur') ← cur.step chunk samplingRate
      probs := probs.push p
      cur := cur'
    pure (probs, cur)

end SileroVADRuntime

end torch.silerovad
