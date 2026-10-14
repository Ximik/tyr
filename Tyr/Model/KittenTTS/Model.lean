/-
  Tyr/Model/KittenTTS/Model.lean

  Lean4 KittenTTS port focused on single-utterance inference.

  The text / duration / prosody / decoder / iSTFT vocoder stack mirrors the
  upstream Kokoro / StyleTTS2-style architecture closely enough for single
  utterance inference, including harmonic source generation and inverse STFT.
 -/
import Tyr.Model.KittenTTS.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.KittenTTS.Config
import Tyr.Model.Utils

namespace torch.kittentts

open torch.Model

private def initUniform (shape : Shape) (scale : Float) : IO (Tensor { shape := shape, dtype := .Float32 }) := do
  let w ← uniformTensor shape (-scale) scale
  Tensor.parameter w

private def xavierScale (dim : Nat) : Float :=
  1.0 / Float.sqrt (max 1 dim).toFloat

private def affineShift {s : Shape} (normed g b : Tensor { shape := s, dtype := .Float32 }) : Tensor { shape := s, dtype := .Float32 } :=
  (normed * Tensor.addScalar g 1.0) + b

private def addBias2d {batch dim : UInt64}
    (x : Tensor { shape := #[batch, dim], dtype := .Float32 })
    (b : Tensor { shape := #[dim], dtype := .Float32 })
    : Tensor { shape := #[batch, dim], dtype := .Float32 } :=
  x + Tensor.expand (Tensor.reshapeChecked b #[1, dim]) #[batch, dim]

private def addBias3d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (b : Tensor { shape := #[channels], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  let be : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked b #[1, channels, 1]) #[batch, channels, frames]
  x + be

private def seqToCF {batch seq channels : UInt64}
    (x : Tensor { shape := #[batch, seq, channels], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, seq], dtype := .Float32 } :=
  Tensor.reshapeChecked (Tensor.transpose x 1 2) #[batch, channels, seq]

private def cfToSeq {batch channels seq : UInt64}
    (x : Tensor { shape := #[batch, channels, seq], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, channels], dtype := .Float32 } :=
  Tensor.reshapeChecked (Tensor.transpose x 1 2) #[batch, seq, channels]

private def repeatNearest1d {batch channels frames scale : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, frames * scale], dtype := .Float32 } :=
  resizeTensor x (frames * scale) "nearest"

private def resize1d {batch channels inFrames outFrames : UInt64}
    (x : Tensor { shape := #[batch, channels, inFrames], dtype := .Float32 })
    (mode : String := "linear")
    : Tensor { shape := #[batch, channels, outFrames], dtype := .Float32 } :=
  resizeTensor x outFrames mode

private def layerNormCF {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (weight bias : Tensor { shape := #[channels], dtype := .Float32 })
    (eps : Float := 1e-5)
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  seqToCF (Tensor.layerNorm (cfToSeq x) weight bias eps)

private def instanceNormCF {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (eps : Float := 1e-5)
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  let mean : Tensor { shape := #[batch, channels, 1], dtype := .Float32 } := Tensor.Preserving.meanDim x 2 true
  let meanE : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := Tensor.expand mean #[batch, channels, frames]
  let centered : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := x - meanE
  let var : Tensor { shape := #[batch, channels, 1], dtype := .Float32 } := Tensor.Preserving.meanDim (centered * centered) 2 true
  let invStd : Tensor { shape := #[batch, channels, 1], dtype := .Float32 } := Tensor.rsqrt (var + eps)
  let invStdE : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := Tensor.expand invStd #[batch, channels, frames]
  centered * invStdE

private def styleExpand {batch dim frames : UInt64}
    (style : Tensor { shape := #[batch, dim], dtype := .Float32 })
    : Tensor { shape := #[batch, dim, frames], dtype := .Float32 } :=
  Tensor.expand (Tensor.reshapeChecked style #[batch, dim, 1]) #[batch, dim, frames]

private def snake1d {batch channels frames : UInt64}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (alpha : Tensor { shape := #[1, channels, 1], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  let a : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := Tensor.expand alpha #[batch, channels, frames]
  let inv : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } := Tensor.div (Tensor.ones #[batch, channels, frames] false x.device) a
  x + inv * Tensor.pow (Tensor.sin (x * a)) 2.0

private def geluApprox {s : Shape} (x : Tensor { shape := s, dtype := .Float32 }) : Tensor { shape := s, dtype := .Float32 } :=
  let cubic := Tensor.pow x 3.0
  let inner := x + Tensor.mulScalar cubic 0.044715
  let t := Tensor.tanh (Tensor.mulScalar inner 0.7978846)
  Tensor.mulScalar x 0.5 * Tensor.addScalar t 1.0

private def safeIndex {α} [Inhabited α] (xs : Array α) (idx : Nat) : α :=
  match xs[idx]? with
  | some x => x
  | none => default

private def durationToFrames (vals : Array Float) (speed : Float) : Array UInt64 :=
  vals.map fun v =>
    let denom := if speed <= 0.0 then 1.0 else speed
    let raw := ((v / denom) + 0.5).toUInt64
    if raw == 0 then 1 else raw

private def buildAlignment {seq frames : UInt64}
    (durations : Array UInt64)
    (device : Device := Device.CPU)
    : Tensor { shape := #[1, seq, frames], dtype := .Float32 } := Id.run do
  if frames == 0 then
    Tensor.reshapeChecked (Tensor.zeros #[1, seq, frames] false device) #[1, seq, frames]
  else
    let mut flat : Array Float := Array.replicate (seq * frames).toNat 0.0
    let mut cursor : Nat := 0
    for i in [:seq.toNat] do
      let dur := durations.getD i 1
      for _ in [:dur.toNat] do
        if cursor < frames.toNat then
          let idx := i * frames.toNat + cursor
          flat := flat.set! idx 1.0
          cursor := cursor + 1
    (Tensor.reshapeChecked (Tensor.fromFloatArray flat) #[1, seq, frames]).to device

/-- Linear layer with explicit weight / bias tensors. -/
structure LinearNorm (inDim outDim : UInt64) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32 }
  bias : Tensor { shape := #[outDim], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace LinearNorm

def init (inDim outDim : UInt64) : IO (LinearNorm inDim outDim) := do
  let scale := xavierScale outDim.toNat
  let weight ← initUniform #[outDim, inDim] scale
  let bias ← initBias #[outDim]
  pure { weight, bias }

def forward2d {batch inDim outDim : UInt64}
    (m : LinearNorm inDim outDim)
    (x : Tensor { shape := #[batch, inDim], dtype := .Float32 })
    : Tensor { shape := #[batch, outDim], dtype := .Float32 } :=
  affine2d x m.weight m.bias

def forward3d {batch seq inDim outDim : UInt64}
    (m : LinearNorm inDim outDim)
    (x : Tensor { shape := #[batch, seq, inDim], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, outDim], dtype := .Float32 } :=
  Tensor.affine3d x m.weight m.bias

end LinearNorm

structure Conv1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weight : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 }
  bias : Tensor { shape := #[outC], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace Conv1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (Conv1dParams inC outC kernel stride padding dilation) := do
  let weight ← initWeight #[outC, inC, kernel] (inC * kernel)
  let bias ← initBias #[outC]
  pure { weight, bias }

def forward {batch frames inC outC kernel stride padding dilation : UInt64}
    (m : Conv1dParams inC outC kernel stride padding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32 } :=
  let y0 : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.conv1d x m.weight stride padding dilation) #[batch, outC, convOutputSize frames kernel stride padding dilation]
  addBias3d y0 m.bias

end Conv1dParams

structure DepthwiseConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weight : Tensor { shape := #[channels, 1, kernel], dtype := .Float32 }
  bias : Tensor { shape := #[channels], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace DepthwiseConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (DepthwiseConv1dParams channels kernel stride padding dilation) := do
  let weight ← initWeight #[channels, 1, kernel] kernel
  let bias ← initBias #[channels]
  pure { weight, bias }

def forward {batch frames channels kernel stride padding dilation : UInt64}
    (m : DepthwiseConv1dParams channels kernel stride padding dilation)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, convOutputSize frames kernel stride padding dilation], dtype := .Float32 } :=
  Tensor.reshapeChecked
    (Tensor.conv1dGroups x m.weight m.bias stride padding dilation channels)
    #[batch, channels, convOutputSize frames kernel stride padding dilation]

end DepthwiseConv1dParams

structure DepthwiseTransConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1) where
  weight : Tensor { shape := #[channels, 1, kernel], dtype := .Float32 }
  bias : Tensor { shape := #[channels], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace DepthwiseTransConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1)
    : IO (DepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation) := do
  let weight ← initWeight #[channels, 1, kernel] kernel
  let bias ← initBias #[channels]
  pure { weight, bias }

def forward {batch frames channels kernel stride padding outputPadding dilation : UInt64}
    (m : DepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, convTransposeOutputSize frames kernel stride padding outputPadding dilation], dtype := .Float32 } := Id.run do
  let outFrames := convTransposeOutputSize frames kernel stride padding outputPadding dilation
  let mut parts : Array (ShapePackedTensor .Float32) := #[]
  for ci in [:channels.toNat] do
    let xc : Tensor { shape := #[batch, 1, frames], dtype := .Float32 } := sliceTensor x 1 ci.toUInt64 1
    let wc : Tensor { shape := #[1, 1, kernel], dtype := .Float32 } := sliceTensor m.weight 0 ci.toUInt64 1
    let bc : Tensor { shape := #[1], dtype := .Float32 } := sliceTensor m.bias 0 ci.toUInt64 1
    let yc : Tensor { shape := #[batch, 1, outFrames], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.convTranspose1d xc wc bc stride padding outputPadding dilation) #[batch, 1, outFrames]
    parts := parts.push (Tensor.packShape yc)
  Tensor.catManyChecked parts 1 #[batch, channels, outFrames]

end DepthwiseTransConv1dParams

private def normDim0_3d {d0 d1 d2 : UInt64}
    (x : Tensor { shape := #[d0, d1, d2], dtype := .Float32 })
    : Tensor { shape := #[d0], dtype := .Float32 } :=
  let sq : Tensor { shape := #[d0, d1, d2], dtype := .Float32 } := x * x
  let s2 : Tensor { shape := #[d0, d1, 1], dtype := .Float32 } := Tensor.sumDim sq 2 true
  let s1 : Tensor { shape := #[d0, 1, 1], dtype := .Float32 } := Tensor.sumDim s2 1 true
  Tensor.reshapeChecked (Tensor.sqrt (s1 + (1e-12 : Float))) #[d0]

private def applyWeightNormConv1d {outC inC kernel : UInt64}
    (weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 })
    (weightG : Tensor { shape := #[outC], dtype := .Float32 })
    : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 } :=
  let denom : Tensor { shape := #[outC, 1, 1], dtype := .Float32 } := Tensor.reshapeChecked (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 } := Tensor.expand denom #[outC, inC, kernel]
  let gain : Tensor { shape := #[outC, 1, 1], dtype := .Float32 } := Tensor.reshapeChecked weightG #[outC, 1, 1]
  let gainE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 } := Tensor.expand gain #[outC, inC, kernel]
  Tensor.div (weightV * gainE) denomE

private def applyWeightNormTransConv1d {inC outC kernel : UInt64}
    (weightV : Tensor { shape := #[inC, outC, kernel], dtype := .Float32 })
    (weightG : Tensor { shape := #[inC], dtype := .Float32 })
    : Tensor { shape := #[inC, outC, kernel], dtype := .Float32 } :=
  let denom : Tensor { shape := #[inC, 1, 1], dtype := .Float32 } := Tensor.reshapeChecked (normDim0_3d weightV) #[inC, 1, 1]
  let denomE : Tensor { shape := #[inC, outC, kernel], dtype := .Float32 } := Tensor.expand denom #[inC, outC, kernel]
  let gain : Tensor { shape := #[inC, 1, 1], dtype := .Float32 } := Tensor.reshapeChecked weightG #[inC, 1, 1]
  let gainE : Tensor { shape := #[inC, outC, kernel], dtype := .Float32 } := Tensor.expand gain #[inC, outC, kernel]
  Tensor.div (weightV * gainE) denomE

structure WNConv1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32 }
  weightG : Tensor { shape := #[outC], dtype := .Float32 }
  bias : Tensor { shape := #[outC], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace WNConv1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (WNConv1dParams inC outC kernel stride padding dilation) := do
  let weightV ← initWeight #[outC, inC, kernel] (inC * kernel)
  let weightG : Tensor { shape := #[outC], dtype := .Float32 } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding dilation : UInt64}
    (m : WNConv1dParams inC outC kernel stride padding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32 } :=
  let weight := applyWeightNormConv1d m.weightV m.weightG
  let y0 : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.conv1d x weight stride padding dilation) #[batch, outC, convOutputSize frames kernel stride padding dilation]
  addBias3d y0 m.bias

end WNConv1dParams

structure WNConvTranspose1dParams
    (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1) where
  weightV : Tensor { shape := #[inC, outC, kernel], dtype := .Float32 }
  weightG : Tensor { shape := #[inC], dtype := .Float32 }
  bias : Tensor { shape := #[outC], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace WNConvTranspose1dParams

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO (WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight #[inC, outC, kernel] (max 1 (outC * kernel).toNat |>.toUInt64)
  let weightG : Tensor { shape := #[inC], dtype := .Float32 } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding outputPadding dilation : UInt64}
    (m : WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation], dtype := .Float32 } :=
  let weight := applyWeightNormTransConv1d m.weightV m.weightG
  Tensor.reshapeChecked
    (Tensor.convTranspose1d x weight m.bias stride padding outputPadding dilation)
    #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation]

end WNConvTranspose1dParams

structure WNDepthwiseTransConv1dParams
    (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1) where
  weightV : Tensor { shape := #[channels, 1, kernel], dtype := .Float32 }
  weightG : Tensor { shape := #[channels], dtype := .Float32 }
  bias : Tensor { shape := #[channels], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace WNDepthwiseTransConv1dParams

def init (channels kernel : UInt64)
    (stride : UInt64 := 2)
    (padding : UInt64 := 1)
    (outputPadding : UInt64 := 1)
    (dilation : UInt64 := 1)
    : IO (WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight #[channels, 1, kernel] kernel
  let weightG : Tensor { shape := #[channels], dtype := .Float32 } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias #[channels]
  pure { weightV, weightG, bias }

def forward {batch frames channels kernel stride padding outputPadding dilation : UInt64}
    (m : WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, convTransposeOutputSize frames kernel stride padding outputPadding dilation], dtype := .Float32 } := Id.run do
  let outFrames := convTransposeOutputSize frames kernel stride padding outputPadding dilation
  let weight : Tensor { shape := #[channels, 1, kernel], dtype := .Float32 } := applyWeightNormTransConv1d m.weightV m.weightG
  let mut parts : Array (ShapePackedTensor .Float32) := #[]
  for ci in [:channels.toNat] do
    let xc : Tensor { shape := #[batch, 1, frames], dtype := .Float32 } := sliceTensor x 1 ci.toUInt64 1
    let wc : Tensor { shape := #[1, 1, kernel], dtype := .Float32 } := sliceTensor weight 0 ci.toUInt64 1
    let bc : Tensor { shape := #[1], dtype := .Float32 } := sliceTensor m.bias 0 ci.toUInt64 1
    let yc : Tensor { shape := #[batch, 1, outFrames], dtype := .Float32 } :=
      Tensor.reshapeChecked (Tensor.convTranspose1d xc wc bc stride padding outputPadding dilation) #[batch, 1, outFrames]
    parts := parts.push (Tensor.packShape yc)
  Tensor.catManyChecked parts 1 #[batch, channels, outFrames]

end WNDepthwiseTransConv1dParams

structure BiLSTM (inputSize hiddenSize : UInt64) where
  wxForward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32 }
  whForward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32 }
  biasIHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32 }
  biasHHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32 }
  wxBackward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32 }
  whBackward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32 }
  biasIHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32 }
  biasHHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace BiLSTM

def init (inputSize hiddenSize : UInt64) : IO (BiLSTM inputSize hiddenSize) := do
  let scale := xavierScale hiddenSize.toNat
  let wxForward ← initUniform #[4 * hiddenSize, inputSize] scale
  let whForward ← initUniform #[4 * hiddenSize, hiddenSize] scale
  let biasIHForward ← initUniform #[4 * hiddenSize] scale
  let biasHHForward ← initUniform #[4 * hiddenSize] scale
  let wxBackward ← initUniform #[4 * hiddenSize, inputSize] scale
  let whBackward ← initUniform #[4 * hiddenSize, hiddenSize] scale
  let biasIHBackward ← initUniform #[4 * hiddenSize] scale
  let biasHHBackward ← initUniform #[4 * hiddenSize] scale
  pure {
    wxForward, whForward, biasIHForward, biasHHForward,
    wxBackward, whBackward, biasIHBackward, biasHHBackward
  }

private def step
    {hidden input : UInt64}
    (x : Tensor { shape := #[1, input], dtype := .Float32 })
    (h c : Tensor { shape := #[1, hidden], dtype := .Float32 })
    (wx : Tensor { shape := #[4 * hidden, input], dtype := .Float32 })
    (wh : Tensor { shape := #[4 * hidden, hidden], dtype := .Float32 })
    (bih bhh : Tensor { shape := #[4 * hidden], dtype := .Float32 })
    : Tensor { shape := #[1, hidden], dtype := .Float32 } × Tensor { shape := #[1, hidden], dtype := .Float32 } :=
  let gatesIH : Tensor { shape := #[1, 4 * hidden], dtype := .Float32 } := affine2d x wx bih
  let gatesHH : Tensor { shape := #[1, 4 * hidden], dtype := .Float32 } := affine2d h wh bhh
  let gates : Tensor { shape := #[1, 4 * hidden], dtype := .Float32 } := gatesIH + gatesHH

  let iRaw : Tensor { shape := #[1, hidden], dtype := .Float32 } := sliceTensor gates 1 0 hidden
  let fRaw : Tensor { shape := #[1, hidden], dtype := .Float32 } := sliceTensor gates 1 hidden hidden
  let gRaw : Tensor { shape := #[1, hidden], dtype := .Float32 } := sliceTensor gates 1 (2 * hidden) hidden
  let oRaw : Tensor { shape := #[1, hidden], dtype := .Float32 } := sliceTensor gates 1 (3 * hidden) hidden

  let iGate : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.sigmoid iRaw
  let fGate : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.sigmoid fRaw
  let gGate : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.tanh gRaw
  let oGate : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.sigmoid oRaw

  let cNext : Tensor { shape := #[1, hidden], dtype := .Float32 } := fGate * c + iGate * gGate
  let hNext : Tensor { shape := #[1, hidden], dtype := .Float32 } := oGate * Tensor.tanh cNext
  (hNext, cNext)

private def runDirection {seq input hidden : UInt64}
    (x : Tensor { shape := #[1, seq, input], dtype := .Float32 })
    (wx : Tensor { shape := #[4 * hidden, input], dtype := .Float32 })
    (wh : Tensor { shape := #[4 * hidden, hidden], dtype := .Float32 })
    (bih bhh : Tensor { shape := #[4 * hidden], dtype := .Float32 })
    (reverse : Bool := false)
    : Tensor { shape := #[1, seq, hidden], dtype := .Float32 } := Id.run do
  let mut h : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.zeros #[1, hidden] false x.device
  let mut c : Tensor { shape := #[1, hidden], dtype := .Float32 } := Tensor.zeros #[1, hidden] false x.device
  let mut outs : Array (ShapePackedTensor .Float32) := #[]
  let indices :=
    if reverse then
      (List.range seq.toNat).reverse
    else
      List.range seq.toNat
  for idx in indices do
    let xt3 : Tensor { shape := #[1, 1, input], dtype := .Float32 } := sliceTensor x 1 idx.toUInt64 1
    let xt : Tensor { shape := #[1, input], dtype := .Float32 } := Tensor.reshapeChecked xt3 #[1, input]
    let (hNext, cNext) := step xt h c wx wh bih bhh
    h := hNext
    c := cNext
    let pushed : Tensor { shape := #[1, 1, hidden], dtype := .Float32 } := Tensor.reshapeChecked hNext #[1, 1, hidden]
    outs := outs.push (Tensor.packShape pushed)
  let ordered := if reverse then outs.reverse else outs
  Tensor.catManyChecked ordered 1 #[1, seq, hidden]

def forward {seq input hidden : UInt64}
    (m : BiLSTM input hidden)
    (x : Tensor { shape := #[1, seq, input], dtype := .Float32 })
    : Tensor { shape := #[1, seq, 2 * hidden], dtype := .Float32 } :=
  let fw := runDirection x m.wxForward m.whForward m.biasIHForward m.biasHHForward false
  let bw := runDirection x m.wxBackward m.whBackward m.biasIHBackward m.biasHHBackward true
  (catTo fw bw 2).reshapeChecked #[1, seq, 2 * hidden]

end BiLSTM

structure AlbertEmbeddings (nToken : UInt64) (cfg : AlbertConfig) where
  wordEmbeddings : Tensor { shape := #[nToken, cfg.embeddingSize], dtype := .Float32 }
  positionEmbeddings : Tensor { shape := #[cfg.maxPositionEmbeddings, cfg.embeddingSize], dtype := .Float32 }
  tokenTypeEmbeddings : Tensor { shape := #[cfg.typeVocabSize, cfg.embeddingSize], dtype := .Float32 }
  layerNormWeight : Tensor { shape := #[cfg.embeddingSize], dtype := .Float32 }
  layerNormBias : Tensor { shape := #[cfg.embeddingSize], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace AlbertEmbeddings

def init (nToken : UInt64) (cfg : AlbertConfig) : IO (AlbertEmbeddings nToken cfg) := do
  let scale := xavierScale cfg.embeddingSize.toNat
  let wordEmbeddings ← initUniform #[nToken, cfg.embeddingSize] scale
  let positionEmbeddings ← initUniform #[cfg.maxPositionEmbeddings, cfg.embeddingSize] scale
  let tokenTypeEmbeddings ← initUniform #[cfg.typeVocabSize, cfg.embeddingSize] scale
  let layerNormWeight ← initUniform #[cfg.embeddingSize] 0.02
  let layerNormBias ← initBias #[cfg.embeddingSize]
  pure { wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNormWeight, layerNormBias }

def forward {seq nToken : UInt64}
    (m : AlbertEmbeddings nToken cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32 } :=
  let tokTypes : Tensor { shape := #[1, seq], dtype := .Int64 } := Tensor.fullInt #[1, seq] 0
  let posIds1d : Tensor { shape := #[seq], dtype := .Int64 } := Tensor.arange seq
  let posIds : Tensor { shape := #[1, seq], dtype := .Int64 } := Tensor.reshapeChecked posIds1d #[1, seq]
  let word : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32 } := Tensor.embedding inputIds m.wordEmbeddings
  let pos : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32 } := Tensor.embedding posIds m.positionEmbeddings
  let typ : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32 } := Tensor.embedding tokTypes m.tokenTypeEmbeddings
  Tensor.layerNorm (word + pos + typ) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end AlbertEmbeddings

structure KittenAlbertSelfAttention (cfg : AlbertConfig) where
  query : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  key : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  value : LinearNorm cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  dense : LinearNorm (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  layerNormWeight : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32 }
  layerNormBias : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace KittenAlbertSelfAttention

def init (cfg : AlbertConfig) : IO (KittenAlbertSelfAttention cfg) := do
  let query ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let key ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let value ← LinearNorm.init cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let dense ← LinearNorm.init (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  let layerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let layerNormBias ← initBias #[cfg.hiddenSize]
  pure { query, key, value, dense, layerNormWeight, layerNormBias }

def forward {seq : UInt64}
    (m : KittenAlbertSelfAttention cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } :=
  let q0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32 } := m.query.forward3d x
  let k0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32 } := m.key.forward3d x
  let v0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32 } := m.value.forward3d x

  let q : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32 } :=
    Tensor.reshapeChecked q0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let k : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32 } :=
    Tensor.reshapeChecked k0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let v : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32 } :=
    Tensor.reshapeChecked v0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]

  let qh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32 } := Tensor.splitHeads q
  let kh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32 } := Tensor.splitHeads k
  let vh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32 } := Tensor.splitHeads v
  let attn : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32 } :=
    Tensor.attentionGQA qh kh vh 0.0 false true
  let out4 : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32 } := Tensor.joinHeads attn
  let out3 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32 } :=
    Tensor.reshapeChecked out4 #[1, seq, AlbertConfig.allHeadSize cfg]
  let denseOut : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } := m.dense.forward3d out3
  Tensor.layerNorm (denseOut + x) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end KittenAlbertSelfAttention

structure KittenAlbertLayer (cfg : AlbertConfig) where
  attention : KittenAlbertSelfAttention cfg
  fullLayerNormWeight : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32 }
  fullLayerNormBias : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32 }
  ffn : LinearNorm cfg.hiddenSize cfg.intermediateSize
  ffnOutput : LinearNorm cfg.intermediateSize cfg.hiddenSize
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayer

def init (cfg : AlbertConfig) : IO (KittenAlbertLayer cfg) := do
  let attention ← KittenAlbertSelfAttention.init cfg
  let fullLayerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let fullLayerNormBias ← initBias #[cfg.hiddenSize]
  let ffn ← LinearNorm.init cfg.hiddenSize cfg.intermediateSize
  let ffnOutput ← LinearNorm.init cfg.intermediateSize cfg.hiddenSize
  pure { attention, fullLayerNormWeight, fullLayerNormBias, ffn, ffnOutput }

def forward {seq : UInt64}
    (m : KittenAlbertLayer cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } :=
  let attnOut := m.attention.forward x
  let f0 : Tensor { shape := #[1, seq, cfg.intermediateSize], dtype := .Float32 } := m.ffn.forward3d attnOut
  let f1 := geluApprox f0
  let f2 : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } := m.ffnOutput.forward3d f1
  Tensor.layerNorm (f2 + attnOut) m.fullLayerNormWeight m.fullLayerNormBias cfg.layerNormEps

end KittenAlbertLayer

structure KittenAlbertLayerGroup (cfg : AlbertConfig) where
  layers : Array (KittenAlbertLayer cfg)
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayerGroup

def init (cfg : AlbertConfig) : IO (KittenAlbertLayerGroup cfg) := do
  let mut layers : Array (KittenAlbertLayer cfg) := #[]
  for _ in [:cfg.innerGroupNum.toNat] do
    layers := layers.push (← KittenAlbertLayer.init cfg)
  pure { layers }

def forward {seq : UInt64}
    (m : KittenAlbertLayerGroup cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } :=
  m.layers.foldl (fun h layer => layer.forward h) x

end KittenAlbertLayerGroup

structure KittenAlbertEncoder (cfg : AlbertConfig) where
  embeddingHiddenMappingIn : LinearNorm cfg.embeddingSize cfg.hiddenSize
  groups : Array (KittenAlbertLayerGroup cfg)
  deriving TensorStruct, Inhabited

namespace KittenAlbertEncoder

def init (cfg : AlbertConfig) : IO (KittenAlbertEncoder cfg) := do
  let embeddingHiddenMappingIn ← LinearNorm.init cfg.embeddingSize cfg.hiddenSize
  let mut groups : Array (KittenAlbertLayerGroup cfg) := #[]
  for _ in [:cfg.numHiddenGroups.toNat] do
    groups := groups.push (← KittenAlbertLayerGroup.init cfg)
  pure { embeddingHiddenMappingIn, groups }

def forward {seq : UInt64}
    (m : KittenAlbertEncoder cfg)
    (x : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32 })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } := Id.run do
  let mapped : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } := m.embeddingHiddenMappingIn.forward3d x
  let layersPerGroup :=
    if cfg.numHiddenGroups == 0 then 1 else max 1 (cfg.numHiddenLayers / cfg.numHiddenGroups)
  let mut h := mapped
  for i in [:cfg.numHiddenLayers.toNat] do
    let groupIdx := if cfg.numHiddenGroups == 0 then 0 else i / layersPerGroup.toNat
    let group := safeIndex m.groups groupIdx
    h := group.forward h
  h

end KittenAlbertEncoder

structure KittenAlbert (nToken : UInt64) (cfg : AlbertConfig) where
  embeddings : AlbertEmbeddings nToken cfg
  encoder : KittenAlbertEncoder cfg
  pooler : LinearNorm cfg.hiddenSize cfg.hiddenSize
  deriving TensorStruct, Inhabited

namespace KittenAlbert

def init (nToken : UInt64) (cfg : AlbertConfig) : IO (KittenAlbert nToken cfg) := do
  let embeddings ← AlbertEmbeddings.init nToken cfg
  let encoder ← KittenAlbertEncoder.init cfg
  let pooler ← LinearNorm.init cfg.hiddenSize cfg.hiddenSize
  pure { embeddings, encoder, pooler }

def forward {seq nToken : UInt64}
    (m : KittenAlbert nToken cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32 } × Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32 } :=
  let emb := m.embeddings.forward inputIds
  let h := m.encoder.forward emb
  let cls3 : Tensor { shape := #[1, 1, cfg.hiddenSize], dtype := .Float32 } := sliceTensor h 1 0 1
  let cls2 : Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32 } := Tensor.reshapeChecked cls3 #[1, cfg.hiddenSize]
  let pooled : Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32 } := Tensor.tanh (m.pooler.forward2d cls2)
  (h, pooled)

end KittenAlbert

structure TextConvLayer (cfg : KittenTTSConfig) where
  conv : Conv1dParams cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  lnWeight : Tensor { shape := #[cfg.hiddenDim], dtype := .Float32 }
  lnBias : Tensor { shape := #[cfg.hiddenDim], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace TextConvLayer

def init (cfg : KittenTTSConfig) : IO (TextConvLayer cfg) := do
  let conv ← Conv1dParams.init cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  let lnWeight ← initUniform #[cfg.hiddenDim] 0.02
  let lnBias ← initBias #[cfg.hiddenDim]
  pure { conv, lnWeight, lnBias }

def forward {seq : UInt64}
    (m : TextConvLayer cfg)
    (x : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 })
    : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } :=
  let y : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := (m.conv.forward x).reshapeChecked #[1, cfg.hiddenDim, seq]
  let y := layerNormCF y m.lnWeight m.lnBias
  leakyRelu y 0.2

end TextConvLayer

structure TextEncoder (cfg : KittenTTSConfig) where
  embedding : Tensor { shape := #[cfg.nToken, cfg.hiddenDim], dtype := .Float32 }
  convs : Array (TextConvLayer cfg)
  lstm : BiLSTM cfg.hiddenDim (cfg.hiddenDim / 2)
  deriving TensorStruct, Inhabited

namespace TextEncoder

def init (cfg : KittenTTSConfig) : IO (TextEncoder cfg) := do
  let scale := xavierScale cfg.hiddenDim.toNat
  let embedding ← initUniform #[cfg.nToken, cfg.hiddenDim] scale
  let mut convs : Array (TextConvLayer cfg) := #[]
  for _ in [:cfg.nLayer.toNat] do
    convs := convs.push (← TextConvLayer.init cfg)
  let lstm ← BiLSTM.init cfg.hiddenDim (cfg.hiddenDim / 2)
  pure { embedding, convs, lstm }

def forward {seq : UInt64}
    (m : TextEncoder cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := Id.run do
  let emb : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := Tensor.embedding inputIds m.embedding
  let mut h : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := seqToCF emb
  for conv in m.convs do
    h := conv.forward h
  let hSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := cfToSeq h
  let outSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := (m.lstm.forward hSeq).reshapeChecked #[1, seq, cfg.hiddenDim]
  seqToCF outSeq

end TextEncoder

structure AdaLayerNorm (styleDim channels : UInt64) where
  fc : LinearNorm styleDim (2 * channels)
  deriving TensorStruct, Inhabited

namespace AdaLayerNorm

def init (styleDim channels : UInt64) : IO (AdaLayerNorm styleDim channels) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  pure { fc }

def forward {batch frames styleDim channels : UInt64}
    (m : AdaLayerNorm styleDim channels)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32 })
    (style : Tensor { shape := #[batch, styleDim], dtype := .Float32 })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32 } :=
  let h : Tensor { shape := #[batch, 2 * channels], dtype := .Float32 } := m.fc.forward2d style
  let gamma : Tensor { shape := #[batch, 1, channels], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 0 channels) #[batch, 1, channels]
  let beta : Tensor { shape := #[batch, 1, channels], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 channels channels) #[batch, 1, channels]
  let xSeq : Tensor { shape := #[batch, frames, channels], dtype := .Float32 } := cfToSeq x
  let normed : Tensor { shape := #[batch, frames, channels], dtype := .Float32 } := Tensor.layerNorm xSeq (Tensor.ones #[channels] false x.device) (Tensor.zeros #[channels] false x.device) 1e-5
  let g : Tensor { shape := #[batch, frames, channels], dtype := .Float32 } := Tensor.expand gamma #[batch, frames, channels]
  let b : Tensor { shape := #[batch, frames, channels], dtype := .Float32 } := Tensor.expand beta #[batch, frames, channels]
  seqToCF (affineShift normed g b)

end AdaLayerNorm

structure AdaIN1d (styleDim channels : UInt64) where
  fc : LinearNorm styleDim (2 * channels)
  normWeight : Tensor { shape := #[channels], dtype := .Float32 }
  normBias : Tensor { shape := #[channels], dtype := .Float32 }
  deriving TensorStruct, Inhabited

namespace AdaIN1d

def init (styleDim channels : UInt64) : IO (AdaIN1d styleDim channels) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  let normWeight ← Tensor.parameter (Tensor.ones #[channels] false Device.CPU)
  let normBias ← initBias #[channels]
  pure { fc, normWeight, normBias }

def forward {frames styleDim channels : UInt64}
    (m : AdaIN1d styleDim channels)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, channels, frames], dtype := .Float32 } :=
  let h : Tensor { shape := #[1, 2 * channels], dtype := .Float32 } := m.fc.forward2d style
  let gamma : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 0 channels) #[1, channels, 1]
  let beta : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 channels channels) #[1, channels, 1]
  let baseNorm := instanceNormCF x
  let normW : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked m.normWeight #[1, channels, 1]) #[1, channels, frames]
  let normB : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked m.normBias #[1, channels, 1]) #[1, channels, frames]
  let normed : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := baseNorm * normW + normB
  let g : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand gamma #[1, channels, frames]
  let b : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand beta #[1, channels, frames]
  affineShift normed g b

end AdaIN1d

structure DurationEncoder (cfg : KittenTTSConfig) where
  lstms : Array (BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
  norms : Array (AdaLayerNorm cfg.styleDim cfg.hiddenDim)
  deriving TensorStruct, Inhabited

namespace DurationEncoder

def init (cfg : KittenTTSConfig) : IO (DurationEncoder cfg) := do
  let mut lstms : Array (BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)) := #[]
  let mut norms : Array (AdaLayerNorm cfg.styleDim cfg.hiddenDim) := #[]
  for _ in [:cfg.nLayer.toNat] do
    lstms := lstms.push (← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
    norms := norms.push (← AdaLayerNorm.init cfg.styleDim cfg.hiddenDim)
  pure { lstms, norms }

def forward {seq : UInt64}
    (m : DurationEncoder cfg)
    (x : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 } := Id.run do
  let s : Tensor { shape := #[1, cfg.styleDim, seq], dtype := .Float32 } := styleExpand style
  let mut h : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 } := (catTo x s 1).reshapeChecked #[1, cfg.hiddenDim + cfg.styleDim, seq]
  for i in [:m.lstms.size] do
    let lstm := safeIndex m.lstms i
    let norm := safeIndex m.norms i
    let ySeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := (BiLSTM.forward lstm (cfToSeq h)).reshapeChecked #[1, seq, cfg.hiddenDim]
    let yCF : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := seqToCF ySeq
    let yNorm : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := AdaLayerNorm.forward norm yCF style
    h := catTo yNorm s 1
  h

end DurationEncoder

structure AdainResBlk1dSame (inC outC styleDim : UInt64) where
  conv1 : WNConv1dParams inC outC 3 1 1 1
  conv2 : WNConv1dParams outC outC 3 1 1 1
  norm1 : AdaIN1d styleDim inC
  norm2 : AdaIN1d styleDim outC
  shortcut : Option (WNConv1dParams inC outC 1 1 0 1) := none
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dSame

def init (inC outC styleDim : UInt64) : IO (AdainResBlk1dSame inC outC styleDim) := do
  let conv1 ← WNConv1dParams.init inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init styleDim inC
  let norm2 ← AdaIN1d.init styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure (some (← WNConv1dParams.init inC outC 1 1 0 1))
  pure { conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : UInt64}
    (m : AdainResBlk1dSame inC outC styleDim)
    (x : Tensor { shape := #[1, inC, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, outC, frames], dtype := .Float32 } :=
  let shortcut : Tensor { shape := #[1, outC, frames], dtype := .Float32 } :=
    match m.shortcut with
    | some sc => (sc.forward x).reshapeChecked #[1, outC, frames]
    | none => Tensor.reshapeChecked x #[1, outC, frames]
  let h0 : Tensor { shape := #[1, inC, frames], dtype := .Float32 } := m.norm1.forward x style
  let h1 : Tensor { shape := #[1, inC, frames], dtype := .Float32 } := leakyRelu h0 0.2
  let h2 : Tensor { shape := #[1, outC, frames], dtype := .Float32 } := (m.conv1.forward h1).reshapeChecked #[1, outC, frames]
  let h3 : Tensor { shape := #[1, outC, frames], dtype := .Float32 } := m.norm2.forward h2 style
  let h4 : Tensor { shape := #[1, outC, frames], dtype := .Float32 } := leakyRelu h3 0.2
  let h5 : Tensor { shape := #[1, outC, frames], dtype := .Float32 } := (m.conv2.forward h4).reshapeChecked #[1, outC, frames]
  (h5 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dSame

structure AdainResBlk1dUp (inC outC styleDim : UInt64) where
  pool : WNDepthwiseTransConv1dParams inC 3 2 1 1 1
  conv1 : WNConv1dParams inC outC 3 1 1 1
  conv2 : WNConv1dParams outC outC 3 1 1 1
  norm1 : AdaIN1d styleDim inC
  norm2 : AdaIN1d styleDim outC
  shortcut : Option (WNConv1dParams inC outC 1 1 0 1) := none
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dUp

def init (inC outC styleDim : UInt64) : IO (AdainResBlk1dUp inC outC styleDim) := do
  let pool ← WNDepthwiseTransConv1dParams.init inC 3 2 1 1 1
  let conv1 ← WNConv1dParams.init inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init styleDim inC
  let norm2 ← AdaIN1d.init styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure (some (← WNConv1dParams.init inC outC 1 1 0 1))
  pure { pool, conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : UInt64}
    (m : AdainResBlk1dUp inC outC styleDim)
    (x : Tensor { shape := #[1, inC, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } :=
  let shortcut0 : Tensor { shape := #[1, inC, 2 * frames], dtype := .Float32 } := (repeatNearest1d (scale := 2) x).reshapeChecked #[1, inC, 2 * frames]
  let shortcut : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } :=
    match m.shortcut with
    | some sc => (sc.forward shortcut0).reshapeChecked #[1, outC, 2 * frames]
    | none => Tensor.reshapeChecked shortcut0 #[1, outC, 2 * frames]
  let h0 : Tensor { shape := #[1, inC, frames], dtype := .Float32 } := m.norm1.forward x style
  let h1 : Tensor { shape := #[1, inC, frames], dtype := .Float32 } := leakyRelu h0 0.2
  let h2 : Tensor { shape := #[1, inC, 2 * frames], dtype := .Float32 } := (m.pool.forward h1).reshapeChecked #[1, inC, 2 * frames]
  let h3 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } := (m.conv1.forward h2).reshapeChecked #[1, outC, 2 * frames]
  let h4 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } := m.norm2.forward h3 style
  let h5 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } := leakyRelu h4 0.2
  let h6 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32 } := (m.conv2.forward h5).reshapeChecked #[1, outC, 2 * frames]
  (h6 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dUp

structure AdaINResBlock1 (channels kernel : UInt64) (styleDim : UInt64) where
  convs1 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1)
  convs2 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1)
  norms1 : Array (AdaIN1d styleDim channels)
  norms2 : Array (AdaIN1d styleDim channels)
  alpha1 : Array (Tensor { shape := #[1, channels, 1], dtype := .Float32 })
  alpha2 : Array (Tensor { shape := #[1, channels, 1], dtype := .Float32 })
  deriving TensorStruct, Inhabited

namespace AdaINResBlock1

def init (channels kernel styleDim : UInt64) : IO (AdaINResBlock1 channels kernel styleDim) := do
  let mut convs1 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut convs2 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut norms1 : Array (AdaIN1d styleDim channels) := #[]
  let mut norms2 : Array (AdaIN1d styleDim channels) := #[]
  let mut alpha1 : Array (Tensor { shape := #[1, channels, 1], dtype := .Float32 }) := #[]
  let mut alpha2 : Array (Tensor { shape := #[1, channels, 1], dtype := .Float32 }) := #[]
  for _ in [:3] do
    convs1 := convs1.push (← Conv1dParams.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    convs2 := convs2.push (← Conv1dParams.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    norms1 := norms1.push (← AdaIN1d.init styleDim channels)
    norms2 := norms2.push (← AdaIN1d.init styleDim channels)
    alpha1 := alpha1.push (Tensor.reshapeChecked (Tensor.ones #[channels] false Device.CPU) #[1, channels, 1])
    alpha2 := alpha2.push (Tensor.reshapeChecked (Tensor.ones #[channels] false Device.CPU) #[1, channels, 1])
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

def forward {frames channels kernel styleDim : UInt64}
    (m : AdaINResBlock1 channels kernel styleDim)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Id.run do
  let mut h := x
  for i in [:3] do
    let a1 := safeIndex m.alpha1 i
    let a2 := safeIndex m.alpha2 i
    let h1 := AdaIN1d.forward (safeIndex m.norms1 i) h style
    let h2 := snake1d h1 a1
    let h3 : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.reshapeChecked (Conv1dParams.forward (safeIndex m.convs1 i) h2) #[1, channels, frames]
    let h4 := AdaIN1d.forward (safeIndex m.norms2 i) h3 style
    let h5 := snake1d h4 a2
    let h6 : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.reshapeChecked (Conv1dParams.forward (safeIndex m.convs2 i) h5) #[1, channels, frames]
    h := h + h6
  h

end AdaINResBlock1

structure ProsodyPredictor (cfg : KittenTTSConfig) where
  durationEncoder : DurationEncoder cfg
  lstm : BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  durationProj : LinearNorm cfg.hiddenDim cfg.maxDur
  shared : BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  f0Blk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  f0Blk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  f0Blk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  nBlk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  nBlk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  nBlk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  f0Proj : Conv1dParams (cfg.hiddenDim / 2) 1 1 1 0 1
  nProj : Conv1dParams (cfg.hiddenDim / 2) 1 1 1 0 1
  deriving TensorStruct

namespace ProsodyPredictor

def init (cfg : KittenTTSConfig) : IO (ProsodyPredictor cfg) := do
  let durationEncoder ← DurationEncoder.init cfg
  let lstm ← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let durationProj ← LinearNorm.init cfg.hiddenDim cfg.maxDur
  let shared ← BiLSTM.init (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let f0Blk0 ← AdainResBlk1dSame.init cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let f0Blk1 ← AdainResBlk1dUp.init cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let f0Blk2 ← AdainResBlk1dSame.init (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk0 ← AdainResBlk1dSame.init cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let nBlk1 ← AdainResBlk1dUp.init cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk2 ← AdainResBlk1dSame.init (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let f0Proj ← Conv1dParams.init (cfg.hiddenDim / 2) 1 1 1 0 1
  let nProj ← Conv1dParams.init (cfg.hiddenDim / 2) 1 1 1 0 1
  pure {
    durationEncoder, lstm, durationProj, shared,
    f0Blk0, f0Blk1, f0Blk2, nBlk0, nBlk1, nBlk2, f0Proj, nProj
  }

def durationEncoding {seq : UInt64}
    (m : ProsodyPredictor cfg)
    (dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 } :=
  m.durationEncoder.forward dEn style

def durationLogits {seq : UInt64}
    (m : ProsodyPredictor cfg)
    (enc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 })
    : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32 } :=
  let hSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := (m.lstm.forward (cfToSeq enc)).reshapeChecked #[1, seq, cfg.hiddenDim]
  m.durationProj.forward3d hSeq

def forwardF0N {frames : UInt64}
    (m : ProsodyPredictor cfg)
    (enc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } × Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } :=
  let sharedSeq : Tensor { shape := #[1, frames, cfg.hiddenDim], dtype := .Float32 } := (m.shared.forward (cfToSeq enc)).reshapeChecked #[1, frames, cfg.hiddenDim]
  let sharedCF : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := seqToCF sharedSeq

  let f0a : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := m.f0Blk0.forward sharedCF style
  let f0b : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32 } := m.f0Blk1.forward f0a style
  let f0c : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32 } := m.f0Blk2.forward f0b style
  let f0 : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := (m.f0Proj.forward f0c).reshapeChecked #[1, 1, 2 * frames]

  let na : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := m.nBlk0.forward sharedCF style
  let nb : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32 } := m.nBlk1.forward na style
  let nc : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32 } := m.nBlk2.forward nb style
  let n : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := (m.nProj.forward nc).reshapeChecked #[1, 1, 2 * frames]

  (f0, n)

end ProsodyPredictor

private def reflectPadLeft1 {channels frames : UInt64}
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32 })
    : Tensor { shape := #[1, channels, frames + 1], dtype := .Float32 } :=
  let pad : Tensor { shape := #[1, channels, 1], dtype := .Float32 } :=
    if frames <= 1 then
      sliceTensor x 2 0 1
    else
      sliceTensor x 2 1 1
  Tensor.catManyChecked #[Tensor.packShape pad, Tensor.packShape x] 2 #[1, channels, frames + 1]

private def harmonicMultipliers (harmonicCount : UInt64) : Tensor { shape := #[harmonicCount + 1], dtype := .Float32 } := Id.run do
  let mut vals : Array Float := #[]
  for i in [: (harmonicCount + 1).toNat] do
    vals := vals.push (i.toFloat + 1.0)
  Tensor.reshapeChecked (Tensor.fromFloatArray vals) #[harmonicCount + 1]

structure AdaIN1dDyn (styleDim : UInt64) where
  fcWeight : ShapePackedTensor .Float32
  fcBias : ShapePackedTensor .Float32
  normWeight : ShapePackedTensor .Float32
  normBias : ShapePackedTensor .Float32
  deriving TensorStruct, Inhabited

namespace AdaIN1dDyn

def init (styleDim channels : UInt64) : IO (AdaIN1dDyn styleDim) := do
  let fc ← LinearNorm.init styleDim (2 * channels)
  let normWeight : Tensor { shape := #[channels], dtype := .Float32 } ← Tensor.parameter (Tensor.ones #[channels] false Device.CPU)
  let normBias ← initBias #[channels]
  pure {
    fcWeight := Tensor.packShape fc.weight
    fcBias := Tensor.packShape fc.bias
    normWeight := Tensor.packShape normWeight
    normBias := Tensor.packShape normBias
  }

def forward {frames styleDim channels : UInt64}
    (m : AdaIN1dDyn styleDim)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, channels, frames], dtype := .Float32 } :=
  let weight : Tensor { shape := #[2 * channels, styleDim], dtype := .Float32 } := Tensor.reshapeChecked m.fcWeight.2 #[2 * channels, styleDim]
  let bias : Tensor { shape := #[2 * channels], dtype := .Float32 } := Tensor.reshapeChecked m.fcBias.2 #[2 * channels]
  let h : Tensor { shape := #[1, 2 * channels], dtype := .Float32 } := affine2d style weight bias
  let gamma : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 0 channels) #[1, channels, 1]
  let beta : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (sliceTensor h 1 channels channels) #[1, channels, 1]
  let normWeight : Tensor { shape := #[channels], dtype := .Float32 } := Tensor.reshapeChecked m.normWeight.2 #[channels]
  let normBias : Tensor { shape := #[channels], dtype := .Float32 } := Tensor.reshapeChecked m.normBias.2 #[channels]
  let baseNorm := instanceNormCF x
  let normW : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked normWeight #[1, channels, 1]) #[1, channels, frames]
  let normB : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked normBias #[1, channels, 1]) #[1, channels, frames]
  let normed : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := baseNorm * normW + normB
  let g : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand gamma #[1, channels, frames]
  let b : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Tensor.expand beta #[1, channels, frames]
  affineShift normed g b

end AdaIN1dDyn

structure WNConv1dDyn where
  weightV : ShapePackedTensor .Float32
  weightG : ShapePackedTensor .Float32
  bias : ShapePackedTensor .Float32
  kernel : UInt64
  stride : UInt64 := 1
  padding : UInt64 := 0
  dilation : UInt64 := 1
  deriving TensorStruct, Inhabited

namespace WNConv1dDyn

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO WNConv1dDyn := do
  let weightV ← initWeight #[outC, inC, kernel] (inC * kernel)
  let weightG : Tensor { shape := #[outC], dtype := .Float32 } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias #[outC]
  pure {
    weightV := Tensor.packShape weightV
    weightG := Tensor.packShape weightG
    bias := Tensor.packShape bias
    kernel, stride, padding, dilation
  }

def forward {batch frames inC outC : UInt64}
    (m : WNConv1dDyn)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation], dtype := .Float32 } :=
  let weightV : Tensor { shape := #[outC, inC, m.kernel], dtype := .Float32 } := Tensor.reshapeChecked m.weightV.2 #[outC, inC, m.kernel]
  let weightG : Tensor { shape := #[outC], dtype := .Float32 } := Tensor.reshapeChecked m.weightG.2 #[outC]
  let bias : Tensor { shape := #[outC], dtype := .Float32 } := Tensor.reshapeChecked m.bias.2 #[outC]
  let weight := applyWeightNormConv1d weightV weightG
  let y0 : Tensor { shape := #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation], dtype := .Float32 } :=
    Tensor.reshapeChecked (Tensor.conv1d x weight m.stride m.padding m.dilation) #[batch, outC, convOutputSize frames m.kernel m.stride m.padding m.dilation]
  addBias3d y0 bias

end WNConv1dDyn

structure WNConvTranspose1dDyn where
  weightV : ShapePackedTensor .Float32
  weightG : ShapePackedTensor .Float32
  bias : ShapePackedTensor .Float32
  kernel : UInt64
  stride : UInt64 := 1
  padding : UInt64 := 0
  outputPadding : UInt64 := 0
  dilation : UInt64 := 1
  deriving TensorStruct, Inhabited

namespace WNConvTranspose1dDyn

def init (inC outC kernel : UInt64)
    (stride : UInt64 := 1)
    (padding : UInt64 := 0)
    (outputPadding : UInt64 := 0)
    (dilation : UInt64 := 1)
    : IO WNConvTranspose1dDyn := do
  let fanIn := if outC * kernel == 0 then 1 else outC * kernel
  let weightV ← initWeight #[inC, outC, kernel] fanIn
  let weightG : Tensor { shape := #[inC], dtype := .Float32 } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias #[outC]
  pure {
    weightV := Tensor.packShape weightV
    weightG := Tensor.packShape weightG
    bias := Tensor.packShape bias
    kernel, stride, padding, outputPadding, dilation
  }

def forward {batch frames inC outC : UInt64}
    (m : WNConvTranspose1dDyn)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32 })
    : Tensor { shape := #[batch, outC, convTransposeOutputSize frames m.kernel m.stride m.padding m.outputPadding m.dilation], dtype := .Float32 } :=
  let weightV : Tensor { shape := #[inC, outC, m.kernel], dtype := .Float32 } := Tensor.reshapeChecked m.weightV.2 #[inC, outC, m.kernel]
  let weightG : Tensor { shape := #[inC], dtype := .Float32 } := Tensor.reshapeChecked m.weightG.2 #[inC]
  let bias : Tensor { shape := #[outC], dtype := .Float32 } := Tensor.reshapeChecked m.bias.2 #[outC]
  let weight := applyWeightNormTransConv1d weightV weightG
  Tensor.reshapeChecked
    (Tensor.convTranspose1d x weight bias m.stride m.padding m.outputPadding m.dilation)
    #[batch, outC, convTransposeOutputSize frames m.kernel m.stride m.padding m.outputPadding m.dilation]

end WNConvTranspose1dDyn

structure GeneratorAdaINResBlock (styleDim : UInt64) where
  convs1 : Array WNConv1dDyn
  convs2 : Array WNConv1dDyn
  norms1 : Array (AdaIN1dDyn styleDim)
  norms2 : Array (AdaIN1dDyn styleDim)
  alpha1 : Array (ShapePackedTensor .Float32)
  alpha2 : Array (ShapePackedTensor .Float32)
  deriving TensorStruct, Inhabited

namespace GeneratorAdaINResBlock

def init (styleDim channels kernel : UInt64) (dilations : Array UInt64) : IO (GeneratorAdaINResBlock styleDim) := do
  let mut convs1 : Array WNConv1dDyn := #[]
  let mut convs2 : Array WNConv1dDyn := #[]
  let mut norms1 : Array (AdaIN1dDyn styleDim) := #[]
  let mut norms2 : Array (AdaIN1dDyn styleDim) := #[]
  let mut alpha1 : Array (ShapePackedTensor .Float32) := #[]
  let mut alpha2 : Array (ShapePackedTensor .Float32) := #[]
  for i in [:3] do
    let dilation := safeIndex dilations i
    let padding := (kernel * dilation - dilation) / 2
    convs1 := convs1.push (← WNConv1dDyn.init channels channels kernel 1 padding dilation)
    convs2 := convs2.push (← WNConv1dDyn.init channels channels kernel 1 ((kernel - 1) / 2) 1)
    norms1 := norms1.push (← AdaIN1dDyn.init styleDim channels)
    norms2 := norms2.push (← AdaIN1dDyn.init styleDim channels)
    alpha1 := alpha1.push (Tensor.packShape (Tensor.reshapeChecked (Tensor.ones #[channels] false Device.CPU) #[1, channels, 1]))
    alpha2 := alpha2.push (Tensor.packShape (Tensor.reshapeChecked (Tensor.ones #[channels] false Device.CPU) #[1, channels, 1]))
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

def forward {frames channels styleDim : UInt64}
    (m : GeneratorAdaINResBlock styleDim)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32 })
    : Tensor { shape := #[1, channels, frames], dtype := .Float32 } := Id.run do
  let mut h := x
  for i in [:3] do
    let a1 : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (safeIndex m.alpha1 i).2 #[1, channels, 1]
    let a2 : Tensor { shape := #[1, channels, 1], dtype := .Float32 } := Tensor.reshapeChecked (safeIndex m.alpha2 i).2 #[1, channels, 1]
    let h1 := AdaIN1dDyn.forward (safeIndex m.norms1 i) h style
    let h2 := snake1d h1 a1
    let h3 : Tensor { shape := #[1, channels, frames], dtype := .Float32 } :=
      Tensor.reshapeChecked (WNConv1dDyn.forward (inC := channels) (outC := channels) (safeIndex m.convs1 i) h2) #[1, channels, frames]
    let h4 := AdaIN1dDyn.forward (safeIndex m.norms2 i) h3 style
    let h5 := snake1d h4 a2
    let h6 : Tensor { shape := #[1, channels, frames], dtype := .Float32 } :=
      Tensor.reshapeChecked (WNConv1dDyn.forward (inC := channels) (outC := channels) (safeIndex m.convs2 i) h5) #[1, channels, frames]
    h := h + h6
  h

end GeneratorAdaINResBlock

structure SourceModuleHnNSF (cfg : KittenTTSConfig) where
  linear : LinearNorm (cfg.generator.harmonicCount + 1) 1
  deriving TensorStruct

namespace SourceModuleHnNSF

private def sineAmp : Float := 0.1
private def noiseStd : Float := 0.003
private def voicedThreshold : Float := 10.0

def init (cfg : KittenTTSConfig) : IO (SourceModuleHnNSF cfg) := do
  let linear ← LinearNorm.init (cfg.generator.harmonicCount + 1) 1
  pure { linear }

def forward {baseFrames : UInt64}
    (m : SourceModuleHnNSF cfg)
    (f0 : Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 })
    : IO
      (Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 } ×
       Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 } ×
       Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 }) := do
  let waveFrames := GeneratorConfig.waveSamples cfg.generator baseFrames
  let harmonics := cfg.generator.harmonicCount + 1
  let mult : Tensor { shape := #[1, 1, harmonics], dtype := .Float32 } := Tensor.reshapeChecked (harmonicMultipliers cfg.generator.harmonicCount) #[1, 1, harmonics]
  let multE : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := Tensor.expand mult #[1, waveFrames, harmonics]
  let f0E : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := Tensor.expand f0 #[1, waveFrames, harmonics]
  let rad0 : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := (f0E * multE) / cfg.sampleRate.toFloat
  let radBase : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := rad0 - floorTensor rad0

  let randIni : Tensor { shape := #[1, harmonics], dtype := .Float32 } ← randTensor #[1, harmonics] false f0.device
  let randIni : Tensor { shape := #[1, harmonics], dtype := .Float32 } :=
    Tensor.sliceScatter randIni 1 0 (Tensor.zeros #[1, 1] false f0.device)
  let phaseNoise0 : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := Tensor.zeros #[1, waveFrames, harmonics] false f0.device
  let phaseNoise : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } :=
    Tensor.sliceScatter phaseNoise0 1 0 (Tensor.reshapeChecked randIni #[1, 1, harmonics])
  let radInit : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := radBase + phaseNoise

  let radCF : Tensor { shape := #[1, harmonics, waveFrames], dtype := .Float32 } := seqToCF radInit
  let radDownCF : Tensor { shape := #[1, harmonics, baseFrames], dtype := .Float32 } := resize1d (outFrames := baseFrames) radCF "linear"
  let radDown : Tensor { shape := #[1, baseFrames, harmonics], dtype := .Float32 } := cfToSeq radDownCF
  let phaseBase : Tensor { shape := #[1, baseFrames, harmonics], dtype := .Float32 } := Tensor.mulScalar (cumsumTensor radDown 1) 6.283185307179586
  let phaseScaledCF : Tensor { shape := #[1, harmonics, baseFrames], dtype := .Float32 } :=
    seqToCF phaseBase * (GeneratorConfig.sourceUpsample cfg.generator).toFloat
  let phaseUpCF : Tensor { shape := #[1, harmonics, waveFrames], dtype := .Float32 } := resize1d (outFrames := waveFrames) phaseScaledCF "linear"
  let sines : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := Tensor.sin (cfToSeq phaseUpCF)
  let sineWaves : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := sines * sineAmp

  let threshold : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32 } := Tensor.full #[1, waveFrames, 1] voicedThreshold false f0.device
  let uv : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32 } := Tensor.toFloat32 (Tensor.lt threshold f0)
  let uvE : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := Tensor.expand uv #[1, waveFrames, harmonics]
  let noiseAmp : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } :=
    uvE * noiseStd + (Tensor.onesLike uvE - uvE) * (sineAmp / 3.0)
  let noise : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } ← Tensor.randn #[1, waveFrames, harmonics] false f0.device
  let sineNoisy : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32 } := sineWaves * uvE + noiseAmp * noise

  let sineMerge : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32 } := Tensor.tanh (m.linear.forward3d sineNoisy)
  let noiseOut : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32 } ← Tensor.randn #[1, waveFrames, 1] false f0.device
  pure (sineMerge, noiseOut * (sineAmp / 3.0), uv)

end SourceModuleHnNSF

structure GeneratorStage (styleDim : UInt64) where
  up : WNConvTranspose1dDyn
  noiseConv : WNConv1dDyn
  noiseRes : GeneratorAdaINResBlock styleDim
  resBlocks : Array (GeneratorAdaINResBlock styleDim)
  deriving TensorStruct, Inhabited

structure GeneratorStageDebugOutput where
  xSource : ShapePackedTensor .Float32
  xUp : ShapePackedTensor .Float32
  xMix : ShapePackedTensor .Float32
  xOut : ShapePackedTensor .Float32
  deriving TensorStruct, Inhabited

structure GeneratorDebugOutput where
  harSource : ShapePackedTensor .Float32
  har : ShapePackedTensor .Float32
  stages : Array GeneratorStageDebugOutput
  post : ShapePackedTensor .Float32
  specLog : ShapePackedTensor .Float32
  phaseRaw : ShapePackedTensor .Float32
  spec : ShapePackedTensor .Float32
  audio : ShapePackedTensor .Float32
  deriving TensorStruct, Inhabited

structure Generator (cfg : KittenTTSConfig) where
  source : SourceModuleHnNSF cfg
  stages : Array (GeneratorStage cfg.styleDim)
  convPost : WNConv1dDyn
  window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32 }
  deriving TensorStruct

namespace Generator

private def buildHarFeatures {baseFrames : UInt64}
    (m : Generator cfg)
    (harmonic : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 })
    : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
  let window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32 } := m.window.to harmonic.device
  let stftDyn : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32 } :=
    stftTensor
      (n := GeneratorConfig.waveSamples cfg.generator baseFrames)
      harmonic
      cfg.generator.genIstftNFft
      cfg.generator.genIstftHopSize
      window
      true
      false
  let packed : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32 } :=
    Tensor.reshapeChecked stftDyn #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let re3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    sliceTensor packed 2 0 1
  let im3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    sliceTensor packed 2 1 1
  let re : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked re3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let im : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked im3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let mag : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.sqrt (re * re + im * im + (1e-12 : Float))
  let phase : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    atan2Tensor im re
  let mag3 : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked mag #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let phase3 : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked phase #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  (catTo mag3 phase3 1).reshapeChecked #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]

def init (cfg : KittenTTSConfig) : IO (Generator cfg) := do
  let source ← SourceModuleHnNSF.init cfg
  let mut stages : Array (GeneratorStage cfg.styleDim) := #[]
  for i in [:cfg.generator.upsampleRates.size] do
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let stride := safeIndex cfg.generator.upsampleRates i
    let kernel := safeIndex cfg.generator.upsampleKernelSizes i
    let up ← WNConvTranspose1dDyn.init inC outC kernel stride ((kernel - stride) / 2) 0 1
    let noiseConv ←
      WNConv1dDyn.init
        (GeneratorConfig.stftFeatureChannels cfg.generator)
        outC
        (GeneratorConfig.noiseKernel cfg.generator i)
        (GeneratorConfig.noiseStride cfg.generator i)
        (GeneratorConfig.noisePadding cfg.generator i)
        1
    let noiseKernel := if i + 1 < cfg.generator.upsampleRates.size then 7 else 11
    let noiseRes ← GeneratorAdaINResBlock.init cfg.styleDim outC noiseKernel #[1, 3, 5]
    let mut resBlocks : Array (GeneratorAdaINResBlock cfg.styleDim) := #[]
    for j in [:cfg.generator.resblockKernelSizes.size] do
      let blockKernel := safeIndex cfg.generator.resblockKernelSizes j
      let blockDilations := safeIndex cfg.generator.resblockDilationSizes j
      resBlocks := resBlocks.push (← GeneratorAdaINResBlock.init cfg.styleDim outC blockKernel blockDilations)
    stages := stages.push { up, noiseConv, noiseRes, resBlocks }
  let convPost ←
    WNConv1dDyn.init
      (GeneratorConfig.finalChannels cfg.generator)
      (cfg.generator.genIstftNFft + 2)
      7
      1
      3
      1
  let window := Tensor.hannWindow cfg.generator.genIstftNFft
  pure { source, stages, convPost, window }

def debugForward {baseFrames : UInt64}
    (m : Generator cfg)
    (x : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, baseFrames], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    (f0Curve : Tensor { shape := #[1, 1, baseFrames], dtype := .Float32 })
    : IO GeneratorDebugOutput := do
  let f0Up : Tensor { shape := #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    repeatNearest1d (scale := GeneratorConfig.sourceUpsample cfg.generator) f0Curve
  let f0Seq : Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 } := cfToSeq f0Up
  let (harSource, _noiseSource, _uv) ← m.source.forward f0Seq
  let har1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked harSource #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  let har : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    buildHarFeatures m har1d

  let mut xDyn : ShapePackedTensor .Float32 := Tensor.packShape x
  let mut stageOutputs : Array GeneratorStageDebugOutput := #[]
  for i in [:m.stages.size] do
    let stage := safeIndex m.stages i
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let inFrames := GeneratorConfig.stageInFrames cfg.generator baseFrames i
    let outFrames := GeneratorConfig.stageOutFrames cfg.generator baseFrames i
    let xIn : Tensor { shape := #[1, inC, inFrames], dtype := .Float32 } := Tensor.reshapeChecked xDyn.2 #[1, inC, inFrames]
    let xAct : Tensor { shape := #[1, inC, inFrames], dtype := .Float32 } := leakyRelu xIn 0.1
    let xSource0 : Tensor { shape := #[1, outC, convOutputSize (GeneratorConfig.harFrames cfg.generator baseFrames) stage.noiseConv.kernel stage.noiseConv.stride stage.noiseConv.padding stage.noiseConv.dilation], dtype := .Float32 } :=
      stage.noiseConv.forward har
    let xSource1 : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := Tensor.reshapeChecked xSource0 #[1, outC, outFrames]
    let xSource : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := stage.noiseRes.forward xSource1 style
    let xUp0 : Tensor { shape := #[1, outC, convTransposeOutputSize inFrames stage.up.kernel stage.up.stride stage.up.padding stage.up.outputPadding stage.up.dilation], dtype := .Float32 } :=
      stage.up.forward xAct
    let xUp : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } :=
      if i + 1 < m.stages.size then
        Tensor.reshapeChecked xUp0 #[1, outC, outFrames]
      else
        (reflectPadLeft1 xUp0).reshapeChecked #[1, outC, outFrames]
    let xMix : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := xUp + xSource
    let first : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := (safeIndex stage.resBlocks 0).forward xMix style
    let mut acc := first
    for j in [1:stage.resBlocks.size] do
      acc := acc + (safeIndex stage.resBlocks j).forward xMix style
    let xOut : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := acc / stage.resBlocks.size.toFloat
    stageOutputs := stageOutputs.push {
      xSource := Tensor.packShape xSource
      xUp := Tensor.packShape xUp
      xMix := Tensor.packShape xMix
      xOut := Tensor.packShape xOut
    }
    xDyn := Tensor.packShape xOut

  let xFinal : Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked xDyn.2 #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let postIn : Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := leakyRelu xFinal 0.01
  let post : Tensor { shape := #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked
      (WNConv1dDyn.forward
        (inC := GeneratorConfig.finalChannels cfg.generator)
        (outC := cfg.generator.genIstftNFft + 2)
        m.convPost
        postIn)
      #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames]

  let freqBins := GeneratorConfig.freqBins cfg.generator
  let specLog : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    sliceTensor post 1 0 freqBins
  let phaseRaw : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    sliceTensor post 1 freqBins freqBins
  let spec : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := Tensor.exp specLog
  let phase : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := Tensor.sin phaseRaw
  let re : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := spec * Tensor.cos phase
  let im : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := spec * Tensor.sin phase
  let re3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    Tensor.reshapeChecked re #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let im3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    Tensor.reshapeChecked im #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let packed : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32 } :=
    Tensor.catManyChecked #[Tensor.packShape re3, Tensor.packShape im3] 2
      #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32 } := m.window.to post.device
  let audio1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked
      (istftTensor
        packed
        cfg.generator.genIstftNFft
        cfg.generator.genIstftHopSize
        cfg.generator.genIstftNFft
        window
        true
        false
        (GeneratorConfig.waveSamples cfg.generator baseFrames))
      #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  pure {
    harSource := Tensor.packShape harSource
    har := Tensor.packShape har
    stages := stageOutputs
    post := Tensor.packShape post
    specLog := Tensor.packShape specLog
    phaseRaw := Tensor.packShape phaseRaw
    spec := Tensor.packShape spec
    audio := Tensor.packShape (Tensor.reshapeChecked audio1d #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames])
  }

def forward {baseFrames : UInt64}
    (m : Generator cfg)
    (x : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, baseFrames], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    (f0Curve : Tensor { shape := #[1, 1, baseFrames], dtype := .Float32 })
    : IO (ShapePackedTensor .Float32) := do
  let f0Up : Tensor { shape := #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    repeatNearest1d (scale := GeneratorConfig.sourceUpsample cfg.generator) f0Curve
  let f0Seq : Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32 } := cfToSeq f0Up
  let (harSource, _noiseSource, _uv) ← m.source.forward f0Seq
  let har1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked harSource #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  let har : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    buildHarFeatures m har1d

  let mut xDyn : ShapePackedTensor .Float32 := Tensor.packShape x
  for i in [:m.stages.size] do
    let stage := safeIndex m.stages i
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let inFrames := GeneratorConfig.stageInFrames cfg.generator baseFrames i
    let outFrames := GeneratorConfig.stageOutFrames cfg.generator baseFrames i
    let xIn : Tensor { shape := #[1, inC, inFrames], dtype := .Float32 } := Tensor.reshapeChecked xDyn.2 #[1, inC, inFrames]
    let xAct : Tensor { shape := #[1, inC, inFrames], dtype := .Float32 } := leakyRelu xIn 0.1
    let xSource0 : Tensor { shape := #[1, outC, convOutputSize (GeneratorConfig.harFrames cfg.generator baseFrames) stage.noiseConv.kernel stage.noiseConv.stride stage.noiseConv.padding stage.noiseConv.dilation], dtype := .Float32 } :=
      stage.noiseConv.forward har
    let xSource1 : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := Tensor.reshapeChecked xSource0 #[1, outC, outFrames]
    let xSource : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := stage.noiseRes.forward xSource1 style
    let xUp0 : Tensor { shape := #[1, outC, convTransposeOutputSize inFrames stage.up.kernel stage.up.stride stage.up.padding stage.up.outputPadding stage.up.dilation], dtype := .Float32 } :=
      stage.up.forward xAct
    let xUp : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } :=
      if i + 1 < m.stages.size then
        Tensor.reshapeChecked xUp0 #[1, outC, outFrames]
      else
        (reflectPadLeft1 xUp0).reshapeChecked #[1, outC, outFrames]
    let xMix : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := xUp + xSource
    let first : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := (safeIndex stage.resBlocks 0).forward xMix style
    let mut acc := first
    for j in [1:stage.resBlocks.size] do
      acc := acc + (safeIndex stage.resBlocks j).forward xMix style
    let xOut : Tensor { shape := #[1, outC, outFrames], dtype := .Float32 } := acc / stage.resBlocks.size.toFloat
    xDyn := Tensor.packShape xOut

  let xFinal : Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked xDyn.2 #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let postIn : Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := leakyRelu xFinal 0.01
  let post : Tensor { shape := #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked
      (WNConv1dDyn.forward
        (inC := GeneratorConfig.finalChannels cfg.generator)
        (outC := cfg.generator.genIstftNFft + 2)
        m.convPost
        postIn)
      #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames]

  let freqBins := GeneratorConfig.freqBins cfg.generator
  let specLog : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    sliceTensor post 1 0 freqBins
  let phaseRaw : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } :=
    sliceTensor post 1 freqBins freqBins
  let spec : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := Tensor.exp specLog
  let phase : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := Tensor.sin phaseRaw
  let re : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := spec * Tensor.cos phase
  let im : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32 } := spec * Tensor.sin phase
  let re3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    Tensor.reshapeChecked re #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let im3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32 } :=
    Tensor.reshapeChecked im #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let packed : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32 } :=
    Tensor.catManyChecked #[Tensor.packShape re3, Tensor.packShape im3] 2
      #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32 } := m.window.to post.device
  let audio1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32 } :=
    Tensor.reshapeChecked
      (istftTensor
        packed
        cfg.generator.genIstftNFft
        cfg.generator.genIstftHopSize
        cfg.generator.genIstftNFft
        window
        true
        false
        (GeneratorConfig.waveSamples cfg.generator baseFrames))
      #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  pure (Tensor.packShape (Tensor.reshapeChecked audio1d #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames]))

end Generator

structure DecoderDebugOutput where
  f0 : ShapePackedTensor .Float32
  n : ShapePackedTensor .Float32
  encode : ShapePackedTensor .Float32
  asrRes : ShapePackedTensor .Float32
  decode0 : ShapePackedTensor .Float32
  decode1 : ShapePackedTensor .Float32
  decode2 : ShapePackedTensor .Float32
  decode3 : ShapePackedTensor .Float32
  generator : GeneratorDebugOutput
  deriving TensorStruct, Inhabited

structure Decoder (cfg : KittenTTSConfig) where
  f0Conv : WNConv1dParams 1 1 3 2 1 1
  nConv : WNConv1dParams 1 1 3 2 1 1
  encode : AdainResBlk1dSame (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  asrRes : WNConv1dParams cfg.hiddenDim cfg.asrResDim 1 1 0 1
  decode0 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode1 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode2 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode3 : AdainResBlk1dUp (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  generator : Generator cfg
  deriving TensorStruct

namespace Decoder

def init (cfg : KittenTTSConfig) : IO (Decoder cfg) := do
  let f0Conv ← WNConv1dParams.init 1 1 3 2 1 1
  let nConv ← WNConv1dParams.init 1 1 3 2 1 1
  let encode ← AdainResBlk1dSame.init (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  let asrRes ← WNConv1dParams.init cfg.hiddenDim cfg.asrResDim 1 1 0 1
  let decode0 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode1 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode2 ← AdainResBlk1dSame.init (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode3 ← AdainResBlk1dUp.init (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  let generator ← Generator.init cfg
  pure { f0Conv, nConv, encode, asrRes, decode0, decode1, decode2, decode3, generator }

def debugForward {frames : UInt64}
    (m : Decoder cfg)
    (asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 })
    (f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 })
    (nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    : IO DecoderDebugOutput := do
  let f0 : Tensor { shape := #[1, 1, frames], dtype := .Float32 } := (m.f0Conv.forward f0Curve).reshapeChecked #[1, 1, frames]
  let n : Tensor { shape := #[1, 1, frames], dtype := .Float32 } := (m.nConv.forward nCurve).reshapeChecked #[1, 1, frames]
  let x0 : Tensor { shape := #[1, cfg.hiddenDim + 2, frames], dtype := .Float32 } := (catTo asr (catTo f0 n 1) 1).reshapeChecked #[1, cfg.hiddenDim + 2, frames]
  let x1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.encode.forward x0 style
  let asrRes : Tensor { shape := #[1, cfg.asrResDim, frames], dtype := .Float32 } := (m.asrRes.forward asr).reshapeChecked #[1, cfg.asrResDim, frames]

  let d0In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo x1 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d0 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode0.forward d0In style
  let d1In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d0 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode1.forward d1In style
  let d2In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d1 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d2 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode2.forward d2In style
  let d3In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d2 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d3 : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames], dtype := .Float32 } := m.decode3.forward d3In style
  let generator ← m.generator.debugForward d3 style f0Curve
  pure {
    f0 := Tensor.packShape f0
    n := Tensor.packShape n
    encode := Tensor.packShape x1
    asrRes := Tensor.packShape asrRes
    decode0 := Tensor.packShape d0
    decode1 := Tensor.packShape d1
    decode2 := Tensor.packShape d2
    decode3 := Tensor.packShape d3
    generator
  }

def forward {frames : UInt64}
    (m : Decoder cfg)
    (asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 })
    (f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 })
    (nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 })
    : IO (ShapePackedTensor .Float32) := do
  let f0 : Tensor { shape := #[1, 1, frames], dtype := .Float32 } := (m.f0Conv.forward f0Curve).reshapeChecked #[1, 1, frames]
  let n : Tensor { shape := #[1, 1, frames], dtype := .Float32 } := (m.nConv.forward nCurve).reshapeChecked #[1, 1, frames]
  let x0 : Tensor { shape := #[1, cfg.hiddenDim + 2, frames], dtype := .Float32 } := (catTo asr (catTo f0 n 1) 1).reshapeChecked #[1, cfg.hiddenDim + 2, frames]
  let x1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.encode.forward x0 style
  let asrRes : Tensor { shape := #[1, cfg.asrResDim, frames], dtype := .Float32 } := (m.asrRes.forward asr).reshapeChecked #[1, cfg.asrResDim, frames]
  let d0In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo x1 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d0 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode0.forward d0In style
  let d1In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d0 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode1.forward d1In style
  let d2In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d1 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d2 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32 } := m.decode2.forward d2In style
  let d3In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32 } := (catTo d2 (catTo asrRes (catTo f0 n 1) 1) 1).reshapeChecked #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames]
  let d3 : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames], dtype := .Float32 } := m.decode3.forward d3In style
  m.generator.forward d3 style f0Curve

end Decoder

structure KittenTTSOutput where
  audio : ShapePackedTensor .Float32
  predDurations : Array UInt64
  alignmentFrames : UInt64
  deriving TensorStruct

structure KittenTTSDebugOutput where
  asr : ShapePackedTensor .Float32
  f0Curve : ShapePackedTensor .Float32
  nCurve : ShapePackedTensor .Float32
  audio : ShapePackedTensor .Float32
  predDurations : Array UInt64
  alignmentFrames : UInt64

structure KittenTTSDurationPrediction where
  durVals : Array Float
  predDurations : Array UInt64
  alignmentFrames : UInt64
  deriving Inhabited, Repr

structure Model (cfg : KittenTTSConfig) where
  bert : KittenAlbert cfg.nToken cfg.plbert
  bertEncoder : LinearNorm cfg.plbert.hiddenSize cfg.hiddenDim
  predictor : ProsodyPredictor cfg
  textEncoder : TextEncoder cfg
  decoder : Decoder cfg
  deriving TensorStruct

namespace Model

private def modelDevice (m : Model cfg) : Device :=
  m.bert.embeddings.wordEmbeddings.device

private structure PreparedSynthesis where
  decoderStyle : ShapePackedTensor .Float32
  asr : ShapePackedTensor .Float32
  f0Curve : ShapePackedTensor .Float32
  nCurve : ShapePackedTensor .Float32
  predDurations : Array UInt64
  alignmentFrames : UInt64

private def prepareInputs {seq : UInt64}
    (m : Model cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 })
    : Tensor { shape := #[1, seq], dtype := .Int64 } × Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 } :=
  let device := modelDevice m
  let inputIds :=
    if inputIds.device == device then inputIds else inputIds.to device
  let refStyle :=
    if refStyle.device == device then refStyle else refStyle.to device
  (inputIds, refStyle)

def init (cfg : KittenTTSConfig) : IO (Model cfg) := do
  if cfg.hiddenDim % 2 != 0 then
    throw <| IO.userError s!"KittenTTS requires hiddenDim divisible by 2, got {cfg.hiddenDim}"
  if cfg.plbert.hiddenSize % cfg.plbert.numAttentionHeads != 0 then
    throw <| IO.userError
      s!"KittenTTS ALBERT hidden size must be divisible by numAttentionHeads, got {cfg.plbert.hiddenSize} and {cfg.plbert.numAttentionHeads}"
  if cfg.generator.upsampleRates.isEmpty then
    throw <| IO.userError "KittenTTS generator requires at least one upsample stage"
  if cfg.generator.upsampleKernelSizes.size != cfg.generator.upsampleRates.size then
    throw <| IO.userError
      s!"KittenTTS generator upsampleKernelSizes/upsampleRates mismatch: {cfg.generator.upsampleKernelSizes.size} vs {cfg.generator.upsampleRates.size}"
  if cfg.generator.resblockKernelSizes.isEmpty then
    throw <| IO.userError "KittenTTS generator requires at least one resblock kernel size"
  if cfg.generator.resblockDilationSizes.size != cfg.generator.resblockKernelSizes.size then
    throw <| IO.userError
      s!"KittenTTS generator resblockDilationSizes/resblockKernelSizes mismatch: {cfg.generator.resblockDilationSizes.size} vs {cfg.generator.resblockKernelSizes.size}"
  if GeneratorConfig.finalChannels cfg.generator == 0 then
    throw <| IO.userError
      s!"KittenTTS generator final channel count collapsed to zero; upsampleInitialChannel={cfg.generator.upsampleInitialChannel}, stages={cfg.generator.upsampleRates.size}"
  if KittenTTSConfig.decoderChannels cfg != cfg.generator.upsampleInitialChannel then
    throw <| IO.userError
      s!"KittenTTS decoder/generator channel mismatch: decoderChannels={KittenTTSConfig.decoderChannels cfg}, upsampleInitialChannel={cfg.generator.upsampleInitialChannel}"
  let bert ← KittenAlbert.init cfg.nToken cfg.plbert
  let bertEncoder ← LinearNorm.init cfg.plbert.hiddenSize cfg.hiddenDim
  let predictor ← ProsodyPredictor.init cfg
  let textEncoder ← TextEncoder.init cfg
  let decoder ← Decoder.init cfg
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

def predictDurations {seq : UInt64}
    (m : Model cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 })
    (speed : Float := 1.0)
    : IO KittenTTSDurationPrediction := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  if seq == 0 then
    pure {
      durVals := #[]
      predDurations := #[]
      alignmentFrames := 0
    }
  else
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := m.bertEncoder.forward3d bertOut
    let dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := seqToCF dEnSeq

    let predictorStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 } := sliceTensor refStyle 1 cfg.styleDim cfg.styleDim

    let durEnc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 } :=
      m.predictor.durationEncoding dEn predictorStyle
    let durLogits : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32 } :=
      m.predictor.durationLogits durEnc
    let durSig : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32 } := Tensor.sigmoid durLogits
    let durSum : Tensor { shape := #[1, seq], dtype := .Float32 } := Tensor.reshapeChecked (Tensor.sumDim durSig 2 false) #[1, seq]
    let durVals ← Tensor.toFloatArray durSum
    let predDurations := durationToFrames durVals speed
    let frames := predDurations.foldl (fun acc x => acc + x) 0
    pure {
      durVals
      predDurations
      alignmentFrames := frames
    }

private def prepareSynthesis {seq : UInt64}
    (m : Model cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 })
    (speed : Float := 1.0)
    : IO (Option PreparedSynthesis) := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  let durPred ← predictDurations m inputIds refStyle speed
  if durPred.alignmentFrames == 0 then
    pure none
  else
    let frames := durPred.alignmentFrames
    let decoderStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 } := sliceTensor refStyle 1 0 cfg.styleDim
    let predictorStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 } := sliceTensor refStyle 1 cfg.styleDim cfg.styleDim
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32 } := m.bertEncoder.forward3d bertOut
    let dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := seqToCF dEnSeq
    let durEnc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32 } :=
      m.predictor.durationEncoding dEn predictorStyle
    let align : Tensor { shape := #[1, seq, frames], dtype := .Float32 } := buildAlignment durPred.predDurations inputIds.device
    let prosodyFrames : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, frames], dtype := .Float32 } := bmm durEnc align
    let (f0Curve, nCurve) := m.predictor.forwardF0N prosodyFrames predictorStyle
    let textEnc : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32 } := m.textEncoder.forward inputIds
    let asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := bmm textEnc align
    pure <| some {
      decoderStyle := Tensor.packShape decoderStyle
      asr := Tensor.packShape asr
      f0Curve := Tensor.packShape f0Curve
      nCurve := Tensor.packShape nCurve
      predDurations := durPred.predDurations
      alignmentFrames := frames
    }

def synthesizeIds {seq : UInt64}
    (m : Model cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 })
    (speed : Float := 1.0)
    : IO KittenTTSOutput := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    pure {
      audio := Tensor.packShape (Tensor.zeros #[1, 1, 0] false (modelDevice m))
      predDurations := #[]
      alignmentFrames := 0
    }
  | some prepared =>
    let frames := prepared.alignmentFrames
    let decoderStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 } := Tensor.reshapeChecked prepared.decoderStyle.2 #[1, cfg.styleDim]
    let asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.asr.2 #[1, cfg.hiddenDim, frames]
    let f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.f0Curve.2 #[1, 1, 2 * frames]
    let nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.nCurve.2 #[1, 1, 2 * frames]
    let audio ← m.decoder.forward asr f0Curve nCurve decoderStyle
    pure {
      audio
      predDurations := prepared.predDurations
      alignmentFrames := frames
    }

def debugSynthesizeIds {seq : UInt64}
    (m : Model cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32 })
    (speed : Float := 1.0)
    : IO KittenTTSDebugOutput := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    let device := modelDevice m
    pure {
      asr := Tensor.packShape (Tensor.zeros #[1, cfg.hiddenDim, 0] false device)
      f0Curve := Tensor.packShape (Tensor.zeros #[1, 1, 0] false device)
      nCurve := Tensor.packShape (Tensor.zeros #[1, 1, 0] false device)
      audio := Tensor.packShape (Tensor.zeros #[1, 1, 0] false device)
      predDurations := #[]
      alignmentFrames := 0
    }
  | some prepared =>
    let frames := prepared.alignmentFrames
    let decoderStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32 } := Tensor.reshapeChecked prepared.decoderStyle.2 #[1, cfg.styleDim]
    let asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.asr.2 #[1, cfg.hiddenDim, frames]
    let f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.f0Curve.2 #[1, 1, 2 * frames]
    let nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32 } := Tensor.reshapeChecked prepared.nCurve.2 #[1, 1, 2 * frames]
    let audio ← m.decoder.forward asr f0Curve nCurve decoderStyle
    pure {
      asr := prepared.asr
      f0Curve := prepared.f0Curve
      nCurve := prepared.nCurve
      audio := audio
      predDurations := prepared.predDurations
      alignmentFrames := frames
    }

end Model

end torch.kittentts
