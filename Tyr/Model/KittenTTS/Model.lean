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

variable {device : Device}

open torch.Model

/-- A stride-1 convolution padded by half its dilated receptive field keeps the frame count. -/
private theorem convOutputSize_same {f k p d : Nat} (hf : 0 < f) (hp : 2 * p = d * (k - 1)) :
    convOutputSize f k 1 p d = f := by
  simp only [convOutputSize]; simp; omega

private theorem convOutputSize_pointwise {f : Nat} (hf : 0 < f) : convOutputSize f 1 1 0 1 = f := by
  simp only [convOutputSize]; simp; omega

private theorem convOutputSize_halve {f : Nat} (hf : 0 < f) : convOutputSize (2 * f) 3 2 1 1 = f := by
  simp only [convOutputSize]; omega

private theorem convTransposeOutputSize_double {f : Nat} (hf : 0 < f) :
    convTransposeOutputSize f 3 2 1 1 1 = 2 * f := by
  simp only [convTransposeOutputSize]; omega

/-- Padding `(k*d - d)/2` centres a dilated odd kernel. -/
private theorem dilated_same_pad {k d : Nat} (hk : k % 2 = 1) : 2 * ((k * d - d) / 2) = d * (k - 1) := by
  rw [show k * d - d = d * (k - 1) by rw [Nat.mul_sub_one, Nat.mul_comm]]
  exact Nat.mul_div_cancel' (Nat.dvd_trans (by omega : 2 ∣ k - 1) (Nat.dvd_mul_left (k - 1) d))

private def initUniform (shape : Shape) (scale : Float) : IO (Tensor { shape := shape, dtype := .Float32, device := device }) := do
  let w ← uniformTensor shape (-scale) scale
  Tensor.parameter w

private def xavierScale (dim : Nat) : Float :=
  1.0 / Float.sqrt (max 1 dim).toFloat

private def affineShift {s : Shape} (normed g b : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  (normed * Tensor.addScalar g 1.0) + b

private def addBias2d {batch dim : Nat}
    (x : Tensor { shape := #[batch, dim], dtype := .Float32, device := device })
    (b : Tensor { shape := #[dim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, dim], dtype := .Float32, device := device } :=
  x + Tensor.broadcastTo (Tensor.reshape b #[1, dim]) #[batch, dim]

private def addBias3d {batch channels frames : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (b : Tensor { shape := #[channels], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } :=
  let be : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo (Tensor.reshape b #[1, channels, 1]) #[batch, channels, frames]
  x + be

private def seqToCF {batch seq channels : Nat}
    (x : Tensor { shape := #[batch, seq, channels], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, channels, seq], dtype := .Float32, device := device } :=
  Tensor.reshape (Tensor.transpose x 1 2) #[batch, channels, seq]

private def cfToSeq {batch channels seq : Nat}
    (x : Tensor { shape := #[batch, channels, seq], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, seq, channels], dtype := .Float32, device := device } :=
  Tensor.reshape (Tensor.transpose x 1 2) #[batch, seq, channels]

private def repeatNearest1d {batch channels frames scale : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, channels, frames * scale], dtype := .Float32, device := device } :=
  resizeTensor x (frames * scale) "nearest"

private def resize1d {batch channels inFrames outFrames : Nat}
    (x : Tensor { shape := #[batch, channels, inFrames], dtype := .Float32, device := device })
    (mode : String := "linear")
    : Tensor { shape := #[batch, channels, outFrames], dtype := .Float32, device := device } :=
  resizeTensor x outFrames mode

private def layerNormCF {batch channels frames : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (weight bias : Tensor { shape := #[channels], dtype := .Float32, device := device })
    (eps : Float := 1e-5)
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } :=
  seqToCF (Tensor.layerNorm (cfToSeq x) weight bias eps)

private def instanceNormCF {batch channels frames : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (eps : Float := 1e-5)
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } :=
  let mean : Tensor { shape := #[batch, channels, 1], dtype := .Float32, device := device } := Tensor.meanDim x 2 true
  let meanE : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo mean #[batch, channels, frames]
  let centered : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := x - meanE
  let var : Tensor { shape := #[batch, channels, 1], dtype := .Float32, device := device } := Tensor.meanDim (centered * centered) 2 true
  let invStd : Tensor { shape := #[batch, channels, 1], dtype := .Float32, device := device } := Tensor.rsqrt (var + eps)
  let invStdE : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo invStd #[batch, channels, frames]
  centered * invStdE

private def styleExpand {batch dim frames : Nat}
    (style : Tensor { shape := #[batch, dim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, dim, frames], dtype := .Float32, device := device } :=
  Tensor.broadcastTo (Tensor.reshape style #[batch, dim, 1]) #[batch, dim, frames]

private def snake1d {batch channels frames : Nat}
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (alpha : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } :=
  let a : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo alpha #[batch, channels, frames]
  let inv : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } := Tensor.div (Tensor.ones #[batch, channels, frames] false x.device) a
  x + inv * Tensor.pow (Tensor.sin (x * a)) 2.0

private def geluApprox {s : Shape} (x : Tensor { shape := s, dtype := .Float32, device := device }) : Tensor { shape := s, dtype := .Float32, device := device } :=
  let cubic := Tensor.pow x 3.0
  let inner := x + Tensor.mulScalar cubic 0.044715
  let t := Tensor.tanh (Tensor.mulScalar inner 0.7978846)
  Tensor.mulScalar x 0.5 * Tensor.addScalar t 1.0

private def durationToFrames (vals : Array Float) (speed : Float) : Array Nat :=
  vals.map fun v =>
    let denom := if speed <= 0.0 then 1.0 else speed
    let raw := ((v / denom) + 0.5).toUInt64.toNat
    if raw == 0 then 1 else raw

private def buildAlignment {seq frames : Nat}
    (durations : Array Nat)
    (device : Device := Device.CPU)
    : Tensor { shape := #[1, seq, frames], dtype := .Float32, device := device } := Id.run do
  if frames == 0 then
    Tensor.reshape (Tensor.zeros #[1, seq, frames] false device) #[1, seq, frames]
  else
    -- Token `i` covers frames `[starts[i], starts[i] + dur_i)`, clipped to `frames`.
    let starts : Array Nat := (List.range seq).toArray.map fun i =>
      (List.range i).foldl (fun acc j => acc + durations.getD j 1) 0
    let flat : Array Float := Array.ofFn (n := seq * frames) fun idx =>
      let i := idx.val / frames
      let t := idx.val % frames
      let start := starts.getD i 0
      if start ≤ t && t < start + durations.getD i 1 then 1.0 else 0.0
    (Tensor.fromFloatArrayShaped flat #[1, seq, frames] (by simp [flat])).to device

/-- Linear layer with explicit weight / bias tensors. -/
structure LinearNorm (inDim outDim : Nat) where
  weight : Tensor { shape := #[outDim, inDim], dtype := .Float32, device := device }
  bias : Tensor { shape := #[outDim], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace LinearNorm

def init (inDim outDim : Nat) (device : Device := .CPU) : IO (LinearNorm (device := device) inDim outDim) := do
  let scale := xavierScale outDim
  let weight ← initUniform #[outDim, inDim] scale
  let bias ← initBias (device := device) #[outDim]
  pure { weight, bias }

def forward2d {batch inDim outDim : Nat}
    (m : LinearNorm (device := device) inDim outDim)
    (x : Tensor { shape := #[batch, inDim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, outDim], dtype := .Float32, device := device } :=
  affine2d x m.weight m.bias

def forward3d {batch seq inDim outDim : Nat}
    (m : LinearNorm (device := device) inDim outDim)
    (x : Tensor { shape := #[batch, seq, inDim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, seq, outDim], dtype := .Float32, device := device } :=
  Tensor.affine3d x m.weight m.bias

end LinearNorm

structure Conv1dParams
    (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (dilation : Nat := 1) where
  weight : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device }
  bias : Tensor { shape := #[outC], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace Conv1dParams

def init (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (dilation : Nat := 1) (device : Device := .CPU)
    : IO (Conv1dParams (device := device) inC outC kernel stride padding dilation) := do
  let weight ← initWeight (device := device) #[outC, inC, kernel] (inC * kernel)
  let bias ← initBias (device := device) #[outC]
  pure { weight, bias }

def forward {batch frames inC outC kernel stride padding dilation : Nat}
    (m : Conv1dParams (device := device) inC outC kernel stride padding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32, device := device })
    (valid : Tensor.ConvValid frames kernel stride padding dilation := by conv_valid)
    : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32, device := device } :=
  let y0 : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32, device := device } :=
    Tensor.reshape (Tensor.conv1d x m.weight stride padding dilation valid) #[batch, outC, convOutputSize frames kernel stride padding dilation]
      (by simp)
  addBias3d y0 m.bias

end Conv1dParams

private def normDim0_3d {d0 d1 d2 : Nat}
    (x : Tensor { shape := #[d0, d1, d2], dtype := .Float32, device := device })
    : Tensor { shape := #[d0], dtype := .Float32, device := device } :=
  let sq : Tensor { shape := #[d0, d1, d2], dtype := .Float32, device := device } := x * x
  let s2 : Tensor { shape := #[d0, d1, 1], dtype := .Float32, device := device } := Tensor.sumDim sq 2 true
  let s1 : Tensor { shape := #[d0, 1, 1], dtype := .Float32, device := device } := Tensor.sumDim s2 1 true
  Tensor.reshape (Tensor.sqrt (s1 + (1e-12 : Float))) #[d0]

private def applyWeightNormConv1d {outC inC kernel : Nat}
    (weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device })
    (weightG : Tensor { shape := #[outC], dtype := .Float32, device := device })
    : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } :=
  let denom : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo denom #[outC, inC, kernel]
  let gain : Tensor { shape := #[outC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape weightG #[outC, 1, 1]
  let gainE : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo gain #[outC, inC, kernel]
  Tensor.div (weightV * gainE) denomE

private def applyWeightNormTransConv1d {inC outC kernel : Nat}
    (weightV : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device })
    (weightG : Tensor { shape := #[inC], dtype := .Float32, device := device })
    : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device } :=
  let denom : Tensor { shape := #[inC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape (normDim0_3d weightV) #[inC, 1, 1]
  let denomE : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo denom #[inC, outC, kernel]
  let gain : Tensor { shape := #[inC, 1, 1], dtype := .Float32, device := device } := Tensor.reshape weightG #[inC, 1, 1]
  let gainE : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device } := Tensor.broadcastTo gain #[inC, outC, kernel]
  Tensor.div (weightV * gainE) denomE

structure WNConv1dParams
    (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (dilation : Nat := 1) where
  weightV : Tensor { shape := #[outC, inC, kernel], dtype := .Float32, device := device }
  weightG : Tensor { shape := #[outC], dtype := .Float32, device := device }
  bias : Tensor { shape := #[outC], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace WNConv1dParams

def init (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (dilation : Nat := 1) (device : Device := .CPU)
    : IO (WNConv1dParams (device := device) inC outC kernel stride padding dilation) := do
  let weightV ← initWeight (device := device) #[outC, inC, kernel] (inC * kernel)
  let weightG : Tensor { shape := #[outC], dtype := .Float32, device := device } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias (device := device) #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding dilation : Nat}
    (m : WNConv1dParams (device := device) inC outC kernel stride padding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32, device := device })
    (valid : Tensor.ConvValid frames kernel stride padding dilation := by conv_valid)
    : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32, device := device } :=
  let weight := applyWeightNormConv1d m.weightV m.weightG
  let y0 : Tensor { shape := #[batch, outC, convOutputSize frames kernel stride padding dilation], dtype := .Float32, device := device } :=
    Tensor.reshape (Tensor.conv1d x weight stride padding dilation valid) #[batch, outC, convOutputSize frames kernel stride padding dilation]
      (by simp)
  addBias3d y0 m.bias

end WNConv1dParams

structure WNConvTranspose1dParams
    (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (outputPadding : Nat := 0)
    (dilation : Nat := 1) where
  weightV : Tensor { shape := #[inC, outC, kernel], dtype := .Float32, device := device }
  weightG : Tensor { shape := #[inC], dtype := .Float32, device := device }
  bias : Tensor { shape := #[outC], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace WNConvTranspose1dParams

def init (inC outC kernel : Nat)
    (stride : Nat := 1)
    (padding : Nat := 0)
    (outputPadding : Nat := 0)
    (dilation : Nat := 1) (device : Device := .CPU)
    : IO (WNConvTranspose1dParams (device := device) inC outC kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight (device := device) #[inC, outC, kernel] (max 1 (outC * kernel))
  let weightG : Tensor { shape := #[inC], dtype := .Float32, device := device } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias (device := device) #[outC]
  pure { weightV, weightG, bias }

def forward {batch frames inC outC kernel stride padding outputPadding dilation : Nat}
    (m : WNConvTranspose1dParams (device := device) inC outC kernel stride padding outputPadding dilation)
    (x : Tensor { shape := #[batch, inC, frames], dtype := .Float32, device := device })
    (valid : Tensor.ConvTransposeValid frames kernel stride padding outputPadding dilation :=
      by first | exact of_decide_eq_true rfl | (refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega))
    : Tensor { shape := #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation], dtype := .Float32, device := device } :=
  let weight := applyWeightNormTransConv1d m.weightV m.weightG
  Tensor.reshape
    (Tensor.convTranspose1d x weight m.bias stride padding outputPadding dilation valid)
    #[batch, outC, convTransposeOutputSize frames kernel stride padding outputPadding dilation]
    (by simp)

end WNConvTranspose1dParams

structure WNDepthwiseTransConv1dParams
    (channels kernel : Nat)
    (stride : Nat := 2)
    (padding : Nat := 1)
    (outputPadding : Nat := 1)
    (dilation : Nat := 1) where
  weightV : Tensor { shape := #[channels, 1, kernel], dtype := .Float32, device := device }
  weightG : Tensor { shape := #[channels], dtype := .Float32, device := device }
  bias : Tensor { shape := #[channels], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace WNDepthwiseTransConv1dParams

def init (channels kernel : Nat)
    (stride : Nat := 2)
    (padding : Nat := 1)
    (outputPadding : Nat := 1)
    (dilation : Nat := 1) (device : Device := .CPU)
    : IO (WNDepthwiseTransConv1dParams (device := device) channels kernel stride padding outputPadding dilation) := do
  let weightV ← initWeight (device := device) #[channels, 1, kernel] kernel
  let weightG : Tensor { shape := #[channels], dtype := .Float32, device := device } ← Tensor.parameter (normDim0_3d weightV)
  let bias ← initBias (device := device) #[channels]
  pure { weightV, weightG, bias }

def forward {batch frames channels kernel stride padding outputPadding dilation : Nat}
    (m : WNDepthwiseTransConv1dParams (device := device) channels kernel stride padding outputPadding dilation)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (valid : Tensor.ConvTransposeValid frames kernel stride padding outputPadding dilation :=
      by first | exact of_decide_eq_true rfl | (refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega))
    : Tensor { shape := #[batch, channels, convTransposeOutputSize frames kernel stride padding outputPadding dilation], dtype := .Float32, device := device } := Id.run do
  let outFrames := convTransposeOutputSize frames kernel stride padding outputPadding dilation
  let weight : Tensor { shape := #[channels, 1, kernel], dtype := .Float32, device := device } := applyWeightNormTransConv1d m.weightV m.weightG
  let mut parts : Array (ShapePackedTensor .Float32 device) := #[]
  for hc : ci in [:channels] do
    let xc : Tensor { shape := #[batch, 1, frames], dtype := .Float32, device := device } := Tensor.slice x 1 ci 1 (_bounds := by have := hc.upper; simp at this ⊢; omega)
    let wc : Tensor { shape := #[1, 1, kernel], dtype := .Float32, device := device } := Tensor.slice weight 0 ci 1 (_bounds := by have := hc.upper; simp at this ⊢; omega)
    let bc : Tensor { shape := #[1], dtype := .Float32, device := device } := Tensor.slice m.bias 0 ci 1 (_bounds := by have := hc.upper; simp at this ⊢; omega)
    let yc : Tensor { shape := #[batch, 1, outFrames], dtype := .Float32, device := device } :=
      Tensor.reshape (Tensor.convTranspose1d xc wc bc stride padding outputPadding dilation valid) #[batch, 1, outFrames]
        (by simp [outFrames])
    parts := parts.push (Tensor.packShape yc)
  Tensor.catMany! parts 1 #[batch, channels, outFrames]

end WNDepthwiseTransConv1dParams

structure BiLSTM (inputSize hiddenSize : Nat) where
  wxForward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32, device := device }
  whForward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32, device := device }
  biasIHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device }
  biasHHForward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device }
  wxBackward : Tensor { shape := #[4 * hiddenSize, inputSize], dtype := .Float32, device := device }
  whBackward : Tensor { shape := #[4 * hiddenSize, hiddenSize], dtype := .Float32, device := device }
  biasIHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device }
  biasHHBackward : Tensor { shape := #[4 * hiddenSize], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace BiLSTM

def init (inputSize hiddenSize : Nat) (device : Device := .CPU) : IO (BiLSTM (device := device) inputSize hiddenSize) := do
  let scale := xavierScale hiddenSize
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
    {hidden input : Nat}
    (x : Tensor { shape := #[1, input], dtype := .Float32, device := device })
    (h c : Tensor { shape := #[1, hidden], dtype := .Float32, device := device })
    (wx : Tensor { shape := #[4 * hidden, input], dtype := .Float32, device := device })
    (wh : Tensor { shape := #[4 * hidden, hidden], dtype := .Float32, device := device })
    (bih bhh : Tensor { shape := #[4 * hidden], dtype := .Float32, device := device })
    : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } × Tensor { shape := #[1, hidden], dtype := .Float32, device := device } :=
  let gatesIH : Tensor { shape := #[1, 4 * hidden], dtype := .Float32, device := device } := affine2d x wx bih
  let gatesHH : Tensor { shape := #[1, 4 * hidden], dtype := .Float32, device := device } := affine2d h wh bhh
  let gates : Tensor { shape := #[1, 4 * hidden], dtype := .Float32, device := device } := gatesIH + gatesHH

  let iRaw : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.slice gates 1 0 hidden
  let fRaw : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.slice gates 1 hidden hidden
  let gRaw : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.slice gates 1 (2 * hidden) hidden
  let oRaw : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.slice gates 1 (3 * hidden) hidden

  let iGate : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.sigmoid iRaw
  let fGate : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.sigmoid fRaw
  let gGate : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.tanh gRaw
  let oGate : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.sigmoid oRaw

  let cNext : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := fGate * c + iGate * gGate
  let hNext : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := oGate * Tensor.tanh cNext
  (hNext, cNext)

private def runDirection {seq input hidden : Nat}
    (x : Tensor { shape := #[1, seq, input], dtype := .Float32, device := device })
    (wx : Tensor { shape := #[4 * hidden, input], dtype := .Float32, device := device })
    (wh : Tensor { shape := #[4 * hidden, hidden], dtype := .Float32, device := device })
    (bih bhh : Tensor { shape := #[4 * hidden], dtype := .Float32, device := device })
    (reverse : Bool := false)
    : Tensor { shape := #[1, seq, hidden], dtype := .Float32, device := device } := Id.run do
  let mut h : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.zeros #[1, hidden] false x.device
  let mut c : Tensor { shape := #[1, hidden], dtype := .Float32, device := device } := Tensor.zeros #[1, hidden] false x.device
  let mut outs : Array (ShapePackedTensor .Float32 device) := #[]
  for ht : t in [:seq] do
    let idx := if reverse then seq - 1 - t else t
    let xt3 : Tensor { shape := #[1, 1, input], dtype := .Float32, device := device } :=
      Tensor.slice x 1 idx 1 (_bounds := by have := ht.upper; simp only [idx]; split <;> simp at this ⊢ <;> omega)
    let xt : Tensor { shape := #[1, input], dtype := .Float32, device := device } := Tensor.reshape xt3 #[1, input]
    let (hNext, cNext) := step xt h c wx wh bih bhh
    h := hNext
    c := cNext
    let pushed : Tensor { shape := #[1, 1, hidden], dtype := .Float32, device := device } := Tensor.reshape hNext #[1, 1, hidden]
    outs := outs.push (Tensor.packShape pushed)
  let ordered := if reverse then outs.reverse else outs
  Tensor.catMany! ordered 1 #[1, seq, hidden]

def forward {seq input hidden : Nat}
    (m : BiLSTM (device := device) input hidden)
    (x : Tensor { shape := #[1, seq, input], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, 2 * hidden], dtype := .Float32, device := device } :=
  let fw := runDirection x m.wxForward m.whForward m.biasIHForward m.biasHHForward false
  let bw := runDirection x m.wxBackward m.whBackward m.biasIHBackward m.biasHHBackward true
  (catTo fw bw 2).reshape #[1, seq, 2 * hidden] (by simp [Nat.two_mul])

end BiLSTM

structure AlbertEmbeddings (nToken : Nat) (cfg : AlbertConfig) where
  wordEmbeddings : Tensor { shape := #[nToken, cfg.embeddingSize], dtype := .Float32, device := device }
  positionEmbeddings : Tensor { shape := #[cfg.maxPositionEmbeddings, cfg.embeddingSize], dtype := .Float32, device := device }
  tokenTypeEmbeddings : Tensor { shape := #[cfg.typeVocabSize, cfg.embeddingSize], dtype := .Float32, device := device }
  layerNormWeight : Tensor { shape := #[cfg.embeddingSize], dtype := .Float32, device := device }
  layerNormBias : Tensor { shape := #[cfg.embeddingSize], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace AlbertEmbeddings

def init (nToken : Nat) (cfg : AlbertConfig) (device : Device := .CPU) : IO (AlbertEmbeddings (device := device) nToken cfg) := do
  let scale := xavierScale cfg.embeddingSize
  let wordEmbeddings ← initUniform #[nToken, cfg.embeddingSize] scale
  let positionEmbeddings ← initUniform #[cfg.maxPositionEmbeddings, cfg.embeddingSize] scale
  let tokenTypeEmbeddings ← initUniform #[cfg.typeVocabSize, cfg.embeddingSize] scale
  let layerNormWeight ← initUniform #[cfg.embeddingSize] 0.02
  let layerNormBias ← initBias (device := device) #[cfg.embeddingSize]
  pure { wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNormWeight, layerNormBias }

def forward {seq nToken : Nat}
    (m : AlbertEmbeddings (device := device) nToken cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32, device := device } :=
  let tokTypes : Tensor { shape := #[1, seq], dtype := .Int64, device := device } := (Tensor.fullInt #[1, seq] 0).to device
  let posIds1d : Tensor { shape := #[seq], dtype := .Int64, device := device } := (Tensor.arange seq).to device
  let posIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device } := Tensor.reshape posIds1d #[1, seq]
  let word : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32, device := device } := Tensor.embedding inputIds m.wordEmbeddings
  let pos : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32, device := device } := Tensor.embedding posIds m.positionEmbeddings
  let typ : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32, device := device } := Tensor.embedding tokTypes m.tokenTypeEmbeddings
  Tensor.layerNorm (word + pos + typ) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end AlbertEmbeddings

structure KittenAlbertSelfAttention (cfg : AlbertConfig) where
  query : LinearNorm (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  key : LinearNorm (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  value : LinearNorm (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  dense : LinearNorm (device := device) (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  layerNormWeight : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32, device := device }
  layerNormBias : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace KittenAlbertSelfAttention

def init (cfg : AlbertConfig) (device : Device := .CPU) : IO (KittenAlbertSelfAttention (device := device) cfg) := do
  let query ← LinearNorm.init (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let key ← LinearNorm.init (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let value ← LinearNorm.init (device := device) cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let dense ← LinearNorm.init (device := device) (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  let layerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let layerNormBias ← initBias (device := device) #[cfg.hiddenSize]
  pure { query, key, value, dense, layerNormWeight, layerNormBias }

def forward {seq : Nat}
    (m : KittenAlbertSelfAttention (device := device) cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } :=
  let q0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32, device := device } := m.query.forward3d x
  let k0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32, device := device } := m.key.forward3d x
  let v0 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32, device := device } := m.value.forward3d x

  let q : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32, device := device } :=
    Tensor.reshape q0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let k : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32, device := device } :=
    Tensor.reshape k0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]
  let v : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32, device := device } :=
    Tensor.reshape v0 #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg]

  let qh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32, device := device } := Tensor.splitHeads q
  let kh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32, device := device } := Tensor.splitHeads k
  let vh : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32, device := device } := Tensor.splitHeads v
  let attn : Tensor { shape := #[1, cfg.numAttentionHeads, seq, AlbertConfig.headDim cfg], dtype := .Float32, device := device } :=
    Tensor.attentionGQA qh kh vh false true
  let out4 : Tensor { shape := #[1, seq, cfg.numAttentionHeads, AlbertConfig.headDim cfg], dtype := .Float32, device := device } := Tensor.joinHeads attn
  let out3 : Tensor { shape := #[1, seq, AlbertConfig.allHeadSize cfg], dtype := .Float32, device := device } :=
    Tensor.reshape out4 #[1, seq, AlbertConfig.allHeadSize cfg]
  let denseOut : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } := m.dense.forward3d out3
  Tensor.layerNorm (denseOut + x) m.layerNormWeight m.layerNormBias cfg.layerNormEps

end KittenAlbertSelfAttention

structure KittenAlbertLayer (cfg : AlbertConfig) where
  attention : KittenAlbertSelfAttention (device := device) cfg
  fullLayerNormWeight : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32, device := device }
  fullLayerNormBias : Tensor { shape := #[cfg.hiddenSize], dtype := .Float32, device := device }
  ffn : LinearNorm (device := device) cfg.hiddenSize cfg.intermediateSize
  ffnOutput : LinearNorm (device := device) cfg.intermediateSize cfg.hiddenSize
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayer

def init (cfg : AlbertConfig) (device : Device := .CPU) : IO (KittenAlbertLayer (device := device) cfg) := do
  let attention ← KittenAlbertSelfAttention.init (device := device) cfg
  let fullLayerNormWeight ← initUniform #[cfg.hiddenSize] 0.02
  let fullLayerNormBias ← initBias (device := device) #[cfg.hiddenSize]
  let ffn ← LinearNorm.init (device := device) cfg.hiddenSize cfg.intermediateSize
  let ffnOutput ← LinearNorm.init (device := device) cfg.intermediateSize cfg.hiddenSize
  pure { attention, fullLayerNormWeight, fullLayerNormBias, ffn, ffnOutput }

def forward {seq : Nat}
    (m : KittenAlbertLayer (device := device) cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } :=
  let attnOut := m.attention.forward x
  let f0 : Tensor { shape := #[1, seq, cfg.intermediateSize], dtype := .Float32, device := device } := m.ffn.forward3d attnOut
  let f1 := geluApprox f0
  let f2 : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } := m.ffnOutput.forward3d f1
  Tensor.layerNorm (f2 + attnOut) m.fullLayerNormWeight m.fullLayerNormBias cfg.layerNormEps

end KittenAlbertLayer

structure KittenAlbertLayerGroup (cfg : AlbertConfig) where
  layers : Array (KittenAlbertLayer (device := device) cfg)
  deriving TensorStruct, Inhabited

namespace KittenAlbertLayerGroup

def init (cfg : AlbertConfig) (device : Device := .CPU) : IO (KittenAlbertLayerGroup (device := device) cfg) := do
  let mut layers : Array (KittenAlbertLayer (device := device) cfg) := #[]
  for _ in [:cfg.innerGroupNum] do
    layers := layers.push (← KittenAlbertLayer.init (device := device) cfg)
  pure { layers }

def forward {seq : Nat}
    (m : KittenAlbertLayerGroup (device := device) cfg)
    (x : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } :=
  m.layers.foldl (fun h layer => layer.forward h) x

end KittenAlbertLayerGroup

structure KittenAlbertEncoder (cfg : AlbertConfig) where
  embeddingHiddenMappingIn : LinearNorm (device := device) cfg.embeddingSize cfg.hiddenSize
  groups : torch.Vector cfg.numHiddenGroups (KittenAlbertLayerGroup (device := device) cfg)
  deriving TensorStruct

namespace KittenAlbertEncoder

def init (cfg : AlbertConfig) (device : Device := .CPU) : IO (KittenAlbertEncoder (device := device) cfg) := do
  let embeddingHiddenMappingIn ← LinearNorm.init (device := device) cfg.embeddingSize cfg.hiddenSize
  let slots : torch.Vector cfg.numHiddenGroups Nat := ⟨Array.range cfg.numHiddenGroups, Array.size_range⟩
  let groups ← slots.mapM fun _ => KittenAlbertLayerGroup.init (device := device) cfg
  pure { embeddingHiddenMappingIn, groups }

def forward {seq : Nat}
    (m : KittenAlbertEncoder (device := device) cfg)
    (x : Tensor { shape := #[1, seq, cfg.embeddingSize], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } := Id.run do
  let mapped : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } := m.embeddingHiddenMappingIn.forward3d x
  let mut h := mapped
  for hi : i in [:cfg.numHiddenLayers] do
    have hLayers : i < cfg.layersPerGroup * cfg.numHiddenGroups := by
      rw [AlbertConfig.layersPerGroup, Nat.div_mul_cancel cfg.hiddenGroupsValid.2]
      exact hi.upper
    let group := m.groups.get ⟨i / cfg.layersPerGroup, Nat.div_lt_of_lt_mul hLayers⟩
    h := group.forward h
  h

end KittenAlbertEncoder

structure KittenAlbert (nToken : Nat) (cfg : AlbertConfig) where
  embeddings : AlbertEmbeddings (device := device) nToken cfg
  encoder : KittenAlbertEncoder (device := device) cfg
  pooler : LinearNorm (device := device) cfg.hiddenSize cfg.hiddenSize
  deriving TensorStruct

namespace KittenAlbert

def init (nToken : Nat) (cfg : AlbertConfig) (device : Device := .CPU) : IO (KittenAlbert (device := device) nToken cfg) := do
  let embeddings ← AlbertEmbeddings.init (device := device) nToken cfg
  let encoder ← KittenAlbertEncoder.init (device := device) cfg
  let pooler ← LinearNorm.init (device := device) cfg.hiddenSize cfg.hiddenSize
  pure { embeddings, encoder, pooler }

def forward {seq nToken : Nat}
    (m : KittenAlbert (device := device) nToken cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (hseq : 0 < seq := by omega)
    : Tensor { shape := #[1, seq, cfg.hiddenSize], dtype := .Float32, device := device } × Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32, device := device } :=
  let emb := m.embeddings.forward inputIds
  let h := m.encoder.forward emb
  let cls3 : Tensor { shape := #[1, 1, cfg.hiddenSize], dtype := .Float32, device := device } := Tensor.slice h 1 0 1
  let cls2 : Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32, device := device } := Tensor.reshape cls3 #[1, cfg.hiddenSize]
  let pooled : Tensor { shape := #[1, cfg.hiddenSize], dtype := .Float32, device := device } := Tensor.tanh (m.pooler.forward2d cls2)
  (h, pooled)

end KittenAlbert

structure TextConvLayer (cfg : KittenTTSConfig) where
  conv : Conv1dParams (device := device) cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  lnWeight : Tensor { shape := #[cfg.hiddenDim], dtype := .Float32, device := device }
  lnBias : Tensor { shape := #[cfg.hiddenDim], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace TextConvLayer

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (TextConvLayer (device := device) cfg) := do
  let conv ← Conv1dParams.init (device := device) cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize 1 ((cfg.textEncoderKernelSize - 1) / 2) 1
  let lnWeight ← initUniform #[cfg.hiddenDim] 0.02
  let lnBias ← initBias (device := device) #[cfg.hiddenDim]
  pure { conv, lnWeight, lnBias }

def forward {seq : Nat}
    (m : TextConvLayer (device := device) cfg)
    (x : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device })
    (hs : 0 < seq)
    : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } :=
  have := cfg.textKernelOdd
  let y : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } :=
    (m.conv.forward x).reshape #[1, cfg.hiddenDim, seq]
      (by rw [convOutputSize_same hs (by have := cfg.textKernelOdd; omega)])
  let y := layerNormCF y m.lnWeight m.lnBias
  leakyRelu y 0.2

end TextConvLayer

structure TextEncoder (cfg : KittenTTSConfig) where
  embedding : Tensor { shape := #[cfg.nToken, cfg.hiddenDim], dtype := .Float32, device := device }
  convs : Array (TextConvLayer (device := device) cfg)
  lstm : BiLSTM (device := device) cfg.hiddenDim (cfg.hiddenDim / 2)
  deriving TensorStruct, Inhabited

namespace TextEncoder

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (TextEncoder (device := device) cfg) := do
  let scale := xavierScale cfg.hiddenDim
  let embedding ← initUniform #[cfg.nToken, cfg.hiddenDim] scale
  let mut convs : Array (TextConvLayer (device := device) cfg) := #[]
  for _ in [:cfg.nLayer] do
    convs := convs.push (← TextConvLayer.init (device := device) cfg)
  let lstm ← BiLSTM.init (device := device) cfg.hiddenDim (cfg.hiddenDim / 2)
  pure { embedding, convs, lstm }

def forward {seq : Nat}
    (m : TextEncoder (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (hs : 0 < seq)
    : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := Id.run do
  let emb : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := Tensor.embedding inputIds m.embedding
  let mut h : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := seqToCF emb
  for conv in m.convs do
    h := conv.forward h hs
  let hSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := cfToSeq h
  let outSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := (m.lstm.forward hSeq).reshape #[1, seq, cfg.hiddenDim] (by rw [Nat.mul_div_cancel' cfg.hiddenDimEven])
  seqToCF outSeq

end TextEncoder

structure AdaLayerNorm (styleDim channels : Nat) where
  fc : LinearNorm (device := device) styleDim (2 * channels)
  deriving TensorStruct, Inhabited

namespace AdaLayerNorm

def init (styleDim channels : Nat) (device : Device := .CPU) : IO (AdaLayerNorm (device := device) styleDim channels) := do
  let fc ← LinearNorm.init (device := device) styleDim (2 * channels)
  pure { fc }

def forward {batch frames styleDim channels : Nat}
    (m : AdaLayerNorm (device := device) styleDim channels)
    (x : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[batch, styleDim], dtype := .Float32, device := device })
    : Tensor { shape := #[batch, channels, frames], dtype := .Float32, device := device } :=
  let h : Tensor { shape := #[batch, 2 * channels], dtype := .Float32, device := device } := m.fc.forward2d style
  let gamma : Tensor { shape := #[batch, 1, channels], dtype := .Float32, device := device } := Tensor.reshape (Tensor.slice h 1 0 channels) #[batch, 1, channels]
  let beta : Tensor { shape := #[batch, 1, channels], dtype := .Float32, device := device } := Tensor.reshape (Tensor.slice h 1 channels channels) #[batch, 1, channels]
  let xSeq : Tensor { shape := #[batch, frames, channels], dtype := .Float32, device := device } := cfToSeq x
  let normed : Tensor { shape := #[batch, frames, channels], dtype := .Float32, device := device } := Tensor.layerNorm xSeq (Tensor.ones #[channels] false x.device) (Tensor.zeros #[channels] false x.device) 1e-5
  let g : Tensor { shape := #[batch, frames, channels], dtype := .Float32, device := device } := Tensor.broadcastTo gamma #[batch, frames, channels]
  let b : Tensor { shape := #[batch, frames, channels], dtype := .Float32, device := device } := Tensor.broadcastTo beta #[batch, frames, channels]
  seqToCF (affineShift normed g b)

end AdaLayerNorm

structure AdaIN1d (styleDim channels : Nat) where
  fc : LinearNorm (device := device) styleDim (2 * channels)
  normWeight : Tensor { shape := #[channels], dtype := .Float32, device := device }
  normBias : Tensor { shape := #[channels], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace AdaIN1d

def init (styleDim channels : Nat) (device : Device := .CPU) : IO (AdaIN1d (device := device) styleDim channels) := do
  let fc ← LinearNorm.init (device := device) styleDim (2 * channels)
  let normWeight ← Tensor.parameter (Tensor.ones #[channels] (device := device))
  let normBias ← initBias (device := device) #[channels]
  pure { fc, normWeight, normBias }

def forward {frames styleDim channels : Nat}
    (m : AdaIN1d (device := device) styleDim channels)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
  let h : Tensor { shape := #[1, 2 * channels], dtype := .Float32, device := device } := m.fc.forward2d style
  let gamma : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device } := Tensor.reshape (Tensor.slice h 1 0 channels) #[1, channels, 1]
  let beta : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device } := Tensor.reshape (Tensor.slice h 1 channels channels) #[1, channels, 1]
  let baseNorm := instanceNormCF x
  let normW : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo (Tensor.reshape m.normWeight #[1, channels, 1]) #[1, channels, frames]
  let normB : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo (Tensor.reshape m.normBias #[1, channels, 1]) #[1, channels, frames]
  let normed : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } := baseNorm * normW + normB
  let g : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo gamma #[1, channels, frames]
  let b : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } := Tensor.broadcastTo beta #[1, channels, frames]
  affineShift normed g b

end AdaIN1d

structure DurationEncoder (cfg : KittenTTSConfig) where
  lstms : torch.Vector cfg.nLayer (BiLSTM (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
  norms : torch.Vector cfg.nLayer (AdaLayerNorm (device := device) cfg.styleDim cfg.hiddenDim)
  deriving TensorStruct

namespace DurationEncoder

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (DurationEncoder (device := device) cfg) := do
  let layers : torch.Vector cfg.nLayer Nat := ⟨Array.range cfg.nLayer, Array.size_range⟩
  let lstms ← layers.mapM fun _ =>
    BiLSTM.init (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let norms ← layers.mapM fun _ => AdaLayerNorm.init (device := device) cfg.styleDim cfg.hiddenDim
  pure { lstms, norms }

def forward {seq : Nat}
    (m : DurationEncoder (device := device) cfg)
    (x : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device } := Id.run do
  let s : Tensor { shape := #[1, cfg.styleDim, seq], dtype := .Float32, device := device } := styleExpand style
  let mut h : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device } := (catTo x s 1).reshape #[1, cfg.hiddenDim + cfg.styleDim, seq]
  for hi : i in [:cfg.nLayer] do
    let lstm := m.lstms.get ⟨i, hi.upper⟩
    let norm := m.norms.get ⟨i, hi.upper⟩
    let ySeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := (BiLSTM.forward lstm (cfToSeq h)).reshape #[1, seq, cfg.hiddenDim] (by rw [Nat.mul_div_cancel' cfg.hiddenDimEven])
    let yCF : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := seqToCF ySeq
    let yNorm : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := AdaLayerNorm.forward norm yCF style
    h := catTo yNorm s 1
  h

end DurationEncoder

/-- Residual path of an AdaIN block: identity when the channel counts agree, else a 1x1 conv. -/
inductive Shortcut (inC outC : Nat) where
  | identity (h : inC = outC)
  | conv (c : WNConv1dParams (device := device) inC outC 1 1 0 1)

instance {inC outC : Nat} : TensorStruct (Shortcut (device := device) inC outC) where
  map f
    | .identity h => .identity h
    | .conv c => .conv (TensorStruct.map f c)
  mapM f
    | .identity h => pure (.identity h)
    | .conv c => do pure (.conv (← TensorStruct.mapM f c))
  zipWith f lhs rhs := match lhs, rhs with
    | .conv x, .conv y => .conv (TensorStruct.zipWith f x y)
    | lhs, _ => lhs
  fold f init
    | .identity _ => init
    | .conv c => TensorStruct.fold f init c

namespace Shortcut

def init (inC outC : Nat) (device : Device := .CPU) : IO (Shortcut (device := device) inC outC) :=
  if h : inC = outC then pure (.identity h)
  else return .conv (← WNConv1dParams.init (device := device) inC outC 1 1 0 1)

def forward {frames inC outC : Nat}
    (m : Shortcut (device := device) inC outC)
    (x : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } :=
  match m with
  | .identity h => Tensor.reshape x #[1, outC, frames] (by simp [h])
  | .conv c => (c.forward x).reshape #[1, outC, frames] (by rw [convOutputSize_pointwise hf])

end Shortcut

instance {inC outC : Nat} : Inhabited (Shortcut (device := device) inC outC) :=
  ⟨.conv default⟩

structure AdainResBlk1dSame (inC outC styleDim : Nat) where
  conv1 : WNConv1dParams (device := device) inC outC 3 1 1 1
  conv2 : WNConv1dParams (device := device) outC outC 3 1 1 1
  norm1 : AdaIN1d (device := device) styleDim inC
  norm2 : AdaIN1d (device := device) styleDim outC
  shortcut : Shortcut (device := device) inC outC
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dSame

def init (inC outC styleDim : Nat) (device : Device := .CPU) : IO (AdainResBlk1dSame (device := device) inC outC styleDim) := do
  let conv1 ← WNConv1dParams.init (device := device) inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init (device := device) outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init (device := device) styleDim inC
  let norm2 ← AdaIN1d.init (device := device) styleDim outC
  let shortcut ← Shortcut.init (device := device) inC outC
  pure { conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : Nat}
    (m : AdainResBlk1dSame (device := device) inC outC styleDim)
    (x : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } :=
  let shortcut : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } := m.shortcut.forward x hf
  let h0 : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device } := m.norm1.forward x style
  let h1 : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device } := leakyRelu h0 0.2
  let h2 : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } := (m.conv1.forward h1).reshape #[1, outC, frames]
    (by rw [convOutputSize_same hf (by decide)])
  let h3 : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } := m.norm2.forward h2 style
  let h4 : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } := leakyRelu h3 0.2
  let h5 : Tensor { shape := #[1, outC, frames], dtype := .Float32, device := device } := (m.conv2.forward h4).reshape #[1, outC, frames]
    (by rw [convOutputSize_same hf (by decide)])
  (h5 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dSame

structure AdainResBlk1dUp (inC outC styleDim : Nat) where
  pool : WNDepthwiseTransConv1dParams (device := device) inC 3 2 1 1 1
  conv1 : WNConv1dParams (device := device) inC outC 3 1 1 1
  conv2 : WNConv1dParams (device := device) outC outC 3 1 1 1
  norm1 : AdaIN1d (device := device) styleDim inC
  norm2 : AdaIN1d (device := device) styleDim outC
  shortcut : Shortcut (device := device) inC outC
  deriving TensorStruct, Inhabited

namespace AdainResBlk1dUp

def init (inC outC styleDim : Nat) (device : Device := .CPU) : IO (AdainResBlk1dUp (device := device) inC outC styleDim) := do
  let pool ← WNDepthwiseTransConv1dParams.init (device := device) inC 3 2 1 1 1
  let conv1 ← WNConv1dParams.init (device := device) inC outC 3 1 1 1
  let conv2 ← WNConv1dParams.init (device := device) outC outC 3 1 1 1
  let norm1 ← AdaIN1d.init (device := device) styleDim inC
  let norm2 ← AdaIN1d.init (device := device) styleDim outC
  let shortcut ← Shortcut.init (device := device) inC outC
  pure { pool, conv1, conv2, norm1, norm2, shortcut }

def forward {frames inC outC styleDim : Nat}
    (m : AdainResBlk1dUp (device := device) inC outC styleDim)
    (x : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } :=
  have h2f : 0 < 2 * frames := by omega
  let shortcut0 : Tensor { shape := #[1, inC, 2 * frames], dtype := .Float32, device := device } := (repeatNearest1d (scale := 2) x).reshape #[1, inC, 2 * frames]
  let shortcut : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } := m.shortcut.forward shortcut0 h2f
  let h0 : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device } := m.norm1.forward x style
  let h1 : Tensor { shape := #[1, inC, frames], dtype := .Float32, device := device } := leakyRelu h0 0.2
  let h2 : Tensor { shape := #[1, inC, 2 * frames], dtype := .Float32, device := device } := (m.pool.forward h1).reshape #[1, inC, 2 * frames]
    (by rw [convTransposeOutputSize_double hf])
  let h3 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } := (m.conv1.forward h2).reshape #[1, outC, 2 * frames]
    (by rw [convOutputSize_same h2f (by decide)])
  let h4 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } := m.norm2.forward h3 style
  let h5 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } := leakyRelu h4 0.2
  let h6 : Tensor { shape := #[1, outC, 2 * frames], dtype := .Float32, device := device } := (m.conv2.forward h5).reshape #[1, outC, 2 * frames]
    (by rw [convOutputSize_same h2f (by decide)])
  (h6 + shortcut) / Float.sqrt 2.0

end AdainResBlk1dUp

structure ProsodyPredictor (cfg : KittenTTSConfig) where
  durationEncoder : DurationEncoder (device := device) cfg
  lstm : BiLSTM (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  durationProj : LinearNorm (device := device) cfg.hiddenDim cfg.maxDur
  shared : BiLSTM (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  f0Blk0 : AdainResBlk1dSame (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  f0Blk1 : AdainResBlk1dUp (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  f0Blk2 : AdainResBlk1dSame (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  nBlk0 : AdainResBlk1dSame (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  nBlk1 : AdainResBlk1dUp (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  nBlk2 : AdainResBlk1dSame (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  f0Proj : Conv1dParams (device := device) (cfg.hiddenDim / 2) 1 1 1 0 1
  nProj : Conv1dParams (device := device) (cfg.hiddenDim / 2) 1 1 1 0 1
  deriving TensorStruct

namespace ProsodyPredictor

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (ProsodyPredictor (device := device) cfg) := do
  let durationEncoder ← DurationEncoder.init (device := device) cfg
  let lstm ← BiLSTM.init (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let durationProj ← LinearNorm.init (device := device) cfg.hiddenDim cfg.maxDur
  let shared ← BiLSTM.init (device := device) (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  let f0Blk0 ← AdainResBlk1dSame.init (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let f0Blk1 ← AdainResBlk1dUp.init (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let f0Blk2 ← AdainResBlk1dSame.init (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk0 ← AdainResBlk1dSame.init (device := device) cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let nBlk1 ← AdainResBlk1dUp.init (device := device) cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk2 ← AdainResBlk1dSame.init (device := device) (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  let f0Proj ← Conv1dParams.init (device := device) (cfg.hiddenDim / 2) 1 1 1 0 1
  let nProj ← Conv1dParams.init (device := device) (cfg.hiddenDim / 2) 1 1 1 0 1
  pure {
    durationEncoder, lstm, durationProj, shared,
    f0Blk0, f0Blk1, f0Blk2, nBlk0, nBlk1, nBlk2, f0Proj, nProj
  }

def durationEncoding {seq : Nat}
    (m : ProsodyPredictor (device := device) cfg)
    (dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device } :=
  m.durationEncoder.forward dEn style

def durationLogits {seq : Nat}
    (m : ProsodyPredictor (device := device) cfg)
    (enc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32, device := device } :=
  let hSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := (m.lstm.forward (cfToSeq enc)).reshape #[1, seq, cfg.hiddenDim] (by rw [Nat.mul_div_cancel' cfg.hiddenDimEven])
  m.durationProj.forward3d hSeq

def forwardF0N {frames : Nat}
    (m : ProsodyPredictor (device := device) cfg)
    (enc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device } × Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device } :=
  have h2f : 0 < 2 * frames := by omega
  let sharedSeq : Tensor { shape := #[1, frames, cfg.hiddenDim], dtype := .Float32, device := device } := (m.shared.forward (cfToSeq enc)).reshape #[1, frames, cfg.hiddenDim] (by rw [Nat.mul_div_cancel' cfg.hiddenDimEven])
  let sharedCF : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device } := seqToCF sharedSeq

  let f0a : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device } := m.f0Blk0.forward sharedCF style hf
  let f0b : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32, device := device } := m.f0Blk1.forward f0a style hf
  let f0c : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32, device := device } := m.f0Blk2.forward f0b style h2f
  let f0 : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device } := (m.f0Proj.forward f0c).reshape #[1, 1, 2 * frames]
    (by rw [convOutputSize_pointwise h2f])

  let na : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device } := m.nBlk0.forward sharedCF style hf
  let nb : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32, device := device } := m.nBlk1.forward na style hf
  let nc : Tensor { shape := #[1, cfg.hiddenDim / 2, 2 * frames], dtype := .Float32, device := device } := m.nBlk2.forward nb style h2f
  let n : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device } := (m.nProj.forward nc).reshape #[1, 1, 2 * frames]
    (by rw [convOutputSize_pointwise h2f])

  (f0, n)

end ProsodyPredictor

private def reflectPadLeft1 {channels frames : Nat}
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, channels, frames + 1], dtype := .Float32, device := device } :=
  let pad : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device } :=
    if h : frames ≤ 1 then
      Tensor.slice x 2 0 1
    else
      Tensor.slice x 2 1 1
  (Tensor.cat pad x 2).reshape #[1, channels, frames + 1] (by simp [Nat.add_comm])

private def harmonicMultipliers (harmonicCount : Nat) : Tensor { shape := #[harmonicCount + 1], dtype := .Float32, device := device } :=
  let vals : Array Float := Array.ofFn (n := harmonicCount + 1) fun i => i.val.toFloat + 1.0
  (Tensor.fromFloatArrayShaped vals #[harmonicCount + 1] (by simp [vals])).to device

private theorem convOutputSize_noise {m S : Nat} (hm : 0 < m) (hS : 2 ≤ S) :
    convOutputSize (m * S + 1) (2 * S) S ((S + 1) / 2) 1 = m := by
  simp only [convOutputSize]
  obtain ⟨q, rfl⟩ : ∃ q, m = q + 1 := ⟨m - 1, by omega⟩
  have hr : 1 + (2 * ((S + 1) / 2) - S) < S := by omega
  have hN : (q + 1) * S + 1 + 2 * ((S + 1) / 2) - (1 * (2 * S - 1) + 1)
      = q * S + (1 + (2 * ((S + 1) / 2) - S)) := by
    rw [Nat.succ_mul]; omega
  rw [hN, Nat.mul_comm q S, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hr]

private theorem convTransposeOutputSize_upsample {L k r : Nat} (hL : 0 < L) (hr : 0 < r) (hrk : r ≤ k)
    (hpar : 2 ∣ k - r) : convTransposeOutputSize L k r ((k - r) / 2) 0 1 = L * r := by
  simp only [convTransposeOutputSize]
  obtain ⟨g, rfl⟩ : ∃ g, L = g + 1 := ⟨L - 1, by omega⟩
  obtain ⟨c, hc⟩ := hpar
  have h1 : (g + 1) * r = g * r + r := by rw [Nat.add_mul, Nat.one_mul]
  simp only [Nat.add_sub_cancel]
  rw [h1]; omega

namespace GeneratorConfig

theorem prefix_zero (g : GeneratorConfig) : g.prefixUpsample 0 = 1 := by
  simp [prefixUpsample]

theorem stageInFrames_pos (g : GeneratorConfig) {b : Nat} (hb : 0 < b) (i : Nat) :
    0 < g.stageInFrames b i :=
  Nat.mul_pos hb (g.prefix_pos i)

theorem stageOutFrames_pos (g : GeneratorConfig) {b : Nat} (hb : 0 < b) (i : Nat) :
    0 < g.stageOutFrames b i := by
  unfold stageOutFrames stageUpsampledFrames harFrames
  split
  · exact Nat.mul_pos hb (g.prefix_pos _)
  · omega

/-- The noise branch lands on the stage's output frame count. -/
theorem noise_frames (g : GeneratorConfig) {b : Nat} (hb : 0 < b) (i : Nat) :
    convOutputSize (g.harFrames b) (g.noiseKernel i) (g.noiseStride i) (g.noisePadding i) 1 =
      g.stageOutFrames b i := by
  by_cases h : i + 1 < g.upsampleRates.size
  · simp only [noiseKernel, noiseStride, noisePadding, stageOutFrames, stageUpsampledFrames,
      harFrames, if_pos h]
    rw [← g.prefix_mul_suffix (i + 1),
      show g.prefixUpsample (i + 1) * g.suffixUpsample (i + 1) * b + 1 =
        b * g.prefixUpsample (i + 1) * g.suffixUpsample (i + 1) + 1 by
          rw [Nat.mul_comm _ b, Nat.mul_assoc]]
    exact convOutputSize_noise (Nat.mul_pos hb (g.prefix_pos _)) (g.suffix_ge_two h)
  · simp only [noiseKernel, noiseStride, noisePadding, stageOutFrames, harFrames, if_neg h]
    simp only [convOutputSize]; omega

theorem noise_valid (g : GeneratorConfig) {b : Nat} (hb : 0 < b) (i : Nat) :
    Tensor.ConvValid (g.harFrames b) (g.noiseKernel i) (g.noiseStride i) (g.noisePadding i) 1 := by
  unfold Tensor.ConvValid
  by_cases h : i + 1 < g.upsampleRates.size
  · have hs := g.suffix_pos (i + 1)
    have hle : g.suffixUpsample (i + 1) ≤ g.totalUpsample * b := by
      rw [← g.prefix_mul_suffix (i + 1)]
      exact Nat.le_trans (Nat.le_mul_of_pos_left _ (g.prefix_pos (i + 1))) (Nat.le_mul_of_pos_right _ hb)
    simp only [noiseKernel, noiseStride, noisePadding, harFrames, if_pos h]
    omega
  · simp only [noiseKernel, noiseStride, noisePadding, harFrames, if_neg h]
    omega

theorem resDilations_pos (g : GeneratorConfig) {j : Nat} (hj : j < g.resblockKernelSizes.size) :
    0 < g.resDilation j 0 ∧ 0 < g.resDilation j 1 ∧ 0 < g.resDilation j 2 := by
  have hv := (g.dilationsValid.2 j (by rw [g.dilationsValid.1]; exact hj)).2
  exact ⟨hv 0 (by decide), hv 1 (by decide), hv 2 (by decide)⟩

theorem up_valid (g : GeneratorConfig) {b : Nat} (hb : 0 < b) {i : Nat}
    (hi : i < g.upsampleRates.size) :
    Tensor.ConvTransposeValid (g.stageInFrames b i) (g.upKernel i) (g.rate i)
      ((g.upKernel i - g.rate i) / 2) 0 1 := by
  have hv := g.stagesValid.2 i hi
  have hf := g.stageInFrames_pos hb i
  unfold upKernel rate
  unfold Tensor.ConvTransposeValid
  omega

/-- The upsampling transposed convolution multiplies the frame count by the stage rate. -/
theorem up_frames (g : GeneratorConfig) {b : Nat} (hb : 0 < b) {i : Nat}
    (hi : i < g.upsampleRates.size) :
    convTransposeOutputSize (g.stageInFrames b i) (g.upKernel i) (g.rate i)
      ((g.upKernel i - g.rate i) / 2) 0 1 = g.stageUpsampledFrames b i := by
  have hv := g.stagesValid.2 i hi
  unfold upKernel rate
  rw [convTransposeOutputSize_upsample (g.stageInFrames_pos hb i) (Nat.zero_lt_of_lt hv.1) hv.2.1 hv.2.2]
  simp only [stageInFrames, stageUpsampledFrames, g.prefix_succ hi, rate, Nat.mul_assoc]

theorem mid_frames (g : GeneratorConfig) (b : Nat) {i : Nat} (h : i + 1 < g.upsampleRates.size) :
    g.stageOutFrames b i = g.stageInFrames b (i + 1) := by
  simp only [stageOutFrames, stageUpsampledFrames, stageInFrames, if_pos h]

theorem last_frames (g : GeneratorConfig) (b : Nat) {i : Nat} (h : i + 1 = g.upsampleRates.size) :
    g.stageUpsampledFrames b i + 1 = g.stageOutFrames b i := by
  simp only [stageOutFrames, if_neg (show ¬ i + 1 < g.upsampleRates.size by omega),
    stageUpsampledFrames, harFrames]
  rw [h, g.prefix_size, Nat.mul_comm]

end GeneratorConfig

/-- One dilated residual unit of a generator AdaIN block. -/
structure GenResUnit (styleDim channels kernel dilation : Nat) where
  conv1 : WNConv1dParams (device := device) channels channels kernel 1 ((kernel * dilation - dilation) / 2) dilation
  conv2 : WNConv1dParams (device := device) channels channels kernel 1 ((kernel - 1) / 2) 1
  norm1 : AdaIN1d (device := device) styleDim channels
  norm2 : AdaIN1d (device := device) styleDim channels
  alpha1 : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device }
  alpha2 : Tensor { shape := #[1, channels, 1], dtype := .Float32, device := device }
  deriving TensorStruct, Inhabited

namespace GenResUnit

def init (styleDim channels kernel dilation : Nat) (device : Device := .CPU) :
    IO (GenResUnit (device := device) styleDim channels kernel dilation) := do
  let conv1 ← WNConv1dParams.init (device := device) channels channels kernel 1 ((kernel * dilation - dilation) / 2) dilation
  let conv2 ← WNConv1dParams.init (device := device) channels channels kernel 1 ((kernel - 1) / 2) 1
  let norm1 ← AdaIN1d.init (device := device) styleDim channels
  let norm2 ← AdaIN1d.init (device := device) styleDim channels
  pure {
    conv1, conv2, norm1, norm2
    alpha1 := Tensor.ones #[1, channels, 1] (device := device)
    alpha2 := Tensor.ones #[1, channels, 1] (device := device)
  }

def forward {frames styleDim channels kernel dilation : Nat}
    (m : GenResUnit (device := device) styleDim channels kernel dilation)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames) (hk : kernel % 2 = 1) (hd : 0 < dilation)
    : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
  have hPad := dilated_same_pad (d := dilation) hk
  let h2 := snake1d (m.norm1.forward x style) m.alpha1
  let h3 : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
    (m.conv1.forward h2).reshape #[1, channels, frames]
      (by rw [convOutputSize_same hf (dilated_same_pad hk)])
  let h5 := snake1d (m.norm2.forward h3 style) m.alpha2
  let h6 : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
    (m.conv2.forward h5).reshape #[1, channels, frames]
      (by rw [convOutputSize_same hf (by omega)])
  x + h6

end GenResUnit

/-- Generator AdaIN residual block: three dilated units sharing one odd kernel. -/
structure GeneratorAdaINResBlock (styleDim channels kernel d0 d1 d2 : Nat) where
  u0 : GenResUnit (device := device) styleDim channels kernel d0
  u1 : GenResUnit (device := device) styleDim channels kernel d1
  u2 : GenResUnit (device := device) styleDim channels kernel d2
  deriving TensorStruct, Inhabited

namespace GeneratorAdaINResBlock

def init (styleDim channels kernel d0 d1 d2 : Nat) (device : Device := .CPU) :
    IO (GeneratorAdaINResBlock (device := device) styleDim channels kernel d0 d1 d2) := do
  pure {
    u0 := ← GenResUnit.init (device := device) styleDim channels kernel d0
    u1 := ← GenResUnit.init (device := device) styleDim channels kernel d1
    u2 := ← GenResUnit.init (device := device) styleDim channels kernel d2
  }

def forward {frames styleDim channels kernel d0 d1 d2 : Nat}
    (m : GeneratorAdaINResBlock (device := device) styleDim channels kernel d0 d1 d2)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames) (hk : kernel % 2 = 1) (hd : 0 < d0 ∧ 0 < d1 ∧ 0 < d2)
    : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
  m.u2.forward (m.u1.forward (m.u0.forward x style hf hk hd.1) style hf hk hd.2.1) style hf hk hd.2.2

end GeneratorAdaINResBlock

/-- Residual block of a generator stage, as configured by `resblock_kernel_sizes[j]`. -/
abbrev GeneratorResBlock (g : GeneratorConfig) (styleDim channels j : Nat) :=
  GeneratorAdaINResBlock (device := device) styleDim channels (g.resKernel j)
    (g.resDilation j 0) (g.resDilation j 1) (g.resDilation j 2)

/-- The residual blocks `j, j+1, …` of a generator stage; never empty. -/
inductive StageResBlocks (g : GeneratorConfig) (styleDim channels : Nat) : Nat → Type where
  | last {j : Nat} (hj : j < g.resblockKernelSizes.size)
      (blk : GeneratorResBlock (device := device) g styleDim channels j) : StageResBlocks g styleDim channels j
  | cons {j : Nat} (hj : j < g.resblockKernelSizes.size)
      (blk : GeneratorResBlock (device := device) g styleDim channels j)
      (rest : StageResBlocks g styleDim channels (j + 1)) : StageResBlocks g styleDim channels j

namespace StageResBlocks

variable {g : GeneratorConfig} {styleDim channels : Nat}

def map (f : ∀ {σ}, Tensor σ → Tensor σ) :
    {j : Nat} → StageResBlocks (device := device) g styleDim channels j → StageResBlocks (device := device) g styleDim channels j
  | _, .last hj b => .last hj (TensorStruct.map f b)
  | _, .cons hj b r => .cons hj (TensorStruct.map f b) (map f r)

def mapM {m : Type → Type} [Monad m] (f : ∀ {σ}, Tensor σ → m (Tensor σ)) :
    {j : Nat} → StageResBlocks (device := device) g styleDim channels j → m (StageResBlocks (device := device) g styleDim channels j)
  | _, .last hj b => do pure (.last hj (← TensorStruct.mapM f b))
  | _, .cons hj b r => do pure (.cons hj (← TensorStruct.mapM f b) (← mapM f r))

def zipWith (f : ∀ {σ}, Tensor σ → Tensor σ → Tensor σ) :
    {j : Nat} → StageResBlocks (device := device) g styleDim channels j →
      StageResBlocks (device := device) g styleDim channels j → StageResBlocks (device := device) g styleDim channels j
  | _, .last hj a, .last _ b => .last hj (TensorStruct.zipWith f a b)
  | _, .cons hj a ra, .cons _ b rb => .cons hj (TensorStruct.zipWith f a b) (zipWith f ra rb)
  | _, lhs, _ => lhs

def fold {β : Type} (f : ∀ {σ}, Tensor σ → β → β) (init : β) :
    {j : Nat} → StageResBlocks (device := device) g styleDim channels j → β
  | _, .last _ b => TensorStruct.fold f init b
  | _, .cons _ b r => fold f (TensorStruct.fold f init b) r

instance {j : Nat} : TensorStruct (StageResBlocks (device := device) g styleDim channels j) where
  map f x := map f x
  mapM f x := mapM f x
  zipWith f x y := zipWith f x y
  fold f init x := fold f init x

def init (g : GeneratorConfig) (styleDim channels : Nat) (device : Device := .CPU) :
    (j : Nat) → j < g.resblockKernelSizes.size → IO (StageResBlocks (device := device) g styleDim channels j)
  | j, hj => do
    let blk ← GeneratorAdaINResBlock.init (device := device) styleDim channels (g.resKernel j)
      (g.resDilation j 0) (g.resDilation j 1) (g.resDilation j 2)
    if hNext : j + 1 < g.resblockKernelSizes.size then
      pure (.cons hj blk (← init g styleDim channels device (j + 1) hNext))
    else
      pure (.last hj blk)
termination_by j => g.resblockKernelSizes.size - j

private def accumulate {frames : Nat}
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    (acc : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device }) (count : Nat) :
    {j : Nat} → StageResBlocks (device := device) g styleDim channels j →
      Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } × Nat
  | _, .last hj b => (acc + b.forward x style hf (g.resblocksValid.2 _ hj) (g.resDilations_pos hj), count + 1)
  | _, .cons hj b r =>
    accumulate x style hf (acc + b.forward x style hf (g.resblocksValid.2 _ hj) (g.resDilations_pos hj)) (count + 1) r

/-- Mean of the block outputs, summed left to right. -/
def forwardMean {frames j : Nat}
    (blocks : StageResBlocks (device := device) g styleDim channels j)
    (x : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : Tensor { shape := #[1, channels, frames], dtype := .Float32, device := device } :=
  let (sum, count) :=
    match blocks with
    | .last hj b => (b.forward x style hf (g.resblocksValid.2 _ hj) (g.resDilations_pos hj), 1)
    | .cons hj b r =>
      accumulate x style hf (b.forward x style hf (g.resblocksValid.2 _ hj) (g.resDilations_pos hj)) 1 r
  sum / count.toFloat

end StageResBlocks

structure SourceModuleHnNSF (cfg : KittenTTSConfig) where
  linear : LinearNorm (device := device) (cfg.generator.harmonicCount + 1) 1
  deriving TensorStruct

namespace SourceModuleHnNSF

private def sineAmp : Float := 0.1
private def noiseStd : Float := 0.003
private def voicedThreshold : Float := 10.0

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (SourceModuleHnNSF (device := device) cfg) := do
  let linear ← LinearNorm.init (device := device) (cfg.generator.harmonicCount + 1) 1
  pure { linear }

def forward {baseFrames : Nat}
    (m : SourceModuleHnNSF (device := device) cfg)
    (f0 : Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32, device := device })
    (hb : 0 < baseFrames)
    : IO
      (Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32, device := device } ×
       Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32, device := device } ×
       Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32, device := device }) := do
  let waveFrames := GeneratorConfig.waveSamples cfg.generator baseFrames
  have _hWave : 1 ≤ waveFrames := cfg.generator.waveSamples_pos hb
  let harmonics := cfg.generator.harmonicCount + 1
  let mult : Tensor { shape := #[1, 1, harmonics], dtype := .Float32, device := device } := Tensor.reshape (harmonicMultipliers cfg.generator.harmonicCount) #[1, 1, harmonics]
  let multE : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := Tensor.broadcastTo mult #[1, waveFrames, harmonics]
  let f0E : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } :=
    Tensor.broadcastTo f0 #[1, waveFrames, harmonics] (by simp only [waveFrames]; broadcast_shape)
  let rad0 : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := (f0E * multE) / cfg.sampleRate.toFloat
  let radBase : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := rad0 - floorTensor rad0

  let randIni : Tensor { shape := #[1, harmonics], dtype := .Float32, device := device } ← randTensor #[1, harmonics] false f0.device
  let randIni : Tensor { shape := #[1, harmonics], dtype := .Float32, device := device } :=
    Tensor.sliceScatter randIni 1 0 (Tensor.zeros #[1, 1] false f0.device)
  let phaseNoise0 : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := Tensor.zeros #[1, waveFrames, harmonics] false f0.device
  let phaseNoise : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } :=
    Tensor.sliceScatter phaseNoise0 1 0 (Tensor.reshape randIni #[1, 1, harmonics])
  let radInit : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := radBase + phaseNoise

  let radCF : Tensor { shape := #[1, harmonics, waveFrames], dtype := .Float32, device := device } := seqToCF radInit
  let radDownCF : Tensor { shape := #[1, harmonics, baseFrames], dtype := .Float32, device := device } := resize1d (outFrames := baseFrames) radCF "linear"
  let radDown : Tensor { shape := #[1, baseFrames, harmonics], dtype := .Float32, device := device } := cfToSeq radDownCF
  let phaseBase : Tensor { shape := #[1, baseFrames, harmonics], dtype := .Float32, device := device } := Tensor.mulScalar (cumsumTensor radDown 1) 6.283185307179586
  let phaseScaledCF : Tensor { shape := #[1, harmonics, baseFrames], dtype := .Float32, device := device } :=
    seqToCF phaseBase * (GeneratorConfig.sourceUpsample cfg.generator).toFloat
  let phaseUpCF : Tensor { shape := #[1, harmonics, waveFrames], dtype := .Float32, device := device } := resize1d (outFrames := waveFrames) phaseScaledCF "linear"
  let sines : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := Tensor.sin (cfToSeq phaseUpCF)
  let sineWaves : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := sines * sineAmp

  let threshold : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32, device := device } := Tensor.full #[1, waveFrames, 1] voicedThreshold false f0.device
  let uv : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32, device := device } := Tensor.toFloat32 (Tensor.lt threshold f0)
  let uvE : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := Tensor.broadcastTo uv #[1, waveFrames, harmonics]
  let noiseAmp : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } :=
    uvE * noiseStd + (Tensor.onesLike uvE - uvE) * (sineAmp / 3.0)
  let noise : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } ← Tensor.randn #[1, waveFrames, harmonics] false f0.device
  let sineNoisy : Tensor { shape := #[1, waveFrames, harmonics], dtype := .Float32, device := device } := sineWaves * uvE + noiseAmp * noise

  let sineMerge : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32, device := device } := Tensor.tanh (m.linear.forward3d sineNoisy)
  let noiseOut : Tensor { shape := #[1, waveFrames, 1], dtype := .Float32, device := device } ← Tensor.randn #[1, waveFrames, 1] false f0.device
  pure (sineMerge, noiseOut * (sineAmp / 3.0), uv)

end SourceModuleHnNSF

namespace GeneratorConfig

/-- Kernel of a stage's noise residual block. -/
def noiseResKernel (g : GeneratorConfig) (i : Nat) : Nat :=
  if i + 1 < g.upsampleRates.size then 7 else 11

theorem noiseResKernel_odd (g : GeneratorConfig) (i : Nat) : g.noiseResKernel i % 2 = 1 := by
  unfold noiseResKernel; split <;> decide

end GeneratorConfig

/-- Upsample stage `i` of the generator, typed by the channel counts and kernels the config gives it. -/
structure GeneratorStage (cfg : KittenTTSConfig) (i : Nat) where
  up : WNConvTranspose1dParams (device := device)
    (GeneratorConfig.stageInChannels cfg.generator i) (GeneratorConfig.stageOutChannels cfg.generator i)
    (GeneratorConfig.upKernel cfg.generator i) (GeneratorConfig.rate cfg.generator i)
    ((GeneratorConfig.upKernel cfg.generator i - GeneratorConfig.rate cfg.generator i) / 2) 0 1
  noiseConv : WNConv1dParams (device := device)
    (GeneratorConfig.stftFeatureChannels cfg.generator) (GeneratorConfig.stageOutChannels cfg.generator i)
    (GeneratorConfig.noiseKernel cfg.generator i) (GeneratorConfig.noiseStride cfg.generator i)
    (GeneratorConfig.noisePadding cfg.generator i) 1
  noiseRes : GeneratorAdaINResBlock (device := device) cfg.styleDim
    (GeneratorConfig.stageOutChannels cfg.generator i) (GeneratorConfig.noiseResKernel cfg.generator i) 1 3 5
  resBlocks : StageResBlocks (device := device) cfg.generator cfg.styleDim (GeneratorConfig.stageOutChannels cfg.generator i) 0
  deriving TensorStruct

structure GeneratorStageDebugOutput where
  xSource : ShapePackedTensor .Float32 device
  xUp : ShapePackedTensor .Float32 device
  xMix : ShapePackedTensor .Float32 device
  xOut : ShapePackedTensor .Float32 device
  deriving TensorStruct, Inhabited

structure GeneratorDebugOutput where
  harSource : ShapePackedTensor .Float32 device
  har : ShapePackedTensor .Float32 device
  stages : Array (GeneratorStageDebugOutput (device := device))
  post : ShapePackedTensor .Float32 device
  specLog : ShapePackedTensor .Float32 device
  phaseRaw : ShapePackedTensor .Float32 device
  spec : ShapePackedTensor .Float32 device
  audio : ShapePackedTensor .Float32 device
  deriving TensorStruct, Inhabited

namespace GeneratorStage

def init (cfg : KittenTTSConfig) (i : Nat) (device : Device := .CPU) : IO (GeneratorStage (device := device) cfg i) := do
  let g := cfg.generator
  let inC := GeneratorConfig.stageInChannels g i
  let outC := GeneratorConfig.stageOutChannels g i
  let kernel := GeneratorConfig.upKernel g i
  let stride := GeneratorConfig.rate g i
  pure {
    up := ← WNConvTranspose1dParams.init (device := device) inC outC kernel stride ((kernel - stride) / 2) 0 1
    noiseConv := ← WNConv1dParams.init (device := device) (GeneratorConfig.stftFeatureChannels g) outC
      (GeneratorConfig.noiseKernel g i) (GeneratorConfig.noiseStride g i) (GeneratorConfig.noisePadding g i) 1
    noiseRes := ← GeneratorAdaINResBlock.init (device := device) cfg.styleDim outC (GeneratorConfig.noiseResKernel g i) 1 3 5
    resBlocks := ← StageResBlocks.init (device := device) g cfg.styleDim outC 0 g.resblocksValid.1
  }

def forward {i baseFrames : Nat}
    (st : GeneratorStage (device := device) cfg i)
    (hi : i < cfg.generator.upsampleRates.size) (hb : 0 < baseFrames)
    (x : Tensor { shape := #[1, GeneratorConfig.stageInChannels cfg.generator i, GeneratorConfig.stageInFrames cfg.generator baseFrames i], dtype := .Float32, device := device })
    (har : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    : Tensor { shape := #[1, GeneratorConfig.stageOutChannels cfg.generator i, GeneratorConfig.stageOutFrames cfg.generator baseFrames i], dtype := .Float32, device := device } ×
      GeneratorStageDebugOutput (device := device) :=
  let outC := GeneratorConfig.stageOutChannels cfg.generator i
  let outFrames := GeneratorConfig.stageOutFrames cfg.generator baseFrames i
  have hOut : 0 < outFrames := cfg.generator.stageOutFrames_pos hb i
  let xAct := leakyRelu x 0.1
  let xSource1 : Tensor { shape := #[1, outC, outFrames], dtype := .Float32, device := device } :=
    (st.noiseConv.forward har (cfg.generator.noise_valid hb i)).reshape #[1, outC, outFrames]
      (by rw [cfg.generator.noise_frames hb i])
  let xSource := st.noiseRes.forward xSource1 style hOut (cfg.generator.noiseResKernel_odd i)
    ⟨by decide, by decide, by decide⟩
  let xUp0 := st.up.forward xAct (cfg.generator.up_valid hb hi)
  let xUp : Tensor { shape := #[1, outC, outFrames], dtype := .Float32, device := device } :=
    if hMid : i + 1 < cfg.generator.upsampleRates.size then
      xUp0.reshape #[1, outC, outFrames]
        (by rw [cfg.generator.up_frames hb hi]; simp only [outC, outFrames, GeneratorConfig.stageOutFrames, if_pos hMid])
    else
      (reflectPadLeft1 xUp0
        (by rw [cfg.generator.up_frames hb hi]; exact Nat.mul_pos hb (cfg.generator.prefix_pos _))).reshape #[1, outC, outFrames]
        (by rw [cfg.generator.up_frames hb hi, cfg.generator.last_frames baseFrames (by omega)])
  let xMix := xUp + xSource
  let xOut := st.resBlocks.forwardMean xMix style hOut
  (xOut, {
    xSource := Tensor.packShape xSource
    xUp := Tensor.packShape xUp
    xMix := Tensor.packShape xMix
    xOut := Tensor.packShape xOut
  })

end GeneratorStage

/-- Generator stages `i, i+1, …` through the last configured stage. -/
inductive GeneratorStages (cfg : KittenTTSConfig) : Nat → Type where
  | last {i : Nat} (h : i + 1 = cfg.generator.upsampleRates.size)
      (stage : GeneratorStage (device := device) cfg i) : GeneratorStages cfg i
  | cons {i : Nat} (h : i + 1 < cfg.generator.upsampleRates.size)
      (stage : GeneratorStage (device := device) cfg i) (rest : GeneratorStages cfg (i + 1)) : GeneratorStages cfg i

namespace GeneratorStages

def map (f : ∀ {σ}, Tensor σ → Tensor σ) :
    {i : Nat} → GeneratorStages (device := device) cfg i → GeneratorStages (device := device) cfg i
  | _, .last h s => .last h (TensorStruct.map f s)
  | _, .cons h s r => .cons h (TensorStruct.map f s) (map f r)

def mapM {m : Type → Type} [Monad m] (f : ∀ {σ}, Tensor σ → m (Tensor σ)) :
    {i : Nat} → GeneratorStages (device := device) cfg i → m (GeneratorStages (device := device) cfg i)
  | _, .last h s => do pure (.last h (← TensorStruct.mapM f s))
  | _, .cons h s r => do pure (.cons h (← TensorStruct.mapM f s) (← mapM f r))

def zipWith (f : ∀ {σ}, Tensor σ → Tensor σ → Tensor σ) :
    {i : Nat} → GeneratorStages (device := device) cfg i → GeneratorStages (device := device) cfg i →
      GeneratorStages (device := device) cfg i
  | _, .last h a, .last _ b => .last h (TensorStruct.zipWith f a b)
  | _, .cons h a ra, .cons _ b rb => .cons h (TensorStruct.zipWith f a b) (zipWith f ra rb)
  | _, lhs, _ => lhs

def fold {β : Type} (f : ∀ {σ}, Tensor σ → β → β) (init : β) :
    {i : Nat} → GeneratorStages (device := device) cfg i → β
  | _, .last _ s => TensorStruct.fold f init s
  | _, .cons _ s r => fold f (TensorStruct.fold f init s) r

instance {i : Nat} : TensorStruct (GeneratorStages (device := device) cfg i) where
  map f x := map f x
  mapM f x := mapM f x
  zipWith f x y := zipWith f x y
  fold f init x := fold f init x

def init (cfg : KittenTTSConfig) (device : Device := .CPU) :
    (i : Nat) → i < cfg.generator.upsampleRates.size → IO (GeneratorStages (device := device) cfg i)
  | i, hi => do
    let stage ← GeneratorStage.init (device := device) cfg i
    if hNext : i + 1 < cfg.generator.upsampleRates.size then
      pure (.cons hNext stage (← init cfg device (i + 1) hNext))
    else
      pure (.last (by omega) stage)
termination_by i => cfg.generator.upsampleRates.size - i

/-- Run stages `i, …`, returning the last stage's output and every stage's debug capture. -/
def run {baseFrames : Nat} (hb : 0 < baseFrames)
    (har : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device }) :
    {i : Nat} → GeneratorStages (device := device) cfg i →
      Tensor { shape := #[1, GeneratorConfig.stageInChannels cfg.generator i, GeneratorConfig.stageInFrames cfg.generator baseFrames i], dtype := .Float32, device := device } →
      Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } ×
        Array (GeneratorStageDebugOutput (device := device))
  | i, .last h st, x =>
    let (y, d) := st.forward (by omega) hb x har style
    (y.reshape #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
      (by rw [cfg.generator.finalChannels_last h]
          simp only [GeneratorConfig.stageOutFrames, if_neg (show ¬ (i + 1 < cfg.generator.upsampleRates.size) by omega)]), #[d])
  | _, .cons h st rest, x =>
    let (y, d) := st.forward (by omega) hb x har style
    let (out, ds) := run hb har style rest
      (y.reshape _ (by rw [cfg.generator.mid_frames baseFrames h]; rfl))
    (out, #[d] ++ ds)

end GeneratorStages

structure Generator (cfg : KittenTTSConfig) where
  source : SourceModuleHnNSF (device := device) cfg
  stages : GeneratorStages (device := device) cfg 0
  convPost : WNConv1dParams (device := device) (GeneratorConfig.finalChannels cfg.generator) (cfg.generator.genIstftNFft + 2) 7 1 3 1
  window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32, device := device }
  deriving TensorStruct

namespace Generator

private def buildHarFeatures {baseFrames : Nat}
    (m : Generator (device := device) cfg)
    (harmonic : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32, device := device })
    : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
  let window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32, device := device } := m.window.to harmonic.device
  let stftDyn : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32, device := device } :=
    stftTensor
      (n := GeneratorConfig.waveSamples cfg.generator baseFrames)
      harmonic
      cfg.generator.genIstftNFft
      cfg.generator.genIstftHopSize
      window
      true
      false
  let packed : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32, device := device } :=
    Tensor.reshape stftDyn #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2]
  let re3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32, device := device } :=
    Tensor.slice packed 2 0 1
  let im3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32, device := device } :=
    Tensor.slice packed 2 1 1
  let re : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape re3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let im : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape im3 #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let mag : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.sqrt (re * re + im * im + (1e-12 : Float))
  let phase : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    atan2Tensor im re
  let mag3 : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape mag #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  let phase3 : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape phase #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
  (catTo mag3 phase3 1).reshape #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames]
    (by simp [Nat.two_mul])

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (Generator (device := device) cfg) := do
  let source ← SourceModuleHnNSF.init (device := device) cfg
  let stages ← GeneratorStages.init (device := device) cfg 0 cfg.generator.stagesValid.1
  let convPost ← WNConv1dParams.init (device := device) (GeneratorConfig.finalChannels cfg.generator) (cfg.generator.genIstftNFft + 2) 7 1 3 1
  let window := (Tensor.hannWindow cfg.generator.genIstftNFft).to device
  pure { source, stages, convPost, window }

def debugForward {baseFrames : Nat}
    (m : Generator (device := device) cfg)
    (x : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, baseFrames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    (f0Curve : Tensor { shape := #[1, 1, baseFrames], dtype := .Float32, device := device })
    (hb : 0 < baseFrames)
    : IO (GeneratorDebugOutput (device := device)) := do
  let f0Up : Tensor { shape := #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32, device := device } :=
    repeatNearest1d (scale := GeneratorConfig.sourceUpsample cfg.generator) f0Curve
  let f0Seq : Tensor { shape := #[1, GeneratorConfig.waveSamples cfg.generator baseFrames, 1], dtype := .Float32, device := device } := cfToSeq f0Up
  let (harSource, _noiseSource, _uv) ← m.source.forward f0Seq hb
  let har1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape harSource #[GeneratorConfig.waveSamples cfg.generator baseFrames]
  let har : Tensor { shape := #[1, GeneratorConfig.stftFeatureChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    buildHarFeatures m har1d

  let x0 : Tensor { shape := #[1, GeneratorConfig.stageInChannels cfg.generator 0, GeneratorConfig.stageInFrames cfg.generator baseFrames 0], dtype := .Float32, device := device } :=
    x.reshape _ (by
      simp only [KittenTTSConfig.decoderChannels, cfg.decoderMatchesGenerator, cfg.generator.stageInChannels_zero,
        GeneratorConfig.stageInFrames, cfg.generator.prefix_zero, Nat.mul_one])
  let (xFinal, stageOutputs) := m.stages.run hb har style x0
  let postIn : Tensor { shape := #[1, GeneratorConfig.finalChannels cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } := leakyRelu xFinal 0.01
  let post : Tensor { shape := #[1, cfg.generator.genIstftNFft + 2, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    (m.convPost.forward postIn).reshape _
      (by rw [convOutputSize_same (by simp [GeneratorConfig.harFrames]) (by decide)])

  let freqBins := GeneratorConfig.freqBins cfg.generator
  let specLog : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.slice post 1 0 freqBins (_bounds := by simp [freqBins, GeneratorConfig.freqBins]; omega)
  let phaseRaw : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.slice post 1 freqBins freqBins (_bounds := by simp [freqBins, GeneratorConfig.freqBins]; omega)
  let spec : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } := Tensor.exp specLog
  let phase : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } := Tensor.sin phaseRaw
  let re : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } := spec * Tensor.cos phase
  let im : Tensor { shape := #[1, GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames], dtype := .Float32, device := device } := spec * Tensor.sin phase
  let re3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32, device := device } :=
    Tensor.reshape re #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let im3 : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1], dtype := .Float32, device := device } :=
    Tensor.reshape im #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 1]
  let packed : Tensor { shape := #[GeneratorConfig.freqBins cfg.generator, GeneratorConfig.harFrames cfg.generator baseFrames, 2], dtype := .Float32, device := device } :=
    Tensor.cat re3 im3 2
  let window : Tensor { shape := #[cfg.generator.genIstftNFft], dtype := .Float32, device := device } := m.window.to post.device
  let audio1d : Tensor { shape := #[GeneratorConfig.waveSamples cfg.generator baseFrames], dtype := .Float32, device := device } :=
    Tensor.reshape
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
    audio := Tensor.packShape (Tensor.reshape audio1d #[1, 1, GeneratorConfig.waveSamples cfg.generator baseFrames])
  }

def forward {baseFrames : Nat}
    (m : Generator (device := device) cfg)
    (x : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, baseFrames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    (f0Curve : Tensor { shape := #[1, 1, baseFrames], dtype := .Float32, device := device })
    (hb : 0 < baseFrames)
    : IO (ShapePackedTensor .Float32 device) := do
  pure (← m.debugForward x style f0Curve hb).audio

end Generator

structure DecoderDebugOutput where
  f0 : ShapePackedTensor .Float32 device
  n : ShapePackedTensor .Float32 device
  encode : ShapePackedTensor .Float32 device
  asrRes : ShapePackedTensor .Float32 device
  decode0 : ShapePackedTensor .Float32 device
  decode1 : ShapePackedTensor .Float32 device
  decode2 : ShapePackedTensor .Float32 device
  decode3 : ShapePackedTensor .Float32 device
  generator : (GeneratorDebugOutput (device := device))
  deriving TensorStruct, Inhabited

structure Decoder (cfg : KittenTTSConfig) where
  f0Conv : WNConv1dParams (device := device) 1 1 3 2 1 1
  nConv : WNConv1dParams (device := device) 1 1 3 2 1 1
  encode : AdainResBlk1dSame (device := device) (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  asrRes : WNConv1dParams (device := device) cfg.hiddenDim cfg.asrResDim 1 1 0 1
  decode0 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode1 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode2 : AdainResBlk1dSame (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  decode3 : AdainResBlk1dUp (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  generator : Generator (device := device) cfg
  deriving TensorStruct

namespace Decoder

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (Decoder (device := device) cfg) := do
  let f0Conv ← WNConv1dParams.init (device := device) 1 1 3 2 1 1
  let nConv ← WNConv1dParams.init (device := device) 1 1 3 2 1 1
  let encode ← AdainResBlk1dSame.init (device := device) (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  let asrRes ← WNConv1dParams.init (device := device) cfg.hiddenDim cfg.asrResDim 1 1 0 1
  let decode0 ← AdainResBlk1dSame.init (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode1 ← AdainResBlk1dSame.init (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode2 ← AdainResBlk1dSame.init (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode3 ← AdainResBlk1dUp.init (device := device) (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  let generator ← Generator.init (device := device) cfg
  pure { f0Conv, nConv, encode, asrRes, decode0, decode1, decode2, decode3, generator }

def debugForward {frames : Nat}
    (m : Decoder (device := device) cfg)
    (asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device })
    (f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device })
    (nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : IO (DecoderDebugOutput (device := device)) := do
  let f0 : Tensor { shape := #[1, 1, frames], dtype := .Float32, device := device } := (m.f0Conv.forward f0Curve).reshape #[1, 1, frames] (by rw [convOutputSize_halve hf])
  let n : Tensor { shape := #[1, 1, frames], dtype := .Float32, device := device } := (m.nConv.forward nCurve).reshape #[1, 1, frames] (by rw [convOutputSize_halve hf])
  let x0 : Tensor { shape := #[1, cfg.hiddenDim + 2, frames], dtype := .Float32, device := device } := (catTo asr (catTo f0 n 1) 1).reshape #[1, cfg.hiddenDim + 2, frames] (by simp)
  let x1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.encode.forward x0 style hf
  let asrRes : Tensor { shape := #[1, cfg.asrResDim, frames], dtype := .Float32, device := device } := (m.asrRes.forward asr).reshape #[1, cfg.asrResDim, frames] (by rw [convOutputSize_pointwise hf])

  let d0In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo x1 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d0 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode0.forward d0In style hf
  let d1In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d0 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode1.forward d1In style hf
  let d2In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d1 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d2 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode2.forward d2In style hf
  let d3In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d2 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d3 : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames], dtype := .Float32, device := device } := m.decode3.forward d3In style hf
  let generator ← m.generator.debugForward d3 style f0Curve (by omega)
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

def forward {frames : Nat}
    (m : Decoder (device := device) cfg)
    (asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device })
    (f0Curve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device })
    (nCurve : Tensor { shape := #[1, 1, 2 * frames], dtype := .Float32, device := device })
    (style : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device })
    (hf : 0 < frames)
    : IO (ShapePackedTensor .Float32 device) := do
  let f0 : Tensor { shape := #[1, 1, frames], dtype := .Float32, device := device } := (m.f0Conv.forward f0Curve).reshape #[1, 1, frames] (by rw [convOutputSize_halve hf])
  let n : Tensor { shape := #[1, 1, frames], dtype := .Float32, device := device } := (m.nConv.forward nCurve).reshape #[1, 1, frames] (by rw [convOutputSize_halve hf])
  let x0 : Tensor { shape := #[1, cfg.hiddenDim + 2, frames], dtype := .Float32, device := device } := (catTo asr (catTo f0 n 1) 1).reshape #[1, cfg.hiddenDim + 2, frames] (by simp)
  let x1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.encode.forward x0 style hf
  let asrRes : Tensor { shape := #[1, cfg.asrResDim, frames], dtype := .Float32, device := device } := (m.asrRes.forward asr).reshape #[1, cfg.asrResDim, frames] (by rw [convOutputSize_pointwise hf])
  let d0In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo x1 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d0 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode0.forward d0In style hf
  let d1In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d0 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d1 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode1.forward d1In style hf
  let d2In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d1 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d2 : Tensor { shape := #[1, cfg.maxConvDim, frames], dtype := .Float32, device := device } := m.decode2.forward d2In style hf
  let d3In : Tensor { shape := #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames], dtype := .Float32, device := device } := (catTo d2 (catTo asrRes (catTo f0 n 1) 1) 1).reshape #[1, cfg.maxConvDim + cfg.asrResDim + 2, frames] (by simp [Nat.add_assoc])
  let d3 : Tensor { shape := #[1, KittenTTSConfig.decoderChannels cfg, 2 * frames], dtype := .Float32, device := device } := m.decode3.forward d3In style hf
  m.generator.forward d3 style f0Curve (by omega)

end Decoder

structure KittenTTSOutput where
  audio : ShapePackedTensor .Float32 device
  predDurations : Array Nat
  alignmentFrames : Nat
  deriving TensorStruct

structure KittenTTSDebugOutput (cfg : KittenTTSConfig) where
  alignmentFrames : Nat
  asr : Tensor { shape := #[1, cfg.hiddenDim, alignmentFrames], dtype := .Float32, device := device }
  f0Curve : Tensor { shape := #[1, 1, 2 * alignmentFrames], dtype := .Float32, device := device }
  nCurve : Tensor { shape := #[1, 1, 2 * alignmentFrames], dtype := .Float32, device := device }
  audio : ShapePackedTensor .Float32 device
  predDurations : Array Nat

structure KittenTTSDurationPrediction where
  durVals : Array Float
  predDurations : Array Nat
  alignmentFrames : Nat
  deriving Inhabited, Repr

structure Model (cfg : KittenTTSConfig) where
  bert : KittenAlbert (device := device) cfg.nToken cfg.plbert
  bertEncoder : LinearNorm (device := device) cfg.plbert.hiddenSize cfg.hiddenDim
  predictor : ProsodyPredictor (device := device) cfg
  textEncoder : TextEncoder (device := device) cfg
  decoder : Decoder (device := device) cfg
  deriving TensorStruct

namespace Model

private def modelDevice (m : Model (device := device) cfg) : Device :=
  m.bert.embeddings.wordEmbeddings.device

private structure PreparedSynthesis (cfg : KittenTTSConfig) where
  alignmentFrames : Nat
  framesPos : 0 < alignmentFrames
  decoderStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device }
  asr : Tensor { shape := #[1, cfg.hiddenDim, alignmentFrames], dtype := .Float32, device := device }
  f0Curve : Tensor { shape := #[1, 1, 2 * alignmentFrames], dtype := .Float32, device := device }
  nCurve : Tensor { shape := #[1, 1, 2 * alignmentFrames], dtype := .Float32, device := device }
  predDurations : Array Nat

private def prepareInputs {seq : Nat}
    (m : Model (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device })
    : Tensor { shape := #[1, seq], dtype := .Int64, device := device } × Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device } :=
  let device := modelDevice m
  let inputIds :=
    if inputIds.device == device then inputIds else inputIds.to device
  let refStyle :=
    if refStyle.device == device then refStyle else refStyle.to device
  (inputIds, refStyle)

def init (cfg : KittenTTSConfig) (device : Device := .CPU) : IO (Model (device := device) cfg) := do
  let bert ← KittenAlbert.init (device := device) cfg.nToken cfg.plbert
  let bertEncoder ← LinearNorm.init (device := device) cfg.plbert.hiddenSize cfg.hiddenDim
  let predictor ← ProsodyPredictor.init (device := device) cfg
  let textEncoder ← TextEncoder.init (device := device) cfg
  let decoder ← Decoder.init (device := device) cfg
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

def predictDurations {seq : Nat}
    (m : Model (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device })
    (speed : Float := 1.0)
    : IO KittenTTSDurationPrediction := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  if hSeq : seq = 0 then
    pure {
      durVals := #[]
      predDurations := #[]
      alignmentFrames := 0
    }
  else
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := m.bertEncoder.forward3d bertOut
    let dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := seqToCF dEnSeq

    let predictorStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device } := Tensor.slice refStyle 1 cfg.styleDim cfg.styleDim (_bounds := by simp [KittenTTSConfig.fullStyleDim]; omega)

    let durEnc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device } :=
      m.predictor.durationEncoding dEn predictorStyle
    let durLogits : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32, device := device } :=
      m.predictor.durationLogits durEnc
    let durSig : Tensor { shape := #[1, seq, cfg.maxDur], dtype := .Float32, device := device } := Tensor.sigmoid durLogits
    let durSum : Tensor { shape := #[1, seq], dtype := .Float32, device := device } := Tensor.reshape (Tensor.sumDim durSig 2 false) #[1, seq]
    let durVals ← Tensor.toFloatArray durSum
    let predDurations := durationToFrames durVals speed
    let frames := predDurations.foldl (fun acc x => acc + x) 0
    pure {
      durVals
      predDurations
      alignmentFrames := frames
    }

private def prepareSynthesis {seq : Nat}
    (m : Model (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device })
    (speed : Float := 1.0)
    : IO (Option (PreparedSynthesis (device := device) cfg)) := do
  let (inputIds, refStyle) := prepareInputs m inputIds refStyle
  let durPred ← predictDurations m inputIds refStyle speed
  let frames := durPred.alignmentFrames
  if hEmpty : seq = 0 ∨ frames = 0 then
    pure none
  else
    have hSeq : 0 < seq := by omega
    have hFrames : 0 < frames := by omega
    let decoderStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device } := Tensor.slice refStyle 1 0 cfg.styleDim (_bounds := by simp [KittenTTSConfig.fullStyleDim]; omega)
    let predictorStyle : Tensor { shape := #[1, cfg.styleDim], dtype := .Float32, device := device } := Tensor.slice refStyle 1 cfg.styleDim cfg.styleDim (_bounds := by simp [KittenTTSConfig.fullStyleDim]; omega)
    let (bertOut, _pooled) := m.bert.forward inputIds
    let dEnSeq : Tensor { shape := #[1, seq, cfg.hiddenDim], dtype := .Float32, device := device } := m.bertEncoder.forward3d bertOut
    let dEn : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := seqToCF dEnSeq
    let durEnc : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, seq], dtype := .Float32, device := device } :=
      m.predictor.durationEncoding dEn predictorStyle
    let align : Tensor { shape := #[1, seq, frames], dtype := .Float32, device := device } := buildAlignment durPred.predDurations inputIds.device
    let prosodyFrames : Tensor { shape := #[1, cfg.hiddenDim + cfg.styleDim, frames], dtype := .Float32, device := device } := bmm durEnc align
    let (f0Curve, nCurve) := m.predictor.forwardF0N prosodyFrames predictorStyle hFrames
    let textEnc : Tensor { shape := #[1, cfg.hiddenDim, seq], dtype := .Float32, device := device } := m.textEncoder.forward inputIds hSeq
    let asr : Tensor { shape := #[1, cfg.hiddenDim, frames], dtype := .Float32, device := device } := bmm textEnc align
    pure <| some {
      alignmentFrames := frames
      framesPos := hFrames
      decoderStyle, asr, f0Curve, nCurve
      predDurations := durPred.predDurations
    }

def synthesizeIds {seq : Nat}
    (m : Model (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device })
    (speed : Float := 1.0)
    : IO (KittenTTSOutput (device := device)) := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    pure {
      audio := Tensor.packShape (Tensor.zeros #[1, 1, 0] (device := modelDevice m))
      predDurations := #[]
      alignmentFrames := 0
    }
  | some prepared =>
    let audio ← m.decoder.forward prepared.asr prepared.f0Curve prepared.nCurve prepared.decoderStyle prepared.framesPos
    pure {
      audio
      predDurations := prepared.predDurations
      alignmentFrames := prepared.alignmentFrames
    }

def debugSynthesizeIds {seq : Nat}
    (m : Model (device := device) cfg)
    (inputIds : Tensor { shape := #[1, seq], dtype := .Int64, device := device })
    (refStyle : Tensor { shape := #[1, KittenTTSConfig.fullStyleDim cfg], dtype := .Float32, device := device })
    (speed : Float := 1.0)
    : IO (KittenTTSDebugOutput (device := device) cfg) := do
  match (← prepareSynthesis m inputIds refStyle speed) with
  | none =>
    let device := modelDevice m
    pure {
      alignmentFrames := 0
      asr := Tensor.zeros #[1, cfg.hiddenDim, 0] (device := device)
      f0Curve := Tensor.zeros #[1, 1, 2 * 0] (device := device)
      nCurve := Tensor.zeros #[1, 1, 2 * 0] (device := device)
      audio := Tensor.packShape (Tensor.zeros #[1, 1, 0] (device := device))
      predDurations := #[]
    }
  | some prepared =>
    let audio ← m.decoder.forward prepared.asr prepared.f0Curve prepared.nCurve prepared.decoderStyle prepared.framesPos
    pure {
      alignmentFrames := prepared.alignmentFrames
      asr := prepared.asr
      f0Curve := prepared.f0Curve
      nCurve := prepared.nCurve
      audio := audio
      predDurations := prepared.predDurations
    }

end Model

end torch.kittentts
