import Tyr.Model.Qwen3TTS.DTypes
import Tyr.Typed.Ops
import Tyr.SafeTensors.Exact

namespace torch.qwen3tts

/-- A codec's weights share `compute`; codebook statistics retain their stored
precisions. Their true-division result must agree with the projection input. -/
structure SpeechTokenizerDTypes where
  compute : RealFloatingDType
  usage : ExactDType
  sum : ExactDType
  usageOrdered : DType.Ordered usage.val
  usagePromotion : DType.Promotes usage.val usage.val usage.val
  promoted : DType
  promotion : DType.Promotes sum.val usage.val promoted
  resultEq : promoted.divideResult = compute.val

def float32SpeechTokenizerDTypes : SpeechTokenizerDTypes :=
  { compute := float32DType, usage := float32DType.toExactDType, sum := float32DType.toExactDType,
    usageOrdered := ⟨rfl⟩, usagePromotion := ⟨rfl⟩,
    promoted := .Float32, promotion := ⟨rfl⟩, resultEq := rfl }

namespace SpeechTokenizerDTypes

instance (dtypes : SpeechTokenizerDTypes) : DType.Ordered dtypes.usage.val := dtypes.usageOrdered
instance (dtypes : SpeechTokenizerDTypes) : DType.Promotes dtypes.usage.val dtypes.usage.val dtypes.usage.val := dtypes.usagePromotion

/-- Validate the relationship without converting any checkpoint tensor. -/
def ofDTypes? (compute usage sum : ExactDType) : Except String SpeechTokenizerDTypes := do
  let floating ← DType.realFloating? compute.val
  let _ := floating.down
  let compute := RealFloatingDType.ofDType compute.val
  let usageOrdered : PLift (DType.Ordered usage.val) ←
    if h : usage.val.orderedDomain = true then pure (⟨⟨h⟩⟩ : PLift (DType.Ordered usage.val))
    else throw s!"Codebook usage counts require an ordered dtype, got {usage.val}"
  let usagePromotion : PLift (DType.Promotes usage.val usage.val usage.val) ←
    if h : DType.promote? usage.val usage.val = some usage.val then pure (⟨⟨h⟩⟩ : PLift (DType.Promotes usage.val usage.val usage.val))
    else throw s!"Codebook usage counts do not support self-promotion: {usage.val}"
  match h : DType.promote? sum.val usage.val with
  | none => throw s!"Unsupported codebook promotion: {sum.val} / {usage.val}"
  | some promoted =>
    if resultEq : promoted.divideResult = compute.val then
      pure { compute, usage, sum, usageOrdered := usageOrdered.down, usagePromotion := usagePromotion.down, promoted, promotion := ⟨h⟩, resultEq }
    else
      throw s!"Codebook division produces {promoted.divideResult}, but codec weights require {compute.val}"

/-- Read the precision anchors; every other field is subsequently checked
against these roles by the model loader. No checkpoint tensor is converted. -/
def load (path computeName : String) (computeShape : Shape)
    (usageName sumName : String) (device : Device) : IO SpeechTokenizerDTypes := do
  let compute ← safetensors.loadTensorExact path computeName computeShape device
  let usage ← safetensors.loadTensorExact path usageName #[2048] device
  let sum ← safetensors.loadTensorExact path sumName #[2048, 256] device
  match ofDTypes? compute.1 usage.1 sum.1 with
  | .ok dtypes => pure dtypes
  | .error message => throw <| IO.userError s!"Speech tokenizer codebook '{sumName}': {message}"

/-- Division keeps the backend's promoted result; the checked layout supplies
the equality needed by subsequent matrix multiplication and convolution. -/
def divideCodebook (dtypes : SpeechTokenizerDTypes) {shape : Shape} {device : Device}
    (sum : Tensor ⟨shape, dtypes.sum.val, device⟩)
    (usage : Tensor ⟨shape, dtypes.usage.val, device⟩) :
    Tensor ⟨shape, dtypes.compute.val, device⟩ :=
  letI := dtypes.promotion
  dtypes.resultEq ▸ Tensor.div sum usage

end SpeechTokenizerDTypes
end torch.qwen3tts
