import Tyr.Model.Qwen3TTS.DTypes
import Tyr.Typed.Ops
import Tyr.SafeTensors.Exact

namespace torch.qwen3tts

/-- A codec's weights share `compute`; codebook statistics retain their stored
precisions. Their true-division result must agree with the projection input. -/
structure SpeechTokenizerDTypes where
  compute : ExactDType
  usage : ExactDType
  sum : ExactDType
  promoted : DType
  promotion : DType.Promotes sum.val usage.val promoted
  resultEq : promoted.divideResult = compute.val

def float32SpeechTokenizerDTypes : SpeechTokenizerDTypes :=
  { compute := float32DType, usage := float32DType, sum := float32DType,
    promoted := .Float32, promotion := ⟨rfl⟩, resultEq := rfl }

namespace SpeechTokenizerDTypes

/-- Validate the relationship without converting any checkpoint tensor. -/
def ofDTypes? (compute usage sum : ExactDType) : Except String SpeechTokenizerDTypes := do
  match h : DType.promote? sum.val usage.val with
  | none => throw s!"Unsupported codebook promotion: {sum.val} / {usage.val}"
  | some promoted =>
    if resultEq : promoted.divideResult = compute.val then
      pure { compute, usage, sum, promoted, promotion := ⟨h⟩, resultEq }
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
