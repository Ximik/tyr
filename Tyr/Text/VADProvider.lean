import Tyr.Model.SileroVAD

namespace Tyr.Text

structure VADSignal where
  speechActive : Bool := true
  boundary : Bool := false
  deriving Repr, Inhabited

structure SileroProvider where
  device : torch.Device
  iterator : torch.silerovad.VADIterator (device := device)
  residue : Array Float := #[]
  speechActive : Bool := false

private def chunkSize : Nat := 512

private def toChunks (audio : Array Float) : Array (Array Float) :=
  if audio.size < chunkSize then
    #[]
  else
    Id.run do
      let mut out : Array (Array Float) := #[]
      let mut i := 0
      while i + chunkSize <= audio.size do
        out := out.push (audio.extract i (i + chunkSize))
        i := i + chunkSize
      out

def initSileroProvider
    (weightsPath : String)
    (threshold : Float := 0.5)
    (minSilenceDurationMs : UInt64 := 100)
    (speechPadMs : UInt64 := 30)
    (device : torch.Device := .CPU)
    : IO SileroProvider := do
  let model ← torch.silerovad.SileroVAD.load weightsPath (device := device)
  let runtime := torch.silerovad.SileroVADRuntime.init model
  let it ← torch.silerovad.VADIterator.init runtime threshold 16000 minSilenceDurationMs speechPadMs
  pure { device, iterator := it, residue := #[], speechActive := false }

def stepSileroProvider
    (p : SileroProvider)
    (pcm16k : Array Float)
    : IO (SileroProvider × VADSignal) := do
  let all := p.residue ++ pcm16k
  let nFull := (all.size / chunkSize) * chunkSize
  let consume := all.extract 0 nFull
  let rem := all.extract nFull all.size
  let chunks := toChunks consume

  let mut it := p.iterator
  let mut active := p.speechActive
  let mut boundary := false

  for ch in chunks do
    let (b?, it') ← torch.silerovad.VADIterator.step it ch
    it := it'
    active := it.triggered
    match b? with
    | some _ => boundary := true
    | none => pure ()

  let p' : SileroProvider := {
    device := p.device
    iterator := it
    residue := rem
    speechActive := active
  }
  pure (p', { speechActive := active, boundary := boundary })

end Tyr.Text
