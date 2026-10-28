/-
  Isolated test: load a WAV, then transcribe via the in-memory
  transcribeWaveform16k path (no temp file round-trip).
-/
import Tyr.Model.Whisper

open torch.whisper
open torch.qwen3asr

def main (argv : List String) : IO UInt32 := do
  let modelDir := argv.getD 0 "weights/whisper-base.en"
  let wavPath := argv.getD 1 "output/mlk_10s.wav"

  IO.println s!"Loading model from {modelDir}..."
  let bundle ← loadFromPretrainedDir modelDir

  IO.println s!"Loading WAV from {wavPath}..."
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  IO.println s!"  {wav16k.size} samples ({wav16k.size.toFloat / 16000.0}s)"

  IO.println "Transcribing via transcribeWaveform16k..."
  let out ← transcribeWaveform16k
    bundle.model bundle.tok bundle.preprocessor bundle.melBinsMatch
    wav16k "en" 128 true {}

  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  pure 0
