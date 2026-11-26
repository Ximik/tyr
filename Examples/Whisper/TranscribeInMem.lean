/-
  Isolated test: load a WAV, then transcribe via the in-memory
  transcribeWaveform16k path (no temp file round-trip).
-/
import Tyr.Model.Whisper

open torch.whisper

def main (argv : List String) : IO UInt32 := do
  let deviceName :: rest := argv
    | IO.eprintln "usage: WhisperTranscribeInMem <device> [model_dir] [wav_path]"; return 2
  let modelDir := rest.getD 0 "weights/whisper-base.en"
  let wavPath := rest.getD 1 "output/mlk_10s.wav"
  let device ← torch.Device.acquireNamed deviceName

  IO.println s!"Loading model from {modelDir}..."
  let bundle ← loadFromPretrainedDir modelDir device

  IO.println s!"Loading WAV from {wavPath}..."
  let wav16k ← torch.audio.normalizeAudioTo16kFromWav wavPath
  IO.println s!"  {wav16k.size} samples ({wav16k.size.toFloat / 16000.0}s)"

  IO.println "Transcribing via transcribeWaveform16k..."
  let out ← transcribeWaveform16k
    bundle.model bundle.tok bundle.preprocessor bundle.melBinsMatch
    wav16k "en" true {}

  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  pure 0
