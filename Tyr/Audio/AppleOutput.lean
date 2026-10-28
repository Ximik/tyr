namespace Tyr.Audio.AppleOutput

/-- Play a sine tone at the given frequency for `durationMs` milliseconds.
    Uses AudioQueue on macOS; raises an IO error on other platforms.
    Blocks until done. -/
@[extern "lean_tyr_audio_output_beep"]
opaque beep (freqHz : Float) (durationMs : Nat := 120) (sampleRate : Nat := 44100) : IO Unit

/-- Fire-and-forget beep on a background thread. Returns immediately.
    WARNING: holds g_output_mu while playing — do not mix with sync beep
    in the same run without ensuring the async beep has finished. -/
@[extern "lean_tyr_audio_output_beep_async"]
opaque beepAsync (freqHz : Float) (durationMs : Nat := 120) (sampleRate : Nat := 44100) : IO Unit

/-- Short high-pitched beep (880 Hz, 120ms) — signals recording start. -/
def beepStart : IO Unit := beep 880.0 120

/-- Double beep (660 Hz) — signals transcription complete. -/
def beepDone : IO Unit := do
  beep 660.0 80
  beep 660.0 80

/-- Low beep (440 Hz, 100ms) — signals no speech detected. -/
def beepNoSpeech : IO Unit := beep 440.0 100

end Tyr.Audio.AppleOutput
