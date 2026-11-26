namespace Tyr.Audio.AppleInput

@[extern "lean_tyr_audio_input_start"]
opaque start (sampleRate : Nat := 16000) (channels : Nat := 1) (bufferMs : Nat := 100) : IO Unit

@[extern "lean_tyr_audio_input_read"]
opaque read (maxSamples : @& Nat) (blockMs : Nat := 250) : IO (Array Float)

/-- Read audio samples into an unboxed `FloatArray`. -/
@[extern "lean_tyr_audio_input_read_buffer"]
opaque readBuffer (maxSamples : @& Nat) (blockMs : Nat := 250) : IO FloatArray

@[extern "lean_tyr_audio_input_stop"]
opaque stop : IO Unit

end Tyr.Audio.AppleInput
