import Tyr.Model.Qwen3ASR.StreamModel
import Tyr.Audio.AppleInput

namespace Examples.Qwen3ASR

open torch.qwen3asr

structure Args where
  source : String := "weights/qwen3-asr-0.6b"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  language : Option String := none
  context : String := ""
  maxNewTokens : Nat := 128
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  runSec : Float := 30.0
  sileroVADPath : Option String := none
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO Nat :=
  match v.toNat? with
  | some n => pure n
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

private def parseFloatArg (name : String) (v : String) : IO Float :=
  match parseFloatLit? v with
  | some x => pure x
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--model-dir" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest => parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest => parseArgsLoop rest { acc with cacheDir := v }
  | "--language" :: v :: rest => parseArgsLoop rest { acc with language := some v }
  | "--context" :: v :: rest => parseArgsLoop rest { acc with context := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--chunk-sec" :: v :: rest =>
      parseArgsLoop rest { acc with chunkSec := (← parseFloatArg "--chunk-sec" v) }
  | "--hop-sec" :: v :: rest =>
      parseArgsLoop rest { acc with hopSec := (← parseFloatArg "--hop-sec" v) }
  | "--run-sec" :: v :: rest =>
      parseArgsLoop rest { acc with runSec := (← parseFloatArg "--run-sec" v) }
  | "--silero-vad-path" :: v :: rest =>
      parseArgsLoop rest { acc with sileroVADPath := some v }
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRLiveMicTrueStream [options]"
      IO.println "  --source <path-or-repo>  Local model dir or HF repo id"
      IO.println "  --model-dir <path>       Alias for --source (backward compatible)"
      IO.println "  --revision <rev>         HF revision/branch/tag (default: main)"
      IO.println "  --cache-dir <path>       Local cache for downloaded files"
      IO.println "  --language <name>        Optional forced language"
      IO.println "  --context <text>         Optional system context"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "  --chunk-sec <f>          Decode window seconds"
      IO.println "  --hop-sec <f>            Step seconds"
      IO.println "  --run-sec <f>            Total runtime seconds"
      IO.println "  --silero-vad-path <p>    Optional Silero VAD safetensors path"
      throw <| IO.userError ""
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.chunkSec <= 0.0 || args.hopSec <= 0.0 || args.runSec <= 0.0 then
    throw <| IO.userError "chunk-sec, hop-sec, and run-sec must be > 0"

  let sm ← loadFromPretrained args.source args.revision args.cacheDir
  IO.println s!"Resolved model dir: {sm.modelDir}"
  let sileroPath ←
    match args.sileroVADPath with
    | some p => pure (some p)
    | none =>
      match (← IO.getEnv "SILERO_VAD_PATH") with
      | some p => pure (some p)
      | none => pure none
  let mut ss ← newSession sm args.chunkSec args.hopSec
    (context := args.context)
    (language := args.language)
    (sileroVADPath := sileroPath)

  let hopSamples := toSamples args.hopSec
  let steps := Nat.max 1 (((args.runSec / args.hopSec) + 0.5).toUInt64.toNat)

  IO.println s!"true-stream session: chunk_sec={args.chunkSec} hop_sec={args.hopSec} run_sec={args.runSec}"
  Tyr.Audio.AppleInput.start 16000 1 100

  try
    let mut lastUnstable := ""
    for _ in [:steps] do
      let pcm ← Tyr.Audio.AppleInput.read hopSamples 1500
      let (ssNext, out) ← pushAudio sm ss pcm (maxNewTokens := args.maxNewTokens)
      ss := ssNext
      if out.didDecode then
        if !out.stableAppend.isEmpty then
          IO.println s!"STABLE+= {out.stableAppend}"
        if !out.unstableText.isEmpty && out.unstableText != lastUnstable then
          IO.println s!"UNSTABLE= {out.unstableText}"
          lastUnstable := out.unstableText
    let (ssFinal, outFinal) ← flush sm ss (maxNewTokens := args.maxNewTokens)
    ss := ssFinal
    if outFinal.didDecode then
      if !outFinal.stableAppend.isEmpty then
        IO.println s!"STABLE+= {outFinal.stableAppend}"
      if !outFinal.unstableText.isEmpty && outFinal.unstableText != lastUnstable then
        IO.println s!"UNSTABLE= {outFinal.unstableText}"
    Tyr.Audio.AppleInput.stop
  catch e =>
    Tyr.Audio.AppleInput.stop
    throw e

  pure 0

end Examples.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen3ASR.runMain argv
