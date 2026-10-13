import Tyr.Model.Qwen35.Pretrained

open torch.qwen35

def main : IO Unit := do
  let opts : hub.DownloadOptions := {
    revision := "main"
    cacheDir := ".model-cache/qwen35"
    includeTokenizer := true
  }
  let dir ← hub.resolvePretrainedDir "Qwen/Qwen3.5-27B" opts
  IO.println s!"resolved_dir={dir}"
