namespace torch

/-- `path` with a leading `~` or `~/` replaced by `$HOME`. Unchanged when it has
neither or `HOME` is unset; `~user` is not expanded. -/
def expandHome (path : String) : IO String := do
  let some home ← IO.getEnv "HOME" | return path
  if path == "~" then return home
  else if path.startsWith "~/" then return s!"{home}{path.drop 1}"
  else return path

end torch
