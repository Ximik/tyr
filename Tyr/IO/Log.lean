/-!
# Tyr.IO.Log

Silent-by-default logging callbacks for library code.

Library modules can depend on these handlers without embedding direct console I/O.
Executables and examples remain responsible for wiring handlers to console sinks,
files, or structured artifact sinks.
-/

namespace torch.Log

/-- A string log sink. -/
abbrev Sink := String → IO Unit

/-- Log callbacks for informational, warning, and error events. -/
structure Handlers where
  onInfo : Sink := fun _ => pure ()
  onWarn : Sink := fun _ => pure ()
  onError : Sink := fun _ => pure ()

/-- Combine two handler sets by invoking both in order. -/
def Handlers.combine (lhs rhs : Handlers) : Handlers := {
  onInfo := fun msg => do
    lhs.onInfo msg
    rhs.onInfo msg
  onWarn := fun msg => do
    lhs.onWarn msg
    rhs.onWarn msg
  onError := fun msg => do
    lhs.onError msg
    rhs.onError msg
}

end torch.Log
