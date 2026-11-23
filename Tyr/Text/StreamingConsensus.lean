namespace Tyr.Text

structure ConsensusConfig where
  historyWindows : Nat := 6
  confirmWindows : Nat := 3
  rollbackTokens : Nat := 2
  freezeAfterSteps : Nat := 4
  mutableTailTokensWhileSpeech : Nat := 6
  deriving Repr

structure ConsensusState where
  cfg : ConsensusConfig := {}
  stableIds : Array UInt32 := #[]
  unstableIds : Array UInt32 := #[]
  history : Array (Array UInt32) := #[]
  lastIds : Array UInt32 := #[]
  unchangedSteps : Nat := 0
  deriving Repr

structure TextDelta where
  stableAppend : String := ""
  unstableText : String := ""
  fullText : String := ""
  deriving Repr

private def safeSub (a b : Nat) : Nat := if a >= b then a - b else 0

private def commonPrefixLenIds (a b : Array UInt32) : Nat :=
  let n := Nat.min a.size b.size
  Id.run do
    let mut i := 0
    while i < n && a[i]! == b[i]! do
      i := i + 1
    i

private def trimHistory (hist : Array (Array UInt32)) (maxKeep : Nat) : Array (Array UInt32) :=
  if hist.size <= maxKeep then hist else hist.extract (hist.size - maxKeep) hist.size

private def consensusPrefixLastK (hist : Array (Array UInt32)) (k : Nat) : Nat :=
  if k == 0 || hist.size < k then
    0
  else
    let start := hist.size - k
    let window := hist.extract start hist.size
    let first := window[0]!
    let minLen := Id.run do
      let mut m := first.size
      for i in [1:window.size] do
        let n := (window[i]!).size
        if n < m then m := n
      m
    Id.run do
      let mut p := 0
      let mut done := false
      while p < minLen && !done do
        let t := first[p]!
        let mut allEq := true
        for i in [1:window.size] do
          if allEq && (window[i]!)[p]! != t then
            allEq := false
        if allEq then
          p := p + 1
        else
          done := true
      p

/-- Update consensus state from a fresh token hypothesis.
    Combines:
    - prefix agreement with previous hypothesis
    - k-window token consensus across recent history
    Then emits append-only stable text plus mutable unstable text. -/
def updateWithSignals
    (st : ConsensusState)
    (ids : Array UInt32)
    (speechActive : Bool := true)
    (boundary : Bool := false)
    (decode : Array UInt32 → String)
    : ConsensusState × TextDelta :=
  let hist0 := trimHistory (st.history.push ids) st.cfg.historyWindows
  let lcpPrev := commonPrefixLenIds st.lastIds ids
  let cPrev := safeSub lcpPrev st.cfg.rollbackTokens
  let cHistRaw := consensusPrefixLastK hist0 st.cfg.confirmWindows
  let cHist := safeSub cHistRaw st.cfg.rollbackTokens
  let candidate0 :=
    if hist0.size >= st.cfg.confirmWindows then cHist else cPrev
  let unchanged := if ids == st.lastIds then st.unchangedSteps + 1 else 0
  let freezeCommit :=
    if st.cfg.freezeAfterSteps > 0 && unchanged >= st.cfg.freezeAfterSteps then
      safeSub ids.size st.cfg.rollbackTokens
    else
      0
  let boundaryCommit :=
    if boundary || !speechActive then
      safeSub ids.size st.cfg.rollbackTokens
    else
      0
  let speechCap :=
    if speechActive then
      safeSub ids.size st.cfg.mutableTailTokensWhileSpeech
    else
      ids.size
  let candidate := Nat.min speechCap (Nat.max candidate0 (Nat.max freezeCommit boundaryCommit))
  let oldStableLen := st.stableIds.size
  let anchoredStable := commonPrefixLenIds st.stableIds ids
  let canAdvanceStable := anchoredStable == oldStableLen
  let commitLenRaw := Nat.max oldStableLen candidate
  let commitLen := if canAdvanceStable then commitLenRaw else oldStableLen
  let stableIds :=
    if canAdvanceStable then
      ids.extract 0 (Nat.min commitLen ids.size)
    else
      st.stableIds
  let unstableIds :=
    if canAdvanceStable then
      if commitLen >= ids.size then #[] else ids.extract commitLen ids.size
    else
      st.unstableIds

  let newStableText := decode stableIds
  let stableAppend :=
    if st.stableIds.size < stableIds.size then
      decode (stableIds.extract st.stableIds.size stableIds.size)
    else
      ""
  let unstableText := decode unstableIds
  let fullText := newStableText ++ unstableText

  let st' : ConsensusState := {
    st with
    stableIds := stableIds
    unstableIds := unstableIds
    history := hist0
    lastIds := ids
    unchangedSteps := unchanged
  }
  (st', { stableAppend, unstableText, fullText })

end Tyr.Text
