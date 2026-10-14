/-
  Tyr/Data/Task.lean

  Task-based data loading for instruction tuning and chat models.

  Based on nanochat's task system:
  - Tasks provide conversation data for training
  - Conversations are rendered with special tokens
  - Loss masking trains only on assistant responses

  Key concepts:
  - Message: A single turn in a conversation (user/assistant)
  - Conversation: A sequence of messages
  - Task: A dataset that yields conversations
  - TaskMixture: Combines multiple tasks with deterministic interleaving
-/
import Tyr.Torch
import Tyr.Typed

namespace torch.Data.Task

open torch

/-! ## Message Types -/

/-- Role in a conversation -/
inductive Role where
  | system
  | user
  | assistant
  | tool
  deriving Repr, BEq, Inhabited

/-- Content part types for structured messages -/
inductive ContentType where
  | text
  | code (language : String)
  | toolCall (name : String)
  | toolResult
  deriving Repr, BEq

/-- A content part within a message -/
structure ContentPart where
  type : ContentType := .text
  content : String
  deriving Repr

/-- A message in a conversation -/
structure Message where
  role : Role
  content : String
  /-- Structured content parts (optional, for tool use) -/
  parts : Array ContentPart := #[]
  deriving Repr

/-- Create a simple text message -/
def Message.text (role : Role) (content : String) : Message :=
  { role, content }

def Message.user (content : String) : Message := Message.text .user content
def Message.assistant (content : String) : Message := Message.text .assistant content
def Message.system (content : String) : Message := Message.text .system content

/-! ## Conversation -/

/-- A conversation is a sequence of messages -/
structure Conversation where
  messages : Array Message
  /-- Optional metadata (task source, ID, etc.) -/
  metadata : List (String × String) := []
  deriving Repr

def Conversation.empty : Conversation := { messages := #[] }

def Conversation.addMessage (conv : Conversation) (msg : Message) : Conversation :=
  { conv with messages := conv.messages.push msg }

def Conversation.fromMessages (msgs : Array Message) : Conversation :=
  { messages := msgs }

/-! ## Tokenized Conversation -/

/-- A tokenized conversation with training mask -/
structure TokenizedConversation where
  /-- Input token IDs -/
  tokens : Array UInt64
  /-- Mask: 1 = train on this token, 0 = don't train (input/user tokens) -/
  mask : Array UInt8
  /-- Original conversation for reference -/
  source : Conversation
  deriving Repr

/-! ## Task Interface -/

/-- Configuration for a task dataset -/
structure TaskConfig where
  /-- Maximum number of rows to use (0 = unlimited) -/
  maxRows : Nat := 0
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  /-- Whether this is a validation set -/
  isValidation : Bool := false
  deriving Repr, Inhabited

/-- A loaded task with its data -/
structure LoadedTask where
  name : String
  conversations : Array Conversation
  config : TaskConfig
  deriving Repr, Inhabited

def LoadedTask.size (t : LoadedTask) : Nat := t.conversations.size
def LoadedTask.get (t : LoadedTask) (idx : Nat) : Conversation :=
  t.conversations[idx]?.getD Conversation.empty

/-! ## Task Mixture -/

/-- Entry in a task mixture -/
structure MixtureEntry where
  task : LoadedTask
  /-- Number of times to repeat this task in the mixture -/
  weight : Nat := 1
  deriving Repr, Inhabited

/-- A mixture of multiple tasks with deterministic interleaving -/
structure TaskMixture where
  /-- Tasks in the mixture -/
  entries : Array MixtureEntry
  /-- Shuffled index mapping: mixture index → (task index, example index) -/
  indexMap : Array (Nat × Nat)
  /-- Random seed used for shuffling -/
  seed : UInt64
  deriving Repr

/-- LCG random number generator for deterministic shuffling -/
private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

/-- Fisher-Yates shuffle with seed for (Nat × Nat) arrays -/
private def shuffleIndices (arr : Array (Nat × Nat)) (seed : UInt64) : Array (Nat × Nat) := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

/-- Create a task mixture from weighted entries -/
def TaskMixture.create (entries : Array MixtureEntry) (seed : UInt64 := 42) : TaskMixture :=
  -- Build index map: for each task, add (taskIdx, exampleIdx) pairs weighted times
  let indexMap := Id.run do
    let mut indices : Array (Nat × Nat) := #[]
    for taskIdx in [:entries.size] do
      let entry := entries[taskIdx]!
      for _ in [:entry.weight] do
        for exampleIdx in [:entry.task.conversations.size] do
          indices := indices.push (taskIdx, exampleIdx)
    return indices
  -- Shuffle deterministically
  let shuffledMap := shuffleIndices indexMap seed
  { entries, indexMap := shuffledMap, seed }

/-- Get total size of the mixture -/
def TaskMixture.size (mix : TaskMixture) : Nat := mix.indexMap.size

/-- Get a conversation from the mixture by index -/
def TaskMixture.get (mix : TaskMixture) (idx : Nat) : IO Conversation := do
  if idx >= mix.indexMap.size then
    return Conversation.empty
  let (taskIdx, exampleIdx) := mix.indexMap[idx]!
  let task := mix.entries[taskIdx]!.task
  return task.conversations[exampleIdx]?.getD Conversation.empty

/-- Shared conversation mixture surface used by SFT and streaming token pipelines. -/
structure ConversationMixture where
  size : Nat
  seed : UInt64
  getAtSeed : UInt64 → Nat → IO Conversation

/-- Get a conversation from the mixture using its current seed. -/
def ConversationMixture.get (mix : ConversationMixture) (idx : Nat) : IO Conversation :=
  mix.getAtSeed mix.seed idx

/-- Reuse the same logical mixture with a different shuffle seed. -/
def ConversationMixture.reshuffle (mix : ConversationMixture) (seed : UInt64) : ConversationMixture :=
  { mix with seed := seed }

/-- Project a `TaskMixture` onto the shared conversation mixture interface. -/
def TaskMixture.toConversationMixture (mix : TaskMixture) : ConversationMixture :=
  {
    size := mix.size
    seed := mix.seed
    getAtSeed := fun seed idx => (TaskMixture.create mix.entries seed).get idx
  }


/-! ## Conversation Rendering -/

/-- Special tokens for conversation rendering -/
structure ChatTokens where
  bos : UInt64
  userStart : UInt64
  userEnd : UInt64
  assistantStart : UInt64
  assistantEnd : UInt64
  pythonStart : UInt64
  pythonEnd : UInt64
  systemStart : UInt64
  systemEnd : UInt64
  toolStart : UInt64
  toolEnd : UInt64
  outputStart : UInt64
  outputEnd : UInt64
  deriving Repr, Inhabited

/-- Merge a leading system message into the first user message (nanochat behavior). -/
private def mergeLeadingSystemMessage (msgs : Array Message) : Array Message :=
  if msgs.size >= 2 then
    match msgs[0]?, msgs[1]? with
    | some first, some second =>
      if first.role == .system && second.role == .user then
        let mergedUser : Message := { second with content := first.content ++ "\n\n" ++ second.content }
        #[mergedUser] ++ msgs.extract 2 msgs.size
      else
        msgs
    | _, _ => msgs
  else
    msgs

/-- Render a conversation to tokens with training mask.
    Only assistant responses are marked for training (mask = 1). -/
def renderConversation
    (conv : Conversation)
    (tokens : ChatTokens)
    (encode : String → Array UInt64)
    : TokenizedConversation := Id.run do
  let messages := mergeLeadingSystemMessage conv.messages
  let mut allTokens : Array UInt64 := #[tokens.bos]
  let mut mask : Array UInt8 := #[0]  -- Don't train on BOS

  for msg in messages do
    match msg.role with
    | .system =>
      -- Keep system content unsupervised; treat it as extra user context.
      allTokens := allTokens.push tokens.userStart
      mask := mask.push 0
      let contentTokens := encode msg.content
      allTokens := allTokens ++ contentTokens
      mask := mask ++ Array.mk (List.replicate contentTokens.size 0)
      allTokens := allTokens.push tokens.userEnd
      mask := mask.push 0

    | .user =>
      -- User messages: don't train
      allTokens := allTokens.push tokens.userStart
      mask := mask.push 0
      let contentTokens := encode msg.content
      allTokens := allTokens ++ contentTokens
      mask := mask ++ Array.mk (List.replicate contentTokens.size 0)
      allTokens := allTokens.push tokens.userEnd
      mask := mask.push 0

    | .assistant =>
      -- Assistant messages: TRAIN on content and end token
      allTokens := allTokens.push tokens.assistantStart
      mask := mask.push 0  -- Don't train on start token
      if msg.parts.isEmpty then
        let contentTokens := encode msg.content
        allTokens := allTokens ++ contentTokens
        mask := mask ++ Array.mk (List.replicate contentTokens.size 1)
      else
        for part in msg.parts do
          match part.type with
          | .text =>
            let valueTokens := encode part.content
            allTokens := allTokens ++ valueTokens
            mask := mask ++ Array.mk (List.replicate valueTokens.size 1)
          | .code lang =>
            if lang == "python" then
              allTokens := allTokens.push tokens.pythonStart
              mask := mask.push 1
              let valueTokens := encode part.content
              allTokens := allTokens ++ valueTokens
              mask := mask ++ Array.mk (List.replicate valueTokens.size 1)
              allTokens := allTokens.push tokens.pythonEnd
              mask := mask.push 1
            else
              let valueTokens := encode part.content
              allTokens := allTokens ++ valueTokens
              mask := mask ++ Array.mk (List.replicate valueTokens.size 1)
          | .toolCall _ =>
            allTokens := allTokens.push tokens.pythonStart
            mask := mask.push 1
            let valueTokens := encode part.content
            allTokens := allTokens ++ valueTokens
            mask := mask ++ Array.mk (List.replicate valueTokens.size 1)
            allTokens := allTokens.push tokens.pythonEnd
            mask := mask.push 1
          | .toolResult =>
            allTokens := allTokens.push tokens.outputStart
            mask := mask.push 0
            let valueTokens := encode part.content
            allTokens := allTokens ++ valueTokens
            mask := mask ++ Array.mk (List.replicate valueTokens.size 0)
            allTokens := allTokens.push tokens.outputEnd
            mask := mask.push 0
      allTokens := allTokens.push tokens.assistantEnd
      mask := mask.push 1  -- Train on end token too

    | .tool =>
      -- Tool outputs: don't train (model didn't generate these)
      allTokens := allTokens.push tokens.outputStart
      mask := mask.push 0
      let contentTokens := encode msg.content
      allTokens := allTokens ++ contentTokens
      mask := mask ++ Array.mk (List.replicate contentTokens.size 0)
      allTokens := allTokens.push tokens.outputEnd
      mask := mask.push 0

  { tokens := allTokens, mask, source := conv }

/-! ## Batch Collation -/

/-- Build an Int64 token tensor from host data over the raw FFI boundary.
    The runtime shape is the array length; `reshapeChecked` validates the
    result metadata (shape and Int64 dtype) against the static spec. -/
private def int64TensorFromArray (arr : Array Int64) : Tensor { shape := #[arr.size.toUInt64], dtype := .Int64 } :=
  Tensor.fromInt64Array arr

/-- A batch of tokenized conversations -/
structure ConversationBatch where
  /-- Batch size (number of conversations) -/
  batchSize : UInt64
  /-- Sequence length after padding -/
  seqLen : UInt64
  /-- Token IDs: [batchSize, seqLen] -/
  tokens : Tensor { shape := #[batchSize, seqLen], dtype := .Int64 }
  /-- Training mask: [batchSize, seqLen], 1 = train, 0 = ignore -/
  mask : Tensor { shape := #[batchSize, seqLen], dtype := .Float32 }
  /-- Actual lengths before padding -/
  lengths : Array Nat
  deriving Repr

/-- Pad a sequence to target length -/
private def padSequence (seq : Array UInt64) (targetLen : Nat) (padValue : UInt64) : Array UInt64 :=
  if seq.size >= targetLen then
    seq.extract 0 targetLen
  else
    seq ++ Array.mk (List.replicate (targetLen - seq.size) padValue)

private def padMask (seq : Array UInt8) (targetLen : Nat) : Array UInt8 :=
  if seq.size >= targetLen then
    seq.extract 0 targetLen
  else
    seq ++ Array.mk (List.replicate (targetLen - seq.size) 0)

/-- Collate tokenized conversations into a batch -/
def collate (convs : Array TokenizedConversation) (maxLen : Nat) (padToken : UInt64 := 0)
    : IO ConversationBatch := do
  let batchSize := convs.size
  if batchSize == 0 then
    return { batchSize := 0, seqLen := 0
             tokens := Tensor.fullInt #[0, 0] 0
             mask := Tensor.zeros #[0, 0]
             lengths := #[] }

  -- Find max length in batch (capped at maxLen)
  let actualMaxLen := convs.foldl (fun acc c => max acc (min c.tokens.size maxLen)) 0

  -- Pad all sequences
  let mut allTokens : Array UInt64 := #[]
  let mut allMasks : Array UInt8 := #[]
  let mut lengths : Array Nat := #[]

  for conv in convs do
    let paddedTokens := padSequence conv.tokens actualMaxLen padToken
    let paddedMask := padMask conv.mask actualMaxLen
    allTokens := allTokens ++ paddedTokens
    allMasks := allMasks ++ paddedMask
    lengths := lengths.push (min conv.tokens.size actualMaxLen)

  -- Convert to tensors
  let tokensFlat := int64TensorFromArray (allTokens.map (·.toInt64))
  let tokensTensor := tokensFlat.reshapeChecked #[batchSize.toUInt64, actualMaxLen.toUInt64]

  let maskFlat := Tensor.fromFloatArray (allMasks.map fun m => m.toNat.toFloat)
  let maskTensor := maskFlat.reshapeChecked #[batchSize.toUInt64, actualMaxLen.toUInt64]

  return {
    batchSize := batchSize.toUInt64
    seqLen := actualMaxLen.toUInt64
    tokens := tokensTensor
    mask := maskTensor
    lengths
  }

/-! ## Task Data Loader -/

/-- Iterator for task-based training -/
structure TaskIterator where
  mixture : ConversationMixture
  currentIdx : Nat
  batchSize : Nat
  maxSeqLen : Nat
  chatTokens : ChatTokens
  encode : String → Array UInt64
  epoch : Nat

def TaskIterator.new
    (mixture : ConversationMixture)
    (batchSize maxSeqLen : Nat)
    (chatTokens : ChatTokens)
    (encode : String → Array UInt64)
    : TaskIterator :=
  { mixture, currentIdx := 0, batchSize, maxSeqLen, chatTokens, encode, epoch := 0 }

/-- Get next batch from task iterator -/
def TaskIterator.nextBatch (iter : TaskIterator)
    : IO (Option ConversationBatch × TaskIterator) := do
  -- Check if we've exhausted the mixture
  if iter.currentIdx >= iter.mixture.size then
    -- Start new epoch with re-shuffled mixture
    let newMixture := iter.mixture.reshuffle (iter.mixture.seed + iter.epoch.toUInt64 + 1)
    let newIter := { iter with
      mixture := newMixture
      currentIdx := 0
      epoch := iter.epoch + 1
    }
    return (none, newIter)

  -- Collect batch of conversations
  let endIdx := min (iter.currentIdx + iter.batchSize) iter.mixture.size
  let mut tokenizedConvs : Array TokenizedConversation := #[]

  for idx in [iter.currentIdx:endIdx] do
    let conv ← iter.mixture.get idx
    let tokenized := renderConversation conv iter.chatTokens iter.encode
    tokenizedConvs := tokenizedConvs.push tokenized

  -- Collate into batch
  let batch ← collate tokenizedConvs iter.maxSeqLen iter.chatTokens.assistantEnd
  let newIter := { iter with currentIdx := endIdx }

  return (some batch, newIter)

/-- Reset iterator to beginning -/
def TaskIterator.reset (iter : TaskIterator) : TaskIterator :=
  { iter with currentIdx := 0 }

/-- Get progress through current epoch -/
def TaskIterator.progress (iter : TaskIterator) : Float :=
  if iter.mixture.size == 0 then 1.0
  else iter.currentIdx.toFloat / iter.mixture.size.toFloat

/-- Streaming token buffer for autoregressive GPT-style training over task mixtures.

    This mirrors nanochat midtraining data flow:
    - iterate task rows in rank-strided order
    - append rendered conversation tokens into a token buffer
    - emit contiguous windows of length `batchSize * seqLen + 1`
    - form `(inputs, targets)` by one-token shift
-/
structure TaskTokenStream where
  mixture : ConversationMixture
  chatTokens : ChatTokens
  encode : String → Array UInt64
  batchSize : Nat
  seqLen : Nat
  rank : Nat := 0
  worldSize : Nat := 1
  cursor : Nat := 0
  buffer : Array UInt64 := #[]
  consumed : Nat := 0
  steps : Nat := 0
  lastWrapped : Bool := false
  lastProgress : Float := 0.0

/-- Construct a task token stream. -/
def TaskTokenStream.new
    (mixture : ConversationMixture)
    (batchSize seqLen : Nat)
    (chatTokens : ChatTokens)
    (encode : String → Array UInt64)
    (rank worldSize : Nat := 0)
    : TaskTokenStream :=
  let size := mixture.size
  let startCursor :=
    if size == 0 then 0 else rank % size
  {
    mixture := mixture
    chatTokens := chatTokens
    encode := encode
    batchSize := batchSize
    seqLen := seqLen
    rank := rank
    worldSize := max 1 worldSize
    cursor := startCursor
  }

private def TaskTokenStream.available (s : TaskTokenStream) : Nat :=
  s.buffer.size - s.consumed

private def TaskTokenStream.compact (s : TaskTokenStream) : TaskTokenStream :=
  if s.consumed == 0 then
    s
  else if s.consumed * 2 < s.buffer.size then
    s
  else
    { s with
      buffer := s.buffer.extract s.consumed s.buffer.size
      consumed := 0
    }

private def TaskTokenStream.pushTokens (s : TaskTokenStream) (tokens : Array UInt64) : TaskTokenStream :=
  { s with buffer := s.buffer ++ tokens }

private def TaskTokenStream.popWindow? (s : TaskTokenStream) (n : Nat)
    : Option (Array UInt64 × TaskTokenStream) := do
  if s.available < n then
    none
  else
    let start := s.consumed
    let stop := start + n
    let window := s.buffer.extract start stop
    let s' := { s with consumed := stop }
    some (window, s'.compact)

/-- Consume one GPT batch from the stream.
    Returns input/target tensors shaped `[batchSize, seqLen]`. -/
def TaskTokenStream.nextGPTBatch (stream : TaskTokenStream)
    : IO (Option (Tensor { shape := #[stream.batchSize.toUInt64, stream.seqLen.toUInt64], dtype := .Int64 } ×
                  Tensor { shape := #[stream.batchSize.toUInt64, stream.seqLen.toUInt64], dtype := .Int64 }) ×
          TaskTokenStream) := do
  if stream.batchSize == 0 || stream.seqLen == 0 || stream.mixture.size == 0 then
    return (none, stream)

  let needed := stream.batchSize * stream.seqLen + 1
  let mut s := { stream with lastWrapped := false }

  while s.available < needed do
    let conv ← s.mixture.get s.cursor
    let rendered := renderConversation conv s.chatTokens s.encode
    s := s.pushTokens rendered.tokens

    let nextCursor := s.cursor + s.worldSize
    if nextCursor >= s.mixture.size then
      s := { s with
        cursor := nextCursor % s.mixture.size
        lastWrapped := true
      }
    else
      s := { s with cursor := nextCursor }

  match s.popWindow? needed with
  | none =>
    return (none, s)
  | some (window, sWindow) =>
    let inputIds := window.extract 0 (needed - 1)
    let targetIds := window.extract 1 needed

    let inputFlat := int64TensorFromArray (inputIds.map (·.toInt64))
    let inputTensor := inputFlat.reshapeChecked
      #[stream.batchSize.toUInt64, stream.seqLen.toUInt64]
    let targetFlat := int64TensorFromArray (targetIds.map (·.toInt64))
    let targetTensor := targetFlat.reshapeChecked
      #[stream.batchSize.toUInt64, stream.seqLen.toUInt64]

    let progress :=
      if sWindow.mixture.size == 0 then
        1.0
      else
        sWindow.cursor.toFloat / sWindow.mixture.size.toFloat

    let sOut := { sWindow with
      steps := sWindow.steps + 1
      lastProgress := progress
    }
    return (some (inputTensor, targetTensor), sOut)

/-- Stream progress over current mixed dataset epoch. -/
def TaskTokenStream.progress (s : TaskTokenStream) : Float :=
  s.lastProgress

end torch.Data.Task
