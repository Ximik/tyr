/-
  Tyr/Tasks/LLM.lean

  LLM-specific task implementations for instruction tuning and evaluation.

  Based on nanochat's task system:
  - ARC (AI2 Reasoning Challenge) - multiple choice
  - MMLU (Massive Multitask Language Understanding) - multiple choice
  - GSM8K (Grade School Math) - generative with tool use
  - SmolTalk (general chat conversations)
  - CustomJSON (local JSONL conversations)
  - SpellingBee - generative with character counting
  - SimpleSpelling - generative spelling practice
  - HumanEval - code generation with execution-based evaluation

  Tasks support:
  - Loading from JSONL files
  - Evaluation (categorical and generative)
  - Reward computation for RL
-/
import Tyr.Lib.Data.Task
import Tyr.Lib.Data.TaskClass
import Lean.Data.Json

namespace torch.Tasks.LLM

open torch.Data.Task
open torch.Data.TaskClass (EvalTask EvalType EvalResult BoxedTask boxTask MixtureEntry entry)

/-! ## String Helpers -/

/-- Find position of substring in string, returns index into original string -/
def findSubstr (haystack : String) (needle : String) : Option Nat := Id.run do
  let hs := haystack.toList
  let ns := needle.toList
  if ns.isEmpty then return some 0
  for i in [:hs.length] do
    let mut found := true
    for j in [:ns.length] do
      if i + j >= hs.length then
        found := false
        break
      if hs[i + j]! != ns[j]! then
        found := false
        break
    if found then return some i
  return none

private def parseRole? (role : String) : Option Role :=
  match role.trimAscii.toString.toLower with
  | "system" => some .system
  | "user" => some .user
  | "assistant" => some .assistant
  | "tool" => some .tool
  | _ => none

private def parseMessageJson? (json : Lean.Json) : Option Message := do
  let roleStr ← (json.getObjValAs? String "role").toOption
  let content ← (json.getObjValAs? String "content").toOption
  let role ← parseRole? roleStr
  return { role, content, parts := #[] }

private def validateAlternatingUserAssistant (msgs : Array Message) : Bool :=
  if msgs.isEmpty then
    false
  else
    let offset :=
      match msgs[0]? with
      | some first =>
        if first.role == .system then 1 else 0
      | none => 0
    let rest := msgs.extract offset msgs.size
    if rest.size < 2 then
      false
    else
      Id.run do
        let mut ok := true
        for i in [:rest.size] do
          let expected := if i % 2 == 0 then Role.user else Role.assistant
          match rest[i]? with
          | some msg =>
            if msg.role != expected then
              ok := false
          | none => ok := false
        ok

/-! ## Answer Extraction -/

/-- Extract numerical answer after "#### " marker (GSM8K/SpellingBee format) -/
def extractHashAnswer (response : String) : Option String := Id.run do
  -- Find "#### " marker
  let marker := "#### "
  let idx := findSubstr response marker
  match idx with
  | none => return none
  | some pos =>
    -- Extract everything after the marker until newline or end
    let start := pos + marker.length
    let rest := (response.drop start).toString
    -- Take until newline or end
    let answer := (rest.takeWhile (fun c => c != '\n' && c != '\r')).toString
    -- Normalize: remove commas, trim whitespace
    let normalized := answer.trimAscii.toString.replace "," ""
    if normalized.isEmpty then none else some normalized

/-- Extract single letter answer (A, B, C, D) for multiple choice -/
def extractLetterAnswer (response : String) (validLetters : Array String := #["A", "B", "C", "D"])
    : Option String := Id.run do
  let trimmed := response.trimAscii.toString
  -- Check if the response is just a letter
  if validLetters.contains trimmed then
    return some trimmed
  -- Otherwise look for first valid letter
  for letter in validLetters do
    if trimmed.startsWith letter then
      return some letter
  return none

/-! ## Multiple Choice Rendering -/

/-- Render a multiple choice question in nanochat format.
    Format: "- choice=letter" with letter AFTER choice for better token binding. -/
def renderMultipleChoice (question : String) (letters : Array String) (choices : Array String)
    : String := Id.run do
  let mut result := s!"Multiple Choice question: {question}\n"
  for (letter, choice) in letters.zip choices do
    result := result ++ s!"- {choice}={letter}\n"
  result := result ++ "\nRespond only with the letter of the correct answer."
  result

/-! ## Task Base Structure -/

/-- Base task configuration -/
structure TaskConfig where
  /-- Starting index for slicing -/
  start : Nat := 0
  /-- Ending index (none = all) -/
  stop : Option Nat := none
  /-- Step size for slicing -/
  step : Nat := 1
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  deriving Repr

/-! ## ARC Task (AI2 Reasoning Challenge) -/

/-- ARC example from dataset -/
structure ARCExample where
  question : String
  choices : Array String
  labels : Array String  -- e.g., ["A", "B", "C", "D"]
  answerKey : String     -- e.g., "A"
  deriving Repr

/-- Parse ARC example from JSON -/
def ARCExample.fromJson? (json : Lean.Json) : Option ARCExample := do
  let question ← (json.getObjValAs? String "question").toOption
  let choicesObj ← (json.getObjVal? "choices").toOption
  let choices ← (choicesObj.getObjValAs? (Array String) "text").toOption
  let labels ← (choicesObj.getObjValAs? (Array String) "label").toOption
  let answerKey ← (json.getObjValAs? String "answerKey").toOption
  return { question, choices, labels, answerKey }

/-- ARC task implementation -/
structure ARCTask where
  /-- Subset: "ARC-Easy" or "ARC-Challenge" -/
  subset : String
  /-- Split: "train", "validation", or "test" -/
  split : String
  /-- Loaded examples -/
  examples : Array ARCExample
  /-- Configuration -/
  config : TaskConfig
  deriving Repr

def ARCTask.size (task : ARCTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def ARCTask.getExample (task : ARCTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  -- Render multiple choice question
  let userContent := renderMultipleChoice ex.question ex.labels ex.choices
  let messages := #[
    Message.user userContent,
    Message.assistant ex.answerKey
  ]
  return { messages, metadata := [("letters", String.intercalate "," ex.labels.toList)] }

def ARCTask.evaluate (_task : ARCTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := extractLetterAnswer response |>.getD response.trimAscii.toString
  let correct := predicted == expected
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## MMLU Task -/

/-- MMLU example from dataset -/
structure MMLUExample where
  question : String
  choices : Array String
  answer : Nat          -- Index 0-3
  subject : String
  deriving Repr

def MMLUExample.fromJson? (json : Lean.Json) : Option MMLUExample := do
  -- HF `auxiliary_train` rows can be wrapped as `{ "train": {...} }`.
  let root := ((json.getObjVal? "train").toOption).getD json
  let question ← (root.getObjValAs? String "question").toOption
  let choices ← (root.getObjValAs? (Array String) "choices").toOption
  let answer :=
    match (root.getObjValAs? Nat "answer").toOption with
    | some a => some a
    | none =>
      match (root.getObjValAs? Int "answer").toOption with
      | some a => Int.toNat? a
      | none => none
  let answer ← answer
  let subject := (root.getObjValAs? String "subject").toOption |>.getD ""
  return { question, choices, answer, subject }

/-- MMLU task implementation -/
structure MMLUTask where
  subset : String
  split : String
  examples : Array MMLUExample
  config : TaskConfig
  deriving Repr

def MMLUTask.letters : Array String := #["A", "B", "C", "D"]

def MMLUTask.size (task : MMLUTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def MMLUTask.getExample (task : MMLUTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  let userContent := renderMultipleChoice ex.question MMLUTask.letters ex.choices
  let answerLetter := MMLUTask.letters[ex.answer]!
  let messages := #[
    Message.user userContent,
    Message.assistant answerLetter
  ]
  return {
    messages
    metadata := [("subject", ex.subject), ("letters", "A,B,C,D")]
  }

def MMLUTask.evaluate (_task : MMLUTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := extractLetterAnswer response MMLUTask.letters |>.getD response.trimAscii.toString
  let correct := predicted == expected
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## SmolTalk Task -/

structure SmolTalkExample where
  messages : Array Message
  source : Option String := none
  deriving Repr

def SmolTalkExample.fromJson? (json : Lean.Json) : Option SmolTalkExample := do
  let rawMsgs ← (json.getObjVal? "messages" >>= (·.getArr?)).toOption
  let msgs := rawMsgs.filterMap parseMessageJson?
  if msgs.size != rawMsgs.size then
    none
  else if !validateAlternatingUserAssistant msgs then
    none
  else
    let source := (json.getObjValAs? String "source").toOption
    some { messages := msgs, source := source }

structure SmolTalkTask where
  split : String
  examples : Array SmolTalkExample
  config : TaskConfig
  deriving Repr

def SmolTalkTask.size (task : SmolTalkTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def SmolTalkTask.getExample (task : SmolTalkTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  let mut metadata : List (String × String) := []
  match ex.source with
  | some src => metadata := ("source", src) :: metadata
  | none => pure ()
  return { messages := ex.messages, metadata := metadata }

def SmolTalkTask.evaluate (_task : SmolTalkTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := response.trimAscii.toString
  let correct := !predicted.isEmpty && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## CustomJSON Task -/

structure CustomJSONTask where
  filepath : String
  examples : Array Conversation
  config : TaskConfig
  deriving Repr

def CustomJSONTask.size (task : CustomJSONTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def CustomJSONTask.getExample (task : CustomJSONTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  task.examples[physicalIdx]?

def CustomJSONTask.evaluate (_task : CustomJSONTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := response.trimAscii.toString
  let correct := !predicted.isEmpty && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## GSM8K Task (Grade School Math) -/

/-- GSM8K example -/
structure GSM8KExample where
  question : String
  answer : String       -- Full solution with #### marker
  deriving Repr

def GSM8KExample.fromJson? (json : Lean.Json) : Option GSM8KExample := do
  let question ← (json.getObjValAs? String "question").toOption
  let answer ← (json.getObjValAs? String "answer").toOption
  return { question, answer }

/-- Parse GSM8K answer into parts (text and tool calls) -/
def parseGSM8KAnswer (answer : String) : Array ContentPart := Id.run do
  let mut parts : Array ContentPart := #[]
  let mut remaining := answer

  -- Simple parsing: split on << and >>
  while true do
    match findSubstr remaining "<<" with
    | none =>
      -- No more tool calls, add remaining text
      if !remaining.isEmpty then
        parts := parts.push { type := .text, content := remaining }
      break
    | some startIdx =>
      -- Add text before tool call
      let textBefore := (remaining.take startIdx).toString
      if !textBefore.isEmpty then
        parts := parts.push { type := .text, content := textBefore }

      -- Find end of tool call
      let afterStart := (remaining.drop (startIdx + 2)).toString
      match findSubstr afterStart ">>" with
      | none =>
        -- Malformed, add rest as text
        parts := parts.push { type := .text, content := (remaining.drop startIdx).toString }
        break
      | some endIdx =>
        -- Parse tool call: expr=result
        let inner := (afterStart.take endIdx).toString
        let (expr, result) := match inner.splitOn "=" with
          | [e, r] => (e, r)
          | [e] => (e, "")
          | _ => (inner, "")
        parts := parts.push { type := .toolCall "calculator", content := expr }
        parts := parts.push { type := .toolResult, content := result }
        remaining := (afterStart.drop (endIdx + 2)).toString

  parts

/-- GSM8K task implementation -/
structure GSM8KTask where
  subset : String
  split : String
  examples : Array GSM8KExample
  config : TaskConfig
  deriving Repr

def GSM8KTask.size (task : GSM8KTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def GSM8KTask.getExample (task : GSM8KTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  -- Parse answer into parts for tool use
  let answerParts := parseGSM8KAnswer ex.answer
  let messages := #[
    Message.user ex.question,
    { role := .assistant, content := ex.answer, parts := answerParts }
  ]
  return { messages }

def GSM8KTask.evaluate (_task : GSM8KTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.bind (fun m => extractHashAnswer m.content) |>.getD ""
  let predicted := extractHashAnswer response |>.getD ""
  let correct := predicted == expected && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

def GSM8KTask.reward (task : GSM8KTask) (conv : Conversation) (response : String) : Float :=
  let result := task.evaluate conv response
  result.score

/-! ## SpellingBee Task -/

/-- Simple LCG random number generator -/
def lcgRandom (seed : UInt64) : UInt64 :=
  seed * 6364136223846793005 + 1442695040888963407

/-- Get random element from array -/
def randomChoice {α : Type} (arr : Array α) (seed : UInt64) : Option α :=
  if arr.isEmpty then none
  else arr[seed.toNat % arr.size]?

/-- Letters of the alphabet -/
def alphabet : Array Char :=
  #['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

/-- User message templates for SpellingBee -/
def spellingBeeTemplates : Array String := #[
  "How many {letter} are in the word {word}",
  "How many {letter} are in {word}",
  "Count the number of {letter} in {word}",
  "How many times does {letter} appear in {word}",
  "In the word {word}, how many {letter} are there",
  "Count how many {letter} appear in {word}",
  "How many {letter}s are in {word}",
  "Number of {letter} in {word}"
]

/-- SpellingBee task - count letter occurrences -/
structure SpellingBeeTask where
  /-- Word list -/
  words : Array String
  /-- Number of examples to generate -/
  size : Nat
  /-- Split: "train" or "test" -/
  split : String
  /-- Configuration -/
  config : TaskConfig
  deriving Repr

def SpellingBeeTask.getExample (task : SpellingBeeTask) (index : Nat) : Option Conversation := do
  let seed := if task.split == "test"
    then 10000000 + index.toUInt64
    else index.toUInt64

  -- Pick random word
  let wordSeed := lcgRandom seed
  let word ← randomChoice task.words wordSeed

  -- Pick letter (90% from word, 10% random)
  let letterSeed := lcgRandom wordSeed
  let useWordLetter := (letterSeed % 10) < 9
  let letterCharSeed := lcgRandom letterSeed
  let letter ← if useWordLetter then
    randomChoice word.toList.toArray letterCharSeed
  else
    randomChoice alphabet letterCharSeed

  -- Count occurrences
  let count := word.toList.filter (· == letter) |>.length

  -- Generate user message
  let templateSeed := lcgRandom letterCharSeed
  let template ← randomChoice spellingBeeTemplates templateSeed
  let userMsg := template.replace "{letter}" letter.toString |>.replace "{word}" word

  -- Generate assistant response with manual counting
  let wordLetters := String.intercalate "," (word.toList.map (·.toString))
  let mut manualText := s!"Let me count the '{letter}' in '{word}'.\n\n"
  manualText := manualText ++ s!"Spelling: {word}:{wordLetters}\n\n"

  let mut runningCount := 0
  let chars := word.toList
  for i in [:chars.length] do
    let c := chars[i]!
    if c == letter then
      runningCount := runningCount + 1
      manualText := manualText ++ s!"{i+1}:{c} hit! count={runningCount}\n"
    else
      manualText := manualText ++ s!"{i+1}:{c}\n"

  manualText := manualText ++ s!"\n#### {count}"

  let messages := #[
    Message.user userMsg,
    Message.assistant manualText
  ]

  return { messages }

def SpellingBeeTask.evaluate (_task : SpellingBeeTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.bind (fun m => extractHashAnswer m.content) |>.getD ""
  let predicted := extractHashAnswer response |>.getD ""
  let correct := predicted == expected && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## SimpleSpelling Task -/

/-- SimpleSpelling task - just spell words -/
structure SimpleSpellingTask where
  words : Array String
  size : Nat
  split : String
  config : TaskConfig
  deriving Repr

def SimpleSpellingTask.getExample (task : SimpleSpellingTask) (index : Nat) : Option Conversation := do
  let seed := if task.split == "test"
    then 10000000 + index.toUInt64
    else index.toUInt64

  let wordSeed := lcgRandom seed
  let word ← randomChoice task.words wordSeed
  let wordLetters := String.intercalate "," (word.toList.map (·.toString))

  let messages := #[
    Message.user s!"Spell the word: {word}",
    Message.assistant s!"{word}:{wordLetters}"
  ]

  return { messages }

/-! ## Task Loading from JSONL -/

/-- Load conversations from CustomJSON JSONL format (one JSON array of messages per line). -/
def loadCustomJSONConversations (path : System.FilePath) : IO (Array Conversation) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut convs : Array Conversation := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok (.arr rawMsgs) =>
      let msgs := rawMsgs.filterMap parseMessageJson?
      if msgs.size == rawMsgs.size && validateAlternatingUserAssistant msgs then
        convs := convs.push { messages := msgs }
    | .ok _ => pure ()
    | .error _ => pure ()
  return convs

/-- Load word list from text file (one word per line) -/
def loadWordList (path : System.FilePath) : IO (Array String) := do
  let content ← IO.FS.readFile path
  let words := content.splitOn "\n"
    |>.map (fun s => s.trimAscii.toString)
    |>.filter (!·.isEmpty)
  return words.toArray

/-! ## Task Factory Functions -/

/-- Create CustomJSON task from a local JSONL file. -/
def createCustomJSONTask (path : System.FilePath)
    (config : TaskConfig := {}) : IO CustomJSONTask := do
  let examples ← loadCustomJSONConversations path
  return { filepath := path.toString, examples, config }

/-- Create SpellingBee task from word list -/
def createSpellingBeeTask (wordListPath : System.FilePath) (size : Nat) (split : String)
    (config : TaskConfig := {}) : IO SpellingBeeTask := do
  let words ← loadWordList wordListPath
  return { words, size, split, config }

/-- Create SimpleSpelling task from word list -/
def createSimpleSpellingTask (wordListPath : System.FilePath) (size : Nat) (split : String)
    (config : TaskConfig := {}) : IO SimpleSpellingTask := do
  let words ← loadWordList wordListPath
  return { words, size, split, config }

/-! ## Task Typeclass Instances -/

/-- ARC task implements Task typeclass -/
instance : EvalTask ARCTask where
  name := "arc"
  evalType _ := .categorical
  size := ARCTask.size
  getExample := ARCTask.getExample
  evaluate := ARCTask.evaluate

/-- MMLU task implements Task typeclass -/
instance : EvalTask MMLUTask where
  name := "mmlu"
  evalType _ := .categorical
  size := MMLUTask.size
  getExample := MMLUTask.getExample
  evaluate := MMLUTask.evaluate

/-- GSM8K task implements Task typeclass -/
instance : EvalTask GSM8KTask where
  name := "gsm8k"
  evalType _ := .generative
  size := GSM8KTask.size
  getExample := GSM8KTask.getExample
  evaluate := GSM8KTask.evaluate
  reward := GSM8KTask.reward

/-- SmolTalk task implements Task typeclass. -/
instance : EvalTask SmolTalkTask where
  name := "smoltalk"
  evalType _ := .generative
  size := SmolTalkTask.size
  getExample := SmolTalkTask.getExample
  evaluate := SmolTalkTask.evaluate

/-- CustomJSON task implements Task typeclass. -/
instance : EvalTask CustomJSONTask where
  name := "customjson"
  evalType _ := .generative
  size := CustomJSONTask.size
  getExample := CustomJSONTask.getExample
  evaluate := CustomJSONTask.evaluate

/-- SpellingBee task implements Task typeclass -/
instance : EvalTask SpellingBeeTask where
  name := "spelling_bee"
  evalType _ := .generative
  size t := t.size
  getExample := SpellingBeeTask.getExample
  evaluate := SpellingBeeTask.evaluate

/-- SimpleSpelling task implements Task typeclass -/
instance : EvalTask SimpleSpellingTask where
  name := "simple_spelling"
  evalType _ := .generative
  size t := t.size
  getExample := SimpleSpellingTask.getExample
  evaluate _ _ response := { correct := true, score := 1.0, expected := "", predicted := response }

end torch.Tasks.LLM
