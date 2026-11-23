/-
  Examples/NanoChat/Eval/COREData.lean

  Load CORE evaluation data from the eval bundle.

  The CORE evaluation bundle contains JSONL files for each task:
  - arc_easy.jsonl, arc_challenge.jsonl
  - hellaswag.jsonl
  - winogrande.jsonl
  - piqa.jsonl
  - sciq.jsonl
  - boolq.jsonl
  - lambada.jsonl

  Each file contains evaluation examples in a task-specific format.
-/
import Tyr.Data.Download
import Tyr.Data.HuggingFace
import Examples.NanoChat.Eval.CORE
import Lean.Data.Json

namespace torch.Eval.COREData

open torch.Data.Download
open torch.Data.HuggingFace
open torch.Eval.CORE

/-! ## CORE Task Configuration -/

/-- Configuration for a CORE evaluation task -/
structure CORETaskConfig where
  /-- Task label (e.g., "arc_easy") -/
  label : String
  /-- Task type (MC, schema, LM) -/
  taskType : TaskType
  /-- Path to dataset file within eval bundle -/
  datasetPath : String
  /-- Number of few-shot examples -/
  numFewshot : Nat := 0
  /-- Delimiter between context and continuation -/
  continuationDelimiter : String := " "
  deriving Repr, Inhabited

/-- Default CORE task configurations matching the DCLM benchmark -/
def defaultCOREConfigs : Array CORETaskConfig := #[
  -- ARC (AI2 Reasoning Challenge)
  { label := "arc_easy"
    taskType := .multipleChoice
    datasetPath := "eval_data/world_knowledge/arc_easy.jsonl"
    numFewshot := 0
    continuationDelimiter := "\nAnswer: " },
  { label := "arc_challenge"
    taskType := .multipleChoice
    datasetPath := "eval_data/world_knowledge/arc_challenge.jsonl"
    numFewshot := 0
    continuationDelimiter := "\nAnswer: " },
  -- HellaSwag
  { label := "hellaswag"
    taskType := .multipleChoice
    datasetPath := "eval_data/language_understanding/hellaswag.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- Winogrande
  { label := "winogrande"
    taskType := .schema
    datasetPath := "eval_data/language_understanding/winogrande.jsonl"
    numFewshot := 0
    continuationDelimiter := "" },
  -- PIQA
  { label := "piqa"
    taskType := .multipleChoice
    datasetPath := "eval_data/commonsense_reasoning/piqa.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- MMLU
  { label := "mmlu"
    taskType := .multipleChoice
    datasetPath := "eval_data/world_knowledge/mmlu.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- BoolQ
  { label := "boolq"
    taskType := .multipleChoice
    datasetPath := "eval_data/reading_comprehension/boolq.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- LAMBADA
  { label := "lambada"
    taskType := .languageModeling
    datasetPath := "eval_data/language_understanding/lambada_openai.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- GSM8K
  { label := "gsm8k"
    taskType := .languageModeling
    datasetPath := "eval_data/symbolic_problem_solving/gsm8k.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- HumanEval
  { label := "humaneval"
    taskType := .languageModeling
    datasetPath := "eval_data/programming/human_eval.jsonl"
    numFewshot := 0
    continuationDelimiter := " " }
]

/-! ## JSON Parsing Utilities -/

/-- Parse multiple-choice item from standard CORE fields.
    Format: {"query": str, "choices": [str], "gold": int} -/
def parseMCQueryChoicesGold (json : Lean.Json) : Option MCItem := do
  let query ← getJsonString json "query"
  let choices ← getJsonStringArray json "choices"
  let gold ← getJsonNat json "gold"
  return { query, choices, gold }

/-- Parse ARC example from JSON -/
def parseARCItem (json : Lean.Json) : Option MCItem := do
  if let some item := parseMCQueryChoicesGold json then
    return item
  let question ← getJsonString json "question"
  let choicesObj ← (json.getObjVal? "choices").toOption
  let choices ← getJsonStringArray choicesObj "text"
  let labels ← getJsonStringArray choicesObj "label"
  let answerKey ← getJsonString json "answerKey"
  -- Find gold index from answerKey
  let goldIdx := labels.findIdx? (· == answerKey) |>.getD 0
  return { query := question, choices, gold := goldIdx }

/-- Parse HellaSwag example from JSON.
    Format: {"ctx": str, "endings": [str], "label": str} -/
def parseHellaSwagItem (json : Lean.Json) : Option MCItem := do
  if let some item := parseMCQueryChoicesGold json then
    return item
  let ctx ← getJsonString json "ctx"
  let endings ← getJsonStringArray json "endings"
  let labelStr ← getJsonString json "label"
  let gold := labelStr.toNat?.getD 0
  return { query := ctx, choices := endings, gold }

/-- Parse Winogrande example from JSON.
    Format: {"sentence": str, "option1": str, "option2": str, "answer": str} -/
def parseWinograndeItem (json : Lean.Json) : Option SchemaItem := do
  if let some contextOptions := getJsonStringArray json "context_options" then
    let continuation ← getJsonString json "continuation"
    let gold ← getJsonNat json "gold"
    return { contextOptions, continuation, gold }
  let sentence ← getJsonString json "sentence"
  let option1 ← getJsonString json "option1"
  let option2 ← getJsonString json "option2"
  let answer ← getJsonString json "answer"
  -- Replace _ with options to create context options
  let ctx1 := sentence.replace "_" option1
  let ctx2 := sentence.replace "_" option2
  -- answer is "1" or "2"
  let gold := if answer == "1" then 0 else 1
  -- The continuation is empty for Winogrande (schema task)
  return { contextOptions := #[ctx1, ctx2], continuation := "", gold }

/-- Parse PIQA example from JSON.
    Format: {"goal": str, "sol1": str, "sol2": str, "label": int} -/
def parsePIQAItem (json : Lean.Json) : Option MCItem := do
  if let some item := parseMCQueryChoicesGold json then
    return item
  let goal ← getJsonString json "goal"
  let sol1 ← getJsonString json "sol1"
  let sol2 ← getJsonString json "sol2"
  let gold ← getJsonNat json "label"
  return { query := goal, choices := #[sol1, sol2], gold }

/-- Parse SciQ example from JSON.
    Format: {"question": str, "distractor1": str, "distractor2": str, "distractor3": str,
             "correct_answer": str, "support": str} -/
def parseSciQItem (json : Lean.Json) : Option MCItem := do
  if let some item := parseMCQueryChoicesGold json then
    return item
  let question ← getJsonString json "question"
  let correct ← getJsonString json "correct_answer"
  let d1 ← getJsonString json "distractor1"
  let d2 ← getJsonString json "distractor2"
  let d3 ← getJsonString json "distractor3"
  -- Put correct answer first, then distractors
  return { query := question, choices := #[correct, d1, d2, d3], gold := 0 }

/-- Parse BoolQ example from JSON.
    Format: {"question": str, "passage": str, "answer": bool} -/
def parseBoolQItem (json : Lean.Json) : Option MCItem := do
  if let some item := parseMCQueryChoicesGold json then
    return item
  let question ← getJsonString json "question"
  let passage ← getJsonString json "passage"
  let answer ← getJsonBool json "answer"
  let query := s!"{passage}\nQuestion: {question}"
  let gold := if answer then 0 else 1  -- Yes=0, No=1
  return { query, choices := #["Yes", "No"], gold }

/-- Parse LAMBADA example from JSON.
    Format: {"text": str} - last word is the target -/
def parseLAMBADAItem (json : Lean.Json) : Option LMItem := do
  if let some context := getJsonString json "context" then
    if let some continuation := getJsonString json "continuation" then
      return { context, continuation }
    if let some answer := getJsonString json "answer" then
      return { context, continuation := answer }
  if let some prompt := getJsonString json "prompt" then
    if let some solution := getJsonString json "canonical_solution" then
      return { context := prompt, continuation := solution }
  let text ← getJsonString json "text"
  -- Split off last word as continuation
  let words := text.splitOn " "
  if words.length < 2 then none
  let context := String.intercalate " " (words.dropLast)
  let continuation := words.getLast!
  return { context, continuation }

/-! ## Loading CORE Data -/

/-- Load CORE task config from metadata file.
    If no metadata file exists, uses default configs. -/
def loadCOREConfig (bundleDir : String) : IO (Array CORETaskConfig) := do
  -- Check if a config file exists
  let configPath := s!"{bundleDir}/core_config.json"
  if ← fileExists configPath then
    -- Parse config file
    let content ← IO.FS.readFile ⟨configPath⟩
    match Lean.Json.parse content with
    | .error _ => return defaultCOREConfigs
    | .ok _json =>
      -- Parse array of configs (simplified - in practice would parse full structure)
      return defaultCOREConfigs
  else
    return defaultCOREConfigs

/-- Load CORE task data from eval bundle.
    Returns array of EvalItem for the specified task. -/
def loadCORETaskData (bundleDir : String) (taskConfig : CORETaskConfig)
    : IO (Array EvalItem) := do
  let dataPath := s!"{bundleDir}/{taskConfig.datasetPath}"

  -- Check if file exists
  if !(← fileExists dataPath) then
    IO.eprintln s!"Warning: CORE data file not found: {dataPath}"
    return #[]

  -- Read JSONL file
  let jsons ← loadJsonlFromFile dataPath

  -- Parse based on task type
  let mut items : Array EvalItem := #[]
  for json in jsons do
    match taskConfig.taskType with
    | .multipleChoice =>
      let parsed := match taskConfig.label with
        | "arc_easy" | "arc_challenge" | "mmlu" => parseARCItem json
        | "hellaswag" => parseHellaSwagItem json
        | "piqa" => parsePIQAItem json
        | "sciq" => parseSciQItem json
        | "boolq" => parseBoolQItem json
        | _ => parseMCQueryChoicesGold json
      if let some item := parsed then
        items := items.push (.mc item)
    | .schema =>
      if let some item := parseWinograndeItem json then
        items := items.push (.schema item)
    | .languageModeling =>
      if let some item := parseLAMBADAItem json then
        items := items.push (.lm item)

  IO.println s!"  Loaded {items.size} examples for {taskConfig.label}"
  return items

/-- Find CORE config for a given TaskMeta -/
private def normalizeTaskLabel (s : String) : String :=
  ((s.toLower.replace "_" "").replace "-" "").replace " " ""

def coreTaskMetaToConfig (taskMeta : TaskMeta) (configs : Array CORETaskConfig)
    : Option CORETaskConfig :=
  configs.find? fun cfg =>
    normalizeTaskLabel cfg.label == normalizeTaskLabel taskMeta.taskName

end torch.Eval.COREData
