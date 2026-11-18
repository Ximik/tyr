/-
  Tyr/Data/Tasks.lean

  Concrete task implementations for instruction tuning.

  Based on nanochat's task types:
  - SmolTalk: General chat conversations
  - GSM8K: Math problems with calculator tool use
  - MMLU: Multiple choice reasoning
  - Spelling tasks: Character-level understanding
  - CustomJSON: Identity/personality conversations
-/
import Tyr.Data.Task

namespace torch.Data.Tasks

open torch.Data.Task

/-! ## Identity Task -/

/-- Create identity/personality conversations.
    Model learns its name and basic identity. -/
def createIdentityTask (modelName : String) (numExamples : Nat := 100) : LoadedTask := Id.run do
  let templates := #[
    ("What is your name?", s!"My name is {modelName}."),
    ("Who are you?", s!"I am {modelName}, an AI assistant."),
    ("Tell me about yourself.", s!"I'm {modelName}, a helpful AI assistant trained to be informative and safe."),
    ("What can you do?", "I can help with questions, writing, analysis, coding, math, and general conversation."),
    ("Are you an AI?", s!"Yes, I am {modelName}, an artificial intelligence assistant."),
    ("Who created you?", "I was created by a team of researchers and engineers."),
    ("What's your purpose?", "My purpose is to be helpful, harmless, and honest in my interactions.")
  ]

  let mut conversations : Array Conversation := #[]
  for i in [:numExamples] do
    let (question, answer) := templates[i % templates.size]!
    let conv : Conversation := {
      messages := #[Message.user question, Message.assistant answer]
    }
    conversations := conversations.push conv

  return { name := "identity", conversations, config := {} }

/-! ## Math Task (GSM8K style) -/

/-- Create a simple math problem with optional calculator tool call -/
def createMathConversation (question answer : String) (useCalculator : Bool := false)
    : Conversation := Id.run do
  let mut messages : Array Message := #[Message.user question]

  if useCalculator then
    -- Assistant thinks and uses calculator
    messages := messages.push {
      role := .assistant
      content := s!"Let me work through this step by step."
      parts := #[{ type := .toolCall "calculator", content := "2+2" }]
    }
    messages := messages.push {
      role := .tool
      content := "4"
    }
    messages := messages.push (Message.assistant answer)
  else
    messages := messages.push (Message.assistant answer)

  { messages }

/-- Sample math problems for testing -/
def sampleMathProblems : Array (String × String × Bool) := #[
  ("What is 2 + 2?", "2 + 2 = 4", false),
  ("If I have 5 apples and buy 3 more, how many do I have?", "5 + 3 = 8 apples", true),
  ("What is 15 divided by 3?", "15 / 3 = 5", true),
  ("A rectangle has length 4 and width 3. What is its area?", "Area = 4 × 3 = 12 square units", true),
  ("What is 7 × 8?", "7 × 8 = 56", true)
]

def createMathTask : LoadedTask :=
  let conversations := sampleMathProblems.map fun (q, a, useCalc) =>
    createMathConversation q a useCalc
  { name := "math", conversations, config := {} }

/-! ## Spelling Task -/

/-- Create a spelling task conversation -/
def createSpellingConversation (word : String) : Conversation :=
  let spelled := String.intercalate "-" (word.toList.map (·.toString))
  {
    messages := #[
      Message.user s!"Spell the word '{word}'",
      Message.assistant s!"The word '{word}' is spelled: {spelled}"
    ]
  }

/-- Create a letter counting task -/
def createLetterCountConversation (word : String) (letter : Char) : Conversation :=
  let count := word.toList.filter (· == letter) |>.length
  {
    messages := #[
      Message.user s!"How many times does the letter '{letter}' appear in '{word}'?",
      Message.assistant s!"The letter '{letter}' appears {count} time(s) in '{word}'."
    ]
  }

def sampleWords : Array String := #[
  "hello", "world", "programming", "artificial", "intelligence",
  "computer", "science", "mathematics", "algorithm", "neural"
]

def createSpellingTask : LoadedTask := Id.run do
  let mut conversations : Array Conversation := #[]

  -- Spelling tasks
  for word in sampleWords do
    conversations := conversations.push (createSpellingConversation word)

  -- Letter counting tasks
  for word in sampleWords do
    for letter in ['a', 'e', 'i', 'o', 'l', 'r', 's', 't'] do
      conversations := conversations.push (createLetterCountConversation word letter)

  return { name := "spelling", conversations, config := {} }

/-! ## Multiple Choice Task (MMLU style) -/

/-- Create a multiple choice question conversation -/
def createMultipleChoiceConversation
    (question : String)
    (choices : Array String)
    (correctIdx : Nat)
    (explanation : String := "")
    : Conversation :=
  let choiceLabels := #["A", "B", "C", "D"]
  let choiceText := choices.mapIdx fun i c =>
    s!"{choiceLabels[i]!}. {c}"
  let fullQuestion := question ++ "\n\n" ++ String.intercalate "\n" choiceText.toList

  let answer := if explanation.isEmpty then
    s!"The answer is {choiceLabels[correctIdx]!}. {choices[correctIdx]!}"
  else
    s!"The answer is {choiceLabels[correctIdx]!}. {choices[correctIdx]!}\n\n{explanation}"

  {
    messages := #[
      Message.user fullQuestion,
      Message.assistant answer
    ]
  }

def sampleMCQuestions : Array (String × Array String × Nat × String) := #[
  ("What is the capital of France?",
   #["Berlin", "Paris", "London", "Madrid"],
   1,
   "Paris has been the capital of France since the 10th century."),
  ("Which planet is known as the Red Planet?",
   #["Venus", "Mars", "Jupiter", "Saturn"],
   1,
   "Mars appears red due to iron oxide on its surface."),
  ("What is H2O commonly known as?",
   #["Salt", "Water", "Sugar", "Oil"],
   1,
   "H2O is the chemical formula for water.")
]

def createMCTask : LoadedTask :=
  let conversations := sampleMCQuestions.map fun (q, choices, correct, expl) =>
    createMultipleChoiceConversation q choices correct expl
  { name := "multiple_choice", conversations, config := {} }

/-! ## Task Mixture Creation -/

/-- Create a standard midtraining task mixture.
    Combines multiple task types with weights matching nanochat. -/
def createMidtrainingMixture (modelName : String) : TaskMixture :=
  TaskMixture.create #[
    { task := createIdentityTask modelName 2000, weight := 1 },
    { task := createMathTask, weight := 1 },
    { task := createSpellingTask, weight := 1 },
    { task := createMCTask, weight := 1 }
  ]

/-- Create a standard SFT task mixture (smaller, higher quality) -/
def createSFTMixture (modelName : String) : TaskMixture :=
  TaskMixture.create #[
    { task := createIdentityTask modelName 500, weight := 1 },
    { task := createMathTask, weight := 2 },     -- Oversample math
    { task := createMCTask, weight := 2 }        -- Oversample MC
  ]

end torch.Data.Tasks
