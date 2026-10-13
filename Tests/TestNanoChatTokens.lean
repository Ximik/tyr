import LeanTest
import Tyr.Tokenizer
import Tyr.Data.Task

open tokenizer
open torch
open torch.Data.Task

private def byteEncode (s : String) : Array UInt64 :=
  s.toUTF8.toList.toArray.map (fun b => b.toUInt64)

private def zerosMask (n : Nat) : Array UInt8 :=
  Array.mk (List.replicate n (0 : UInt8))

private def onesMask (n : Nat) : Array UInt8 :=
  Array.mk (List.replicate n (1 : UInt8))

@[test]
def testDefaultChatSpecialTokensOrdering : IO Unit := do
  let expected : Array String := #[
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>"
  ]
  LeanTest.assertEqual defaultChatSpecialTokens expected

@[test]
def testEncodeWithSpecialsLongestMatch : IO Unit := do
  let tok := initBaseTokenizer #["<|assistant|>", "<|assistant_start|>"]
  let ids := tokenizer.encodeWithSpecials tok "<|assistant_start|>"
  let expectedId := (tok.specialTokens.get? "<|assistant_start|>").getD (0 : UInt32)
  LeanTest.assertEqual ids #[expectedId]
  LeanTest.assertEqual (tokenizer.decode tok ids) "<|assistant_start|>"

@[test]
def testFindBestMergeUsesPriorityMap : IO Unit := do
  let base := initBaseTokenizer #[]
  let a : TokenId := 97
  let b : TokenId := 98
  let c : TokenId := 99
  let bcId : TokenId := base.vocabSize
  let abId : TokenId := base.vocabSize + 1
  let ruleBC : MergeRule := { left := b, right := c, result := bcId }
  let ruleAB : MergeRule := { left := a, right := b, result := abId }

  let byteBC := ByteArray.mk #[(98 : UInt8), (99 : UInt8)]
  let byteAB := ByteArray.mk #[(97 : UInt8), (98 : UInt8)]
  let idToBytes := (base.idToBytes.push byteBC).push byteAB
  let bytesToId := (base.bytesToId.insert byteBC bcId).insert byteAB abId
  let mergeLookup : Std.HashMap (TokenId × TokenId) TokenId :=
    (({} : Std.HashMap (TokenId × TokenId) TokenId).insert (b, c) bcId).insert (a, b) abId
  let mergePriority : Std.HashMap (TokenId × TokenId) Nat :=
    (({} : Std.HashMap (TokenId × TokenId) Nat).insert (b, c) 0).insert (a, b) 1

  let tok : BPETokenizer := {
    base with
    vocabSize := base.vocabSize + 2
    idToBytes := idToBytes
    bytesToId := bytesToId
    merges := #[ruleBC, ruleAB]
    mergeLookup := mergeLookup
    mergePriority := mergePriority
  }

  match findBestMerge tok #[a, b, c] with
  | some (idx, rule) =>
    LeanTest.assertEqual idx 1
    LeanTest.assertEqual rule.left ruleBC.left
    LeanTest.assertEqual rule.right ruleBC.right
    LeanTest.assertEqual rule.result ruleBC.result
  | none =>
    LeanTest.fail "Expected findBestMerge to select the ranked merge"

  LeanTest.assertEqual (encodeWord tok "abc") #[a, bcId]

  let serialized <-
    match serialize tok with
    | .ok bytes => pure bytes
    | .error err => LeanTest.fail s!"Expected serialize to succeed: {err}"
  let roundTripped <-
    match deserialize serialized with
    | some decoded => pure decoded
    | none => LeanTest.fail "Expected deserialize to succeed after serialize"
  LeanTest.assertEqual (encodeWord roundTripped "abc") #[a, bcId]

@[test]
def testRenderConversationMasking : IO Unit := do
  let chatTokens : ChatTokens := {
    bos := 100
    userStart := 101
    userEnd := 102
    assistantStart := 103
    assistantEnd := 104
    pythonStart := 105
    pythonEnd := 106
    systemStart := 111
    systemEnd := 112
    toolStart := 113
    toolEnd := 114
    outputStart := 107
    outputEnd := 108
  }

  let conv : Conversation := Conversation.fromMessages #[
    Message.system "SYS",
    Message.user "USR",
    {
      role := .assistant
      content := ""
      parts := #[
        { type := .text, content := "TXT" },
        { type := .code "python", content := "print(1)" },
        { type := .toolResult, content := "1" }
      ]
    }
  ]

  let rendered := renderConversation conv chatTokens byteEncode

  let mergedUser := byteEncode "SYS\n\nUSR"
  let textPart := byteEncode "TXT"
  let pythonPart := byteEncode "print(1)"
  let outputPart := byteEncode "1"

  let expectedTokens : Array UInt64 :=
    #[chatTokens.bos, chatTokens.userStart] ++ mergedUser ++
    #[chatTokens.userEnd, chatTokens.assistantStart] ++
    textPart ++
    #[chatTokens.pythonStart] ++ pythonPart ++ #[chatTokens.pythonEnd] ++
    #[chatTokens.outputStart] ++ outputPart ++ #[chatTokens.outputEnd] ++
    #[chatTokens.assistantEnd]

  let expectedMask : Array UInt8 :=
    #[0, 0] ++ zerosMask mergedUser.size ++
    #[0, 0] ++
    onesMask textPart.size ++
    #[1] ++ onesMask pythonPart.size ++ #[1] ++
    #[0] ++ zerosMask outputPart.size ++ #[0] ++
    #[1]

  LeanTest.assertEqual rendered.tokens expectedTokens
  LeanTest.assertEqual rendered.mask expectedMask

  let userStartCount := rendered.tokens.foldl (fun acc tok =>
    if tok == chatTokens.userStart then acc + 1 else acc) 0
  LeanTest.assertEqual userStartCount 1

private def toyChatTokens : ChatTokens := {
  bos := 10
  userStart := 11
  userEnd := 12
  assistantStart := 13
  assistantEnd := 14
  pythonStart := 15
  pythonEnd := 16
  systemStart := 17
  systemEnd := 18
  toolStart := 19
  toolEnd := 20
  outputStart := 21
  outputEnd := 22
}

private def toyEncode (s : String) : Array UInt64 :=
  s.toUTF8.toList.toArray.map (fun b => 1000 + b.toUInt64)

@[test]
def testTaskTokenStreamWindowShift : IO Unit := do
  let conv := Conversation.fromMessages #[
    Message.user "u",
    Message.assistant "a"
  ]
  let task : LoadedTask := { name := "toy", conversations := #[conv], config := {} }
  let mix := TaskMixture.create #[{ task := task, weight := 1 }] 0
  let stream := TaskTokenStream.new mix.toConversationMixture 1 6 toyChatTokens toyEncode 0 1
  let (batch?, stream') ← stream.nextGPTBatch
  let some (inputs, targets) := batch?
    | LeanTest.fail "Expected TaskTokenStream to produce one batch"

  let rendered := renderConversation conv toyChatTokens toyEncode
  let needed := 1 * 6 + 1
  let expectedWindow := rendered.tokens.extract 0 needed
  let expectedInputs := expectedWindow.extract 0 (needed - 1)
  let expectedTargets := expectedWindow.extract 1 needed

  let inputFlat ← Tensor.toUInt64Array inputs.toInt64
  let targetFlat ← Tensor.toUInt64Array targets.toInt64
  LeanTest.assertEqual inputFlat expectedInputs
  LeanTest.assertEqual targetFlat expectedTargets
  LeanTest.assertEqual stream'.steps 1

@[test]
def testTaskTokenStreamRankStrideCursor : IO Unit := do
  let mkConv (u a : String) : Conversation :=
    Conversation.fromMessages #[Message.user u, Message.assistant a]

  let task : LoadedTask := {
    name := "rank_stride"
    conversations := #[
      mkConv "a" "x",
      mkConv "b" "y",
      mkConv "c" "z"
    ]
    config := {}
  }
  let mix := TaskMixture.create #[{ task := task, weight := 1 }] 123
  let stream := TaskTokenStream.new mix.toConversationMixture 1 3 toyChatTokens toyEncode 1 2

  let expectedConv ← mix.get 1
  let expectedRendered := renderConversation expectedConv toyChatTokens toyEncode
  let expectedThird := expectedRendered.tokens[2]?.getD 0

  let (batch?, stream') ← stream.nextGPTBatch
  let some (inputs, _targets) := batch?
    | LeanTest.fail "Expected first rank-strided batch"
  let inputFlat ← Tensor.toUInt64Array inputs.toInt64
  let gotThird := inputFlat[2]?.getD 0

  LeanTest.assertEqual gotThird expectedThird
  LeanTest.assertEqual stream'.cursor ((1 + 2) % mix.size)
  LeanTest.assertEqual stream'.lastWrapped true
