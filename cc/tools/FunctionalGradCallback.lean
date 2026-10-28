import Init

-- Compile the callback with Lean's real ownership transformations, while
-- avoiding project module builds in this native subprocess regression.
private opaque RawSpec : NonemptyType := ⟨Unit, inferInstance⟩
private def Raw : Type := RawSpec.type
private instance : Nonempty Raw := RawSpec.property

@[extern "lean_torch_reshape_exact"] private opaque reshape
    (input : @& Raw) (target : @& Array Nat) : Raw
@[extern "lean_torch_sum"] private opaque sum
    (input : @& Raw) (dims : Option (Array Nat)) (keepdim : Bool) : Raw

@[export tyr_test_callback_sum] private def sumCallback
    (fallback : Raw) (inputs : Array Raw) : Raw :=
  match inputs[0]? with
  | some input => sum input none false
  | none => fallback

@[export tyr_test_callback_identity] private def identityCallback
    (fallback : Raw) (inputs : Array Raw) : Raw :=
  inputs[0]?.getD fallback

@[export tyr_test_callback_throw] private def throwingCallback
    (fallback : Raw) (inputs : Array Raw) : Raw :=
  match inputs[0]? with
  | some input => reshape input #[3]
  | none => fallback
