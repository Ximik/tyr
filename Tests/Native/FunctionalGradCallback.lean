import Tyr.FFI.RawTensor

-- Callbacks for `cc/tools/test_functional_grad_callback.cpp`, compiled by Lean so
-- they carry its real ownership code.

open torch

@[export tyr_test_callback_sum] private def sumCallback
    (fallback : RawTensor) (inputs : Array RawTensor) : RawTensor :=
  match inputs[0]? with
  | some input => RawTensor.sum! input none false
  | none => fallback

@[export tyr_test_callback_identity] private def identityCallback
    (fallback : RawTensor) (inputs : Array RawTensor) : RawTensor :=
  inputs[0]?.getD fallback

@[export tyr_test_callback_throw] private def throwingCallback
    (fallback : RawTensor) (inputs : Array RawTensor) : RawTensor :=
  match inputs[0]? with
  | some input => RawTensor.reshape! input #[3]
  | none => fallback
