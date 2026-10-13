import Tyr.Basic

/-!
# Typed dtype policy

LeanMLX-style dtype algebra for Tyr's typed facade.

This module is intentionally pure: it does not change the runtime tensor
representation.  It gives higher layers a single place to ask what dtype an
operation should produce before they wrap or validate native tensor handles.
-/

namespace torch

namespace DType

def isBool : DType → _root_.Bool
  | .Bool => true
  | _ => false

def isUnsignedIntegral : DType → _root_.Bool
  | .UInt8 => true
  | _ => false

def isSignedIntegral : DType → _root_.Bool
  | .Int8 | .Int16 | .Int32 | .Int64 => true
  | _ => false

def isIntegral (dtype : DType) : _root_.Bool :=
  dtype.isUnsignedIntegral || dtype.isSignedIntegral

def isFloating : DType → _root_.Bool
  | .Float16 | .BFloat16 | .Float32 | .Float64
  | .Float8E4M3FN | .Float8E5M2 => true
  | _ => false

def isInexact : DType → _root_.Bool :=
  isFloating

def isDifferentiable : DType → _root_.Bool :=
  isInexact

def isIndex : DType → _root_.Bool
  | .UInt8 | .Int32 | .Int64 => true
  | _ => false

def bitWidth? : DType → Option Nat
  | .Bool => some 1
  | .UInt8 | .Int8 | .Float8E4M3FN | .Float8E5M2 => some 8
  | .Int16 => some 16
  | .Int32 | .Float32 => some 32
  | .Int64 | .Float64 => some 64
  | .Float16 | .BFloat16 => some 16
  | .Unknown _ => none

def atLeastFloat : DType → DType
  | .Bool | .UInt8
  | .Int8 | .Int16 | .Int32 | .Int64
  | .Float8E4M3FN | .Float8E5M2 => .Float32
  | dtype => dtype

private def intRank : DType → Nat
  | .Bool => 0
  | .UInt8 => 1
  | .Int8 => 2
  | .Int16 => 3
  | .Int32 => 4
  | .Int64 => 5
  | _ => 0

private def promoteIntegral (lhs rhs : DType) : DType :=
  -- Neither eight-bit type contains the full range of the other.
  if (lhs == .UInt8 && rhs == .Int8) || (lhs == .Int8 && rhs == .UInt8) then
    .Int16
  else
    match Nat.max (intRank lhs) (intRank rhs) with
    | 0 => .Bool
    | 1 => .UInt8
    | 2 => .Int8
    | 3 => .Int16
    | 4 => .Int32
    | _ => .Int64

def promote : DType → DType → DType
  | .Unknown raw, _ => .Unknown raw
  | _, .Unknown raw => .Unknown raw
  | .Float64, _ | _, .Float64 => .Float64
  | .Float32, _ | _, .Float32 => .Float32
  | .Float16, .BFloat16 | .BFloat16, .Float16 => .Float32
  | .Float16, _ | _, .Float16 => .Float16
  | .BFloat16, _ | _, .BFloat16 => .BFloat16
  | .Float8E4M3FN, .Float8E4M3FN => .Float8E4M3FN
  | .Float8E5M2, .Float8E5M2 => .Float8E5M2
  | .Float8E4M3FN, _ | _, .Float8E4M3FN => .Float32
  | .Float8E5M2, _ | _, .Float8E5M2 => .Float32
  | lhs, rhs => promoteIntegral lhs rhs

def absResult : DType → DType
  | dtype => dtype

def sumResult : DType → DType
  | .Bool | .UInt8 | .Int8 | .Int16 | .Int32 => .Int64
  | .Float8E4M3FN | .Float8E5M2 => .Float32
  | dtype => dtype

def prodResult : DType → DType :=
  sumResult

def meanResult (dtype : DType) : DType :=
  promote (sumResult dtype) (atLeastFloat dtype)

def divideResult (dtype : DType) : DType :=
  atLeastFloat dtype

def canRepresentSafeTensors (dtype : DType) : _root_.Bool :=
  dtype.safeTensorTag?.isSome

def expectedEq (expected actual : DType) : Except String Unit :=
  if actual == expected then
    .ok ()
  else
    .error s!"Expected dtype {expected}, got {actual}"

end DType

end torch
