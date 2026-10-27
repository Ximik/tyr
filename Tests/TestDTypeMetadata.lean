import LeanTest
import Tyr.Basic.Types

namespace Tests.DTypeMetadata

open torch

@[test] def testKnownDtypeInventoryRoundtrips : IO Unit := do
  LeanTest.assertEqual DType.all.size 46
  LeanTest.assertEqual DType.all.toList.eraseDups.length 46
  for dtype in DType.all do
    LeanTest.assertEqual (DType.ofString? dtype.canonicalName) (some dtype)
      s!"Canonical dtype spelling must roundtrip: {dtype}"
    if let some tag := dtype.safeTensorTag? then
      LeanTest.assertEqual (DType.ofString? tag) (some dtype)
  LeanTest.assertEqual (DType.all.filter (·.safeTensorTag?.isSome)).size 12
    "Metadata recognition must not widen the native SafeTensors storage support."

@[test] def testNativeAndPythonDtypeAliases : IO Unit := do
  let aliases : Array (String × DType) := #[
    ("Byte", .UInt8), ("torch.uint8", .UInt8),
    ("Char", .Int8), ("torch.int8", .Int8),
    ("Short", .Int16), ("torch.int16", .Int16),
    ("Int", .Int32), ("torch.int32", .Int32),
    ("Long", .Int64), ("torch.int64", .Int64),
    ("Half", .Float16), ("torch.float16", .Float16),
    ("BFloat16", .BFloat16), ("torch.bfloat16", .BFloat16),
    ("Float", .Float32), ("torch.float32", .Float32),
    ("Double", .Float64), ("torch.float64", .Float64),
    ("Bool", .Bool), ("torch.bool", .Bool),
    ("Float8_e4m3fn", .Float8E4M3FN), ("torch.float8_e4m3fn", .Float8E4M3FN),
    ("Float8_e5m2", .Float8E5M2), ("torch.float8_e5m2", .Float8E5M2),
    ("ComplexHalf", .Complex32), ("torch.complex32", .Complex32),
    ("ComplexFloat", .Complex64), ("torch.complex64", .Complex64),
    ("ComplexDouble", .Complex128), ("torch.complex128", .Complex128),
    ("QInt8", .QInt8), ("torch.qint8", .QInt8),
    ("QUInt8", .QUInt8), ("torch.quint8", .QUInt8),
    ("QInt32", .QInt32), ("torch.qint32", .QInt32),
    ("QUInt4x2", .QUInt4x2), ("torch.quint4x2", .QUInt4x2),
    ("QUInt2x4", .QUInt2x4), ("torch.quint2x4", .QUInt2x4),
    ("Bits1x8", .Bits1x8), ("torch.bits1x8", .Bits1x8),
    ("Bits2x4", .Bits2x4), ("torch.bits2x4", .Bits2x4),
    ("Bits4x2", .Bits4x2), ("torch.bits4x2", .Bits4x2),
    ("Bits8", .Bits8), ("torch.bits8", .Bits8),
    ("Bits16", .Bits16), ("torch.bits16", .Bits16),
    ("Float8_e5m2fnuz", .Float8E5M2FNUZ), ("torch.float8_e5m2fnuz", .Float8E5M2FNUZ),
    ("Float8_e4m3fnuz", .Float8E4M3FNUZ), ("torch.float8_e4m3fnuz", .Float8E4M3FNUZ),
    ("UInt16", .UInt16), ("torch.uint16", .UInt16),
    ("UInt32", .UInt32), ("torch.uint32", .UInt32),
    ("UInt64", .UInt64), ("torch.uint64", .UInt64),
    ("UInt1", .UInt1), ("torch.uint1", .UInt1), ("torch.bit", .UInt1),
    ("UInt2", .UInt2), ("torch.uint2", .UInt2),
    ("UInt3", .UInt3), ("torch.uint3", .UInt3),
    ("UInt4", .UInt4), ("torch.uint4", .UInt4),
    ("UInt5", .UInt5), ("torch.uint5", .UInt5),
    ("UInt6", .UInt6), ("torch.uint6", .UInt6),
    ("UInt7", .UInt7), ("torch.uint7", .UInt7),
    ("Int1", .Int1), ("torch.int1", .Int1),
    ("Int2", .Int2), ("torch.int2", .Int2),
    ("Int3", .Int3), ("torch.int3", .Int3),
    ("Int4", .Int4), ("torch.int4", .Int4),
    ("Int5", .Int5), ("torch.int5", .Int5),
    ("Int6", .Int6), ("torch.int6", .Int6),
    ("Int7", .Int7), ("torch.int7", .Int7),
    ("Float8_e8m0fnu", .Float8E8M0FNU), ("torch.float8_e8m0fnu", .Float8E8M0FNU),
    ("Float4_e2m1fn_x2", .Float4E2M1FNX2), ("torch.float4_e2m1fn_x2", .Float4E2M1FNX2)
  ]
  for (token, dtype) in aliases do
    LeanTest.assertEqual (DType.ofString? token) (some dtype) s!"Unknown dtype alias {token}"
  for (token, dtype) in #[ ("torch.chalf", DType.Complex32),
      ("cfloat", DType.Complex64), ("cdouble", DType.Complex128) ] do
    LeanTest.assertEqual (DType.ofString? token) (some dtype)
  for token in #["Undefined", "NumOptions", "dynamic", "ComplexFloat_typo"] do
    LeanTest.assertEqual (DType.ofString? token) none

end Tests.DTypeMetadata
