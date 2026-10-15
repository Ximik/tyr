#!/usr/bin/env python3
"""Adversarial fixtures for the recoverable native-export source gate."""

from pathlib import Path
import tempfile
import unittest

from check_ffi_recovery import audit, lean_exports


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tyr-ffi-recovery-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "Tyr/RawTensor").mkdir(parents=True)
        (self.root / "Tyr/Data").mkdir(parents=True)
        (self.root / "cc/src").mkdir(parents=True)
        self.write("Tyr/Basic.lean", "")
        self.write("Tyr/Torch.lean", "")
        self.write("Tyr/Data/Pretraining.lean", "")
        self.declare()
        self.native('return tyr::checkedIoCall("example", [&] { return run(); });')

    def write(self, path, source):
        (self.root / path).write_text(source)

    def declare(self, carrier="IO RawTensor", symbol="lean_torch_example"):
        self.write("Tyr/RawTensor.lean",
                   f'@[extern "{symbol}"] opaque example (value : @& RawTensor) : {carrier}\n')

    def native(self, body, path="cc/src/example.cpp", symbol="lean_torch_example"):
        self.write(path, f'extern "C" LEAN_EXPORT lean_object* {symbol}(b_lean_obj_arg x) {{\n{body}\n}}\n')

    def errors(self):
        return audit(self.root)[0]

    def test_guarded_io_and_except_exports(self):
        self.assertEqual(self.errors(), [])
        self.declare("Except String (Array RawTensor)")
        self.native('return tyr::checkedCall("example", [&] { return run(); });')
        self.assertEqual(self.errors(), [])

    def test_new_declaration_is_not_silently_omitted(self):
        self.write("Tyr/RawTensor/New.lean",
                   '@[extern "new_export"] private opaque fresh : IO Unit\n')
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("Tyr/RawTensor/New.lean:1", errors[0])
        self.assertIn("no recognized native definition for new_export", errors[0])

    def test_pretraining_io_declarations_are_in_scope_without_symbol_list(self):
        self.write("Tyr/Data/Pretraining.lean", '''
@[extern "lean_parquet_get_metadata"] opaque metadata : IO ParquetMetadata
@[extern "lean_parquet_future_operation"] opaque futureOperation : IO Unit
''')
        self.native('return read_metadata();', "cc/src/tyr_parquet.cpp",
                    "lean_parquet_get_metadata")
        errors = self.errors()
        self.assertTrue(any("lean_parquet_get_metadata from Tyr/Data/Pretraining.lean:2" in error
                            and "must contain only" in error for error in errors))
        self.assertTrue(any("no recognized native definition for lean_parquet_future_operation" in error
                            for error in errors))

    def test_torch_runtime_io_declarations_are_in_scope_without_symbol_list(self):
        self.write("Tyr/Torch.lean", '''
@[extern "lean_torch_resample_soxr_hq"] opaque resample : IO (Array Float)
@[extern "lean_torch_future_runtime_operation"] opaque future : IO Unit
''')
        self.native('return resample();', "cc/src/tyr.cpp", "lean_torch_resample_soxr_hq")
        errors = self.errors()
        self.assertTrue(any("lean_torch_resample_soxr_hq from Tyr/Torch.lean:2" in error
                            and "must contain only" in error for error in errors))
        self.assertTrue(any("no recognized native definition for lean_torch_future_runtime_operation" in error
                            for error in errors))

    def test_parquet_arrow_and_disabled_variants_are_both_checked(self):
        symbol = "lean_parquet_get_metadata"
        self.write("Tyr/Data/Pretraining.lean",
                   f'@[extern "{symbol}"] opaque metadata : IO ParquetMetadata\n')
        guarded = 'return tyr::checkedIoCall("metadata", [&] { return read_metadata(); });'
        unavailable = 'return tyr::checkedIoCall("metadata", [&] { return parquet_unavailable(); });'

        def variants(enabled, disabled):
            self.write("cc/src/tyr_parquet.cpp",
                       f'#if TYR_HAS_ARROW\nlean_object* {symbol}() {{ {enabled} }}\n'
                       f'#else\nlean_object* {symbol}() {{ {disabled} }}\n#endif\n')

        variants(guarded, unavailable)
        self.assertEqual(self.errors(), [])
        self.assertEqual(audit(self.root)[1]["guarded definitions"], 3)
        variants('return read_metadata();', unavailable)
        self.assertTrue(any("cc/src/tyr_parquet.cpp:2" in error for error in self.errors()))
        for disabled in ['return parquet_unavailable();', 'std::string message; ' + unavailable]:
            with self.subTest(disabled=disabled):
                variants(guarded, disabled)
                self.assertTrue(any("cc/src/tyr_parquet.cpp:4" in error for error in self.errors()))

    def test_unreviewed_native_modules_are_not_silently_claimed_in_scope(self):
        (self.root / "Tyr/GPU").mkdir()
        self.write("Tyr/GPU/Unreviewed.lean",
                   '@[extern "unreviewed_gpu_export"] opaque launch : IO Unit\n')
        self.native('return unguarded();', "cc/src/unreviewed.cpp", "unreviewed_gpu_export")
        self.assertEqual(self.errors(), [])

    def test_io_in_argument_and_comments_is_not_a_result(self):
        declarations = lean_exports('''
/- @[extern "comment"] opaque bogus : IO Unit
   /- nested comment -/ -/
@[extern "pure"] opaque plain (callback : IO Unit) : RawTensor
@[extern "checked", inline] private opaque foo
    (f : IO RawTensor → Except String RawTensor) (s : String := "IO") :
    (_root_.Except String (Array RawTensor))
''', "fixture.lean")
        self.assertEqual([(item.symbol, item.carrier) for item in declarations],
                         [("checked", "Except")])

    def test_unsupported_extern_syntax_fails_closed(self):
        self.write("Tyr/RawTensor.lean", '@[extern nativeName] opaque example : IO Unit\n')
        self.assertTrue(any("literal C symbol" in error for error in self.errors()))
        self.write("Tyr/RawTensor.lean", '@[extern "example"] opaque example := IO Unit\n')
        self.assertTrue(any("explicit result type" in error for error in self.errors()))

    def test_result_alias_cannot_hide_recoverable_export(self):
        self.write("Tyr/RawTensor.lean", '''
abbrev NativeResult := IO RawTensor
@[extern "lean_torch_example"] opaque example : NativeResult
''')
        self.assertTrue(any("unrecognized extern result constructor NativeResult" in error
                            for error in self.errors()))

    def test_except_error_type_must_match_native_string_payload(self):
        for result in ["Except UInt64 RawTensor", "Except String.Custom RawTensor"]:
            with self.subTest(result=result):
                self.declare(result)
                self.native('return tyr::checkedCall("example", [&] { return run(); });')
                self.assertTrue(any("checkedCall constructs Except String" in error
                                    for error in self.errors()))

    def test_qualified_result_alias_cannot_look_like_pure_constructor(self):
        self.declare("RawTensor.Recovery")
        self.assertTrue(any("qualified extern result constructor" in error
                            for error in self.errors()))

    def test_work_before_guard_is_rejected_including_release(self):
        for prelude in ['auto tensor = borrowTensor(x);', 'lean_dec(x);',
                        'try { return run(); } catch (...) { return error(); }']:
            with self.subTest(prelude=prelude):
                self.native(prelude + '\nreturn tyr::checkedIoCall("x", [&] { return run(); });')
                self.assertTrue(any("no work before the guard" in error for error in self.errors()))

    def test_pre_call_argument_evaluation_is_rejected(self):
        for body in [
            'return tyr::checkedIoCall(makeContext(), [&] { return run(); });',
            'return tyr::checkedIoCall("x", [value = allocate()] { return run(value); });',
            'return tyr::checkedIoCall("x", callable());',
        ]:
            with self.subTest(body=body):
                self.native(body)
                self.assertTrue(self.errors())

    def test_wrong_carrier_and_trailing_work_are_rejected(self):
        self.native('return tyr::checkedCall("x", [&] { return run(); });')
        self.assertTrue(any("checkedIoCall" in error for error in self.errors()))
        self.native('return tyr::checkedIoCall("x", [&] { return run(); }); other();')
        self.assertTrue(self.errors())

    def test_every_conditional_definition_is_checked(self):
        self.write("cc/src/example.cpp", '''
#if GPU
lean_object* lean_torch_example(b_lean_obj_arg x) {
  return tyr::checkedIoCall("x", [&] { return run(); });
}
#else
lean_object* lean_torch_example(b_lean_obj_arg x) { return unguarded(); }
#endif
''')
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("cc/src/example.cpp:7", errors[0])

    def test_platform_variant_return_types_and_attributes_cannot_hide_body(self):
        headers = [
            'NativeResult lean_torch_example()',
            'auto lean_torch_example() -> lean_object*',
            'auto lean_torch_example() noexcept(false) -> NativeResult',
            'lean_object* lean_torch_example() [[gnu::cold]]',
            'lean_object* lean_torch_example() __attribute__((cold))',
        ]
        good = 'return tyr::checkedIoCall("x", [&] { return run(); });'
        for header in headers:
            with self.subTest(header=header):
                # The recognized good branch must not mask a missed bad branch.
                prefix = ('using NativeResult = lean_object*;\n#if GPU\n'
                          'lean_object* lean_torch_example() { ' + good + ' }\n#else\n')
                self.write("cc/src/example.cpp", prefix + header + ' { return unguarded(); }\n#endif\n')
                errors = self.errors()
                self.assertEqual(len(errors), 1)
                self.assertIn("must contain only", errors[0])
                self.write("cc/src/example.cpp", prefix + header + ' { ' + good + ' }\n#endif\n')
                self.assertEqual(self.errors(), [])

    def test_unsupported_platform_declarator_or_macro_fails_closed(self):
        good = ('#if GPU\nlean_object* lean_torch_example() { '
                'return tyr::checkedIoCall("x", [&] { return run(); }); }\n#else\n')
        for other in ['lean_object* lean_torch_example() CUSTOM_ATTRIBUTE { return run(); }',
                      'DEFINE_EXPORT(lean_torch_example)']:
            with self.subTest(other=other):
                self.write("cc/src/example.cpp", good + other + '\n#endif\n')
                self.assertTrue(any("unsupported" in error for error in self.errors()))

    def test_declaration_alone_does_not_count_as_implementation(self):
        self.write("cc/src/example.cpp", 'lean_object* lean_torch_example(b_lean_obj_arg x);')
        self.assertTrue(any("no recognized native definition" in error for error in self.errors()))

    def test_literals_comments_and_nested_blocks_do_not_confuse_braces(self):
        self.native('''
// braces and a fake return } return unguarded();
return tyr::checkedIoCall("x", [&] {
  const char* a = "} /* not a comment */";
  const char* b = R"tag({"quoted"})tag";
  const char c = '}';
  /* } { */ if (a) { return run(b, c); }
  return run();
});
''')
        self.assertEqual(self.errors(), [])

    def test_fixed_error_stub_exception_is_location_and_body_specific(self):
        symbol = "lean_torch_media_load_image_patchified"
        self.declare(symbol=symbol)
        stub = 'return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("unavailable")));'
        self.native(stub, symbol=symbol)
        self.assertTrue(self.errors())  # not a reviewed stub location
        (self.root / "cc/src/example.cpp").unlink()
        self.native(stub, "cc/src/tyr_media_stub.cpp", symbol)
        self.assertEqual(self.errors(), [])
        self.native('std::string context; ' + stub, "cc/src/tyr_media_stub.cpp", symbol)
        self.assertTrue(self.errors())

    def test_unterminated_function_fails_closed(self):
        self.write("cc/src/example.cpp", 'lean_object* lean_torch_example() {')
        self.assertTrue(any("unclosed delimiter" in error for error in self.errors()))


if __name__ == "__main__":
    unittest.main()
