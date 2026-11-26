"""Regression checks for release protection triage using small, synthetic modules."""

import textwrap
import json
import unittest

from protection_diff import build_public_symbol_index, compare_module


def code(text):
    return (textwrap.dedent(text).strip() + "\n").encode()


def changes(old, new, **kwargs):
    return compare_module("rlinf/observer.py", code(new), code(old) if old is not None else None,
                          **kwargs)


class ProtectionDiffTest(unittest.TestCase):
    def test_changes_to_docs_and_comments_do_not_raise_logic_priority(self):
        before = '"""Guide A."""\n\ndef f(a):\n    """Example A."""\n    return a\n'
        after = before.replace(' A.', ' B.').replace('return a', 'return a  # formatting')
        result = changes(before, after)
        self.assertEqual(result["status"], "text_only")
        self.assertEqual(result["priority"], "low")
        self.assertEqual({s["qualified_name"] for s in result["symbols"]}, {"<module>", "f"})
        self.assertNotIn("Example", json.dumps(result))

    def test_constant_change_has_a_module_scope_and_keeps_types(self):
        r = changes("DEFAULT = 1", "DEFAULT = True")
        self.assertEqual((r["status"], r["priority"]), ("modified", "review"))
        self.assertEqual(r["symbols"][0]["qualified_name"], "<module>")

    def test_stateful_decision_inside_a_method_is_high_without_filename_keywords(self):
        old = """
            class History:
                def take(self, sample):
                    return sample
        """
        new = """
            class History:
                def take(self, sample):
                    if self.pending and sample.weight > 0:
                        self.pending.append(sample)
                    return sample
        """
        result = changes(old, new)
        self.assertEqual(result["priority"], "high")
        self.assertEqual([s["qualified_name"] for s in result["symbols"]], ["History.take"])
        self.assertEqual(result["symbols"][0]["current_line"], 2)

    def test_nested_helpers_stay_with_the_function(self):
        old = "def outer(x):\n    def normalize(v):\n        return v + 1\n    return normalize(x)"
        result = changes(old, old.replace("v + 1", "v * 2"))
        self.assertEqual([s["qualified_name"] for s in result["symbols"]], ["outer"])

    def test_class_header_decorator_and_signature_changes(self):
        old = "@register\nclass Driver(B):\n    @scalar\n    def f(self, count=1):\n        return count"
        new = old.replace("Driver(B)", "Driver(A)").replace("@scalar", "@batch").replace("count=1", "count=5")
        result = changes(old, new)
        self.assertEqual({s["qualified_name"]: s["changes"] for s in result["symbols"]},
                         {"Driver": ["declaration"], "Driver.f": ["declaration"]})

    def test_conditional_definition_and_property_names_do_not_collide(self):
        before = """
            if enabled:
                class Data:
                    @property
                    def value(self): return self._value
                    @value.setter
                    def value(self, v): self._value = v
        """
        result = changes(before, before.replace(" = v", " = 3 * v"))
        self.assertEqual([s["qualified_name"] for s in result["symbols"]], ["Data.value [setter]"])

    def test_removal_reports_no_remaining_symbol_to_encrypt(self):
        before = "def gone():\n    return True\n\nclass A:\n    pass"
        result = changes(before, "class A:\n    pass")
        row, = result["symbols"]
        self.assertEqual(result["priority"], "none")
        self.assertEqual((row["qualified_name"], row["status"], row["current_line"]),
                         ("gone", "deleted", None))

    def test_reordered_declaration_relative_to_eager_call_is_semantic(self):
        result = changes("def f(): return 1\nx = f()", "x = f()\ndef f(): return 1")
        self.assertEqual(result["status"], "modified")
        self.assertEqual(result["symbols"][0]["qualified_name"], "<module>")

    def test_invalid_encoding_or_syntax_reports_manual_review_without_the_source_line(self):
        for malformed in (b"key-name = 44", b"# coding: unheardof\n", b"def f():\n\x00"):
            result = compare_module("rlinf/lib.py", malformed, b"")
            self.assertEqual(result["priority"], "review")
            self.assertIn("analysis_error", result)
            self.assertNotIn("key-name", json.dumps(result))

    def test_cross_path_public_reuse_cannot_hide_fresh_state_logic(self):
        block = """
            def take(state, value):
                if state.history and value > 0:
                    state.history.append(value)
                return value
        """
        oracle = build_public_symbol_index({"rlinf/old.py": code(block)})
        reused = changes(None, block, public_symbols=oracle)
        self.assertEqual(reused["priority"], "review")
        row = next(r for r in reused["symbols"] if r["qualified_name"] == "take")
        self.assertEqual(row["public_matches"], [
            {"file": "rlinf/old.py", "qualified_name": "take", "line": 1}
        ])
        self.assertEqual(row["priority"], "low")
        altered = changes(None, block.replace("value > 0", "value <= 0"), public_symbols=oracle)
        self.assertEqual(altered["priority"], "high")

    def test_tiny_data_projection_is_low_even_in_a_training_filename(self):
        result = compare_module("rlinf/algorithms/loss.py", code("def train(x): return x[:3]"), None)
        self.assertEqual(result["priority"], "low")


if __name__ == "__main__":
    unittest.main()
