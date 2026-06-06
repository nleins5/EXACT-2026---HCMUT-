"""Unit tests for z3_output_parser — all 6 parsing strategies + edge cases.

These tests are critical for competition scoring: the parser determines
the final prediction (True/False/Unknown), so any parsing bug directly
reduces accuracy.
"""
import pytest
from src.utils.z3_output_parser import parse_z3_output


# ── Strategy 1: Canonical "Predicted: X" ────────────────────────────

class TestCanonicalPredicted:
    def test_predicted_true(self):
        assert parse_z3_output("Predicted: True") == "True"

    def test_predicted_false(self):
        assert parse_z3_output("Predicted: False") == "False"

    def test_predicted_unknown(self):
        assert parse_z3_output("Predicted: Unknown") == "Unknown"

    def test_predicted_case_insensitive(self):
        assert parse_z3_output("predicted: true") == "True"
        assert parse_z3_output("PREDICTED: FALSE") == "False"

    def test_predicted_with_surrounding_text(self):
        output = "Some debug info\nPredicted: True\nMore stuff"
        assert parse_z3_output(output) == "True"

    def test_predicted_with_whitespace(self):
        assert parse_z3_output("  Predicted:   True  ") == "True"


# ── Strategy 2: "Expected: X, Predicted: Y" → use Predicted ────────

class TestExpectedPredicted:
    def test_expected_predicted_match(self):
        assert parse_z3_output("Expected: True, Predicted: True") == "True"

    def test_expected_predicted_mismatch(self):
        assert parse_z3_output("Expected: True, Predicted: False") == "False"

    def test_expected_predicted_unknown(self):
        assert parse_z3_output("Expected: False, Predicted: Unknown") == "Unknown"


# ── Strategy 3: "Expected: X" alone (model shortcut) ───────────────

class TestExpectedAlone:
    def test_expected_true(self):
        assert parse_z3_output("Expected: True") == "True"

    def test_expected_false(self):
        assert parse_z3_output("Expected: False") == "False"

    def test_expected_unknown(self):
        assert parse_z3_output("Expected: Unknown") == "Unknown"


# ── Strategy 4: Multi-conclusion "Conclusion N: [Entailed/...]" ─────

class TestMultiConclusion:
    def test_all_entailed(self):
        output = "Conclusion 1: [Entailed]\nConclusion 2: [Entailed]"
        assert parse_z3_output(output) == "True"

    def test_all_not_entailed(self):
        output = "Conclusion 1: [NotEntailed]\nConclusion 2: [NotEntailed]"
        assert parse_z3_output(output) == "Unknown"

    def test_mixed_conclusions(self):
        output = "Conclusion 1: [Entailed]\nConclusion 2: [NotEntailed]"
        assert parse_z3_output(output) == "Unknown"

    def test_conclusion_true_false(self):
        output = "Conclusion 1: [True]\nConclusion 2: [True]"
        assert parse_z3_output(output) == "True"

    def test_single_conclusion_entailed(self):
        output = "Conclusion 1: [Entailed]"
        assert parse_z3_output(output) == "True"

    def test_conclusion_without_brackets(self):
        output = "Conclusion 1: Entailed\nConclusion 2: Entailed"
        assert parse_z3_output(output) == "True"


# ── Strategy 5: "Conclusion N is entailed/not entailed" ─────────────

class TestConclusionIsEntailed:
    def test_all_entailed(self):
        output = "Conclusion 1 is entailed\nConclusion 2 is entailed"
        assert parse_z3_output(output) == "True"

    def test_all_not_entailed(self):
        output = "Conclusion 1 is not entailed\nConclusion 2 is not entailed"
        assert parse_z3_output(output) == "Unknown"

    def test_mixed(self):
        output = "Conclusion 1 is entailed\nConclusion 2 is not entailed"
        assert parse_z3_output(output) == "Unknown"


# ── Strategy 6: Bare True/False/Unknown on its own line ─────────────

class TestBarePrediction:
    def test_bare_true(self):
        assert parse_z3_output("True") == "True"

    def test_bare_false(self):
        assert parse_z3_output("False") == "False"

    def test_bare_unknown(self):
        assert parse_z3_output("Unknown") == "Unknown"

    def test_bare_with_newlines(self):
        assert parse_z3_output("\nTrue\n") == "True"

    def test_bare_case_insensitive(self):
        assert parse_z3_output("true") == "True"
        assert parse_z3_output("FALSE") == "False"


# ── Fallback + Edge Cases ───────────────────────────────────────────

class TestFallbackAndEdgeCases:
    def test_empty_string(self):
        assert parse_z3_output("") == "Unknown"

    def test_none_like_empty(self):
        assert parse_z3_output("   ") == "Unknown"

    def test_gibberish(self):
        assert parse_z3_output("sat\nunsat\nmodel available") == "Unknown"

    def test_z3_raw_sat_output(self):
        """Z3 may output just 'sat'/'unsat' — should fallback to Unknown."""
        assert parse_z3_output("sat") == "Unknown"
        assert parse_z3_output("unsat") == "Unknown"

    def test_multiline_with_predicted(self):
        """Predicted: X should take priority even with other noise."""
        output = "sat\nmodel:\n  x = 5\nPredicted: False"
        assert parse_z3_output(output) == "False"

    def test_priority_predicted_over_expected(self):
        """Strategy 1 (Predicted:) takes priority over Strategy 3 (Expected:)."""
        output = "Expected: True\nPredicted: False"
        assert parse_z3_output(output) == "False"

    def test_priority_predicted_over_bare(self):
        """Predicted: X takes priority over bare True/False."""
        output = "True\nPredicted: False"
        assert parse_z3_output(output) == "False"
