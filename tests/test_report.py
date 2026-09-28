"""
Unit tests for the report layer (Layer 3).

These tests deliberately cover only what runs WITHOUT the 3 GB dataset and
without a trained checkpoint, because neither is in the repository. That is
what CI can actually verify on a fresh clone:

    - the clinical rule base is valid and complete;
    - every super-class has a reachable rule;
    - the deterministic template never invents content;
    - the multi-label logic picks the right primary finding.

    pytest tests/ -v
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "report"))

from rules import load_rules, validate, get_rule, SUPER_CLASSES  # noqa: E402
from explain import explain, explain_prediction, _template, _card  # noqa: E402


# --- Rule base ------------------------------------------------------------

def test_rule_base_loads():
    rules = load_rules()
    assert "entries" in rules
    assert "_schema" in rules


def test_rule_base_is_valid():
    """The validator must pass: this is what CI guards against."""
    rules = load_rules()
    ok, errors, _ = validate(rules, verbose=False)
    assert ok, f"Rule base invalid: {errors}"


def test_all_classes_have_a_rule():
    """Every class the classifier can predict must have a rule to explain it."""
    rules = load_rules()
    for class_id in SUPER_CLASSES:
        entry = get_rule(rules, class_id)
        assert entry is not None, f"No rule for {class_id}"


def test_rules_have_required_fields():
    rules = load_rules()
    required = rules["_schema"]["required_fields"]
    for entry in rules["entries"]:
        for field in required:
            assert field in entry, f"{entry.get('class_id')} missing {field}"
            assert str(entry[field]).strip(), f"{entry.get('class_id')}.{field} is empty"


def test_disclaimer_present():
    """A medical rule base without a disclaimer is a safety problem."""
    rules = load_rules()
    assert rules["_schema"].get("disclaimer")


# --- Grounded explanation -------------------------------------------------

@pytest.mark.parametrize("class_id", SUPER_CLASSES)
def test_explain_returns_report(class_id):
    out = explain(class_id, use_llm=False)
    assert out["found"] is True
    assert out["class_id"] == class_id
    assert out["source"] == "template"
    assert len(out["report"]) > 50
    assert out["disclaimer"]


def test_explain_unknown_class():
    out = explain("NOT_A_CLASS", use_llm=False)
    assert out["found"] is False


def test_template_is_grounded():
    """The template must only reuse text from the rule entry: nothing new."""
    rules = load_rules()
    entry = get_rule(rules, "MI")
    text = _template(entry)
    # Each rule field must appear in the generated report.
    assert entry["description"] in text
    assert entry["ecg_signs"] in text
    assert entry["action"] in text


def test_confidence_appears_in_report():
    out = explain("MI", confidence=0.87, use_llm=False)
    assert "87%" in out["report"]
    assert out["confidence"] == 0.87


def test_card_contains_only_rule_facts():
    """The prompt card is the LLM's only source: check it is built from the rule."""
    rules = load_rules()
    entry = get_rule(rules, "STTC")
    card = _card(entry, confidence=0.6)
    assert entry["full_name"] in card
    assert entry["urgency"] in card
    assert "60%" in card


# --- Multi-label logic ----------------------------------------------------

def test_prediction_picks_highest_probability():
    # NORM, MI, STTC, CD, HYP
    probs = [0.12, 0.89, 0.64, 0.21, 0.08]
    out = explain_prediction(probs, threshold=0.5, use_llm=False)
    assert out["class_id"] == "MI"
    assert out["confidence"] == pytest.approx(0.89, abs=1e-3)


def test_co_findings_respect_threshold():
    probs = [0.05, 0.90, 0.64, 0.30, 0.02]

    # At 0.5, only STTC (0.64) is a co-finding.
    high = explain_prediction(probs, threshold=0.5, use_llm=False)
    assert "ST/T change" in high["report"]
    assert "Conduction disturbance" not in high["report"]

    # At 0.25, CD (0.30) joins in.
    low = explain_prediction(probs, threshold=0.25, use_llm=False)
    assert "Conduction disturbance" in low["report"]


def test_probabilities_are_returned():
    probs = [0.93, 0.05, 0.11, 0.08, 0.04]
    out = explain_prediction(probs, use_llm=False)
    assert set(out["probabilities"].keys()) == set(SUPER_CLASSES)
    assert out["class_id"] == "NORM"


def test_template_is_deterministic():
    """Same input, same output: required for reproducible clinical reports."""
    probs = [0.1, 0.8, 0.2, 0.1, 0.1]
    a = explain_prediction(probs, use_llm=False)
    b = explain_prediction(probs, use_llm=False)
    assert a["report"] == b["report"]
