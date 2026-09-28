"""
Layer 3 — Grounded clinical report generation.

Turns a predicted ECG class into a natural-language clinical report that is
*grounded* in the verified rule base: every fact comes from the rule card,
never from the language model's own memory.

Two layers guarantee this:

  1. The prompt hands the LLM ONE rule card and forbids adding anything that
     is not written in it. Temperature is kept low (0.1) so the model stays
     close to the source rather than paraphrasing creatively — in a clinical
     setting, consistency matters more than variety.

  2. If no local LLM is reachable (Ollama not installed or not running), we
     fall back to a deterministic template assembled directly from the rule
     fields. That fallback cannot hallucinate at all: it only reorders text
     that a human wrote and validated.

The report is also enriched with *case-specific* facts the model actually
computed — the predicted class, its confidence, and any co-predicted classes.
That is what legitimately differentiates two reports, rather than the LLM
rewording the same card differently.

    python src/report/explain.py --class-id MI --no-llm
    python src/report/explain.py --class-id STTC --confidence 0.87
    python src/report/explain.py --class-id NORM        # uses Ollama if running

Ollama is optional and local (https://ollama.com). Default model 'llama3.2'.
Everything is CPU-friendly; the LLM runs in Ollama's own process, so no
patient data ever leaves the machine.

DISCLAIMER: research and demonstration only, not for clinical diagnosis.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from rules import load_rules, get_rule, SUPER_CLASSES  # noqa: E402

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("CARDIO_LLM_MODEL", "llama3.2")

SYSTEM = (
    "You are a careful clinical decision-support assistant writing a short ECG "
    "report for a clinician. In 2-3 sentences, summarise the finding in clear, "
    "professional language. Use ONLY the facts in the rule card below. Do not "
    "invent any measurement, threshold, percentage, drug, diagnosis or "
    "recommendation that is not written in the card. If the card does not state "
    "something, do not mention it. Never claim certainty the card does not support."
)


def _card(entry, confidence=None, also_predicted=None):
    """Build the rule card — the only ground truth the LLM may use.

    Case-specific facts (confidence, co-predictions) are added here because
    they are computed by the classifier, not invented by the model.
    """
    lines = [
        f"Finding: {entry['full_name']} ({entry['class_id']})",
        f"Category: {entry['category']}",
        f"Description: {entry['description']}",
        f"Typical ECG signs: {entry['ecg_signs']}",
        f"Urgency: {entry['urgency']}",
        f"Recommended action: {entry['action']}",
    ]
    if confidence is not None:
        lines.append(f"Model confidence for this finding: {confidence:.0%}")
    if also_predicted:
        lines.append(
            "Other findings flagged on the same recording: "
            + ", ".join(also_predicted)
        )
    return "\n".join(lines)


def _prompt(card):
    """Assemble the final prompt: fixed instructions + the variable card."""
    return (
        f"{SYSTEM}\n\n"
        "--- RULE CARD (the only authorised source) ---\n"
        f"{card}\n"
        "--- END RULE CARD ---\n\n"
        "Write the report now."
    )


def _call_ollama(prompt, model, host, timeout=90.0):
    """Return the LLM text, or None if Ollama is unreachable or errors out."""
    try:
        import requests
        resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # Low temperature: in a clinical context we want the wording to
                # stay close to the card and be reproducible across runs.
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None


def _template(entry, confidence=None, also_predicted=None):
    """Deterministic, hallucination-proof report built from rule fields only."""
    parts = [f"{entry['full_name']}."]
    if confidence is not None:
        parts.append(f"Detected with {confidence:.0%} model confidence.")
    parts.append(entry["description"])
    parts.append(f"Typical ECG signs: {entry['ecg_signs']}")
    parts.append(f"Urgency: {entry['urgency']}")
    parts.append(f"Recommended action: {entry['action']}")
    if also_predicted:
        parts.append(
            "Other findings flagged on the same recording: "
            + ", ".join(also_predicted) + "."
        )
    return " ".join(parts)


# --- public API ------------------------------------------------------------

def explain(class_id, confidence=None, also_predicted=None, use_llm=True,
            model=DEFAULT_LLM, host=OLLAMA_HOST, rules=None):
    """Generate a grounded clinical report for one predicted class.

    Parameters
    ----------
    class_id : str
        One of SUPER_CLASSES ('NORM', 'MI', 'STTC', 'CD', 'HYP').
    confidence : float or None
        Model confidence (0-1) for this finding, if available.
    also_predicted : list[str] or None
        Other class names flagged on the same ECG (multi-label case).
    use_llm : bool
        If False, skip Ollama entirely and use the deterministic template.

    Returns
    -------
    dict with keys: class_id, found, name, category, source, report, disclaimer.
    """
    r = rules if rules is not None else load_rules()
    entry = get_rule(r, class_id)

    if entry is None:
        return {
            "class_id": class_id,
            "found": False,
            "report": f"No clinical rule entry for class '{class_id}'.",
        }

    card = _card(entry, confidence=confidence, also_predicted=also_predicted)

    text, source = None, "template"
    if use_llm:
        text = _call_ollama(_prompt(card), model, host)
        source = "llm" if text else "template"
    if text is None:
        text = _template(entry, confidence=confidence, also_predicted=also_predicted)

    disclaimer = r.get("_schema", {}).get("disclaimer", "")

    return {
        "class_id": class_id,
        "found": True,
        "name": entry["name"],
        "category": entry["category"],
        "confidence": round(confidence, 4) if confidence is not None else None,
        "source": source,
        "report": text,
        "disclaimer": disclaimer,
    }


def explain_prediction(probabilities, threshold=0.5, use_llm=True,
                       model=DEFAULT_LLM, host=OLLAMA_HOST, rules=None):
    """Generate a report from a full model output vector (multi-label).

    Parameters
    ----------
    probabilities : sequence of float, length 5
        Sigmoid outputs from the classifier, in SUPER_CLASSES order.
    threshold : float
        Probability above which a class is considered present.

    Returns
    -------
    dict with the primary finding's report plus the full probability table.
    """
    probs = {sc: float(p) for sc, p in zip(SUPER_CLASSES, probabilities)}
    # Primary finding = highest probability.
    primary = max(probs, key=probs.get)
    # Co-findings = other classes above threshold.
    also = [sc for sc, p in probs.items() if p >= threshold and sc != primary]

    r = rules if rules is not None else load_rules()
    also_names = []
    for sc in also:
        e = get_rule(r, sc)
        also_names.append(e["name"] if e else sc)

    out = explain(primary, confidence=probs[primary],
                  also_predicted=also_names or None,
                  use_llm=use_llm, model=model, host=host, rules=r)
    out["probabilities"] = {k: round(v, 4) for k, v in probs.items()}
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Grounded clinical report for an ECG finding.")
    p.add_argument("--class-id", choices=SUPER_CLASSES,
                    help="Predicted super-class.")
    p.add_argument("--confidence", type=float, default=None,
                    help="Model confidence 0-1 for this finding.")
    p.add_argument("--probs", nargs=5, type=float, default=None,
                    metavar=("NORM", "MI", "STTC", "CD", "HYP"),
                    help="Full probability vector (5 values) instead of --class-id.")
    p.add_argument("--no-llm", action="store_true",
                    help="Skip Ollama, use the deterministic template only.")
    p.add_argument("--model", default=DEFAULT_LLM)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_llm = not args.no_llm

    if args.probs is not None:
        out = explain_prediction(args.probs, use_llm=use_llm, model=args.model)
    elif args.class_id is not None:
        out = explain(args.class_id, confidence=args.confidence,
                      use_llm=use_llm, model=args.model)
    else:
        raise SystemExit("Give --class-id CLASS or --probs p1 p2 p3 p4 p5")

    print(json.dumps(out, indent=2, ensure_ascii=False))
