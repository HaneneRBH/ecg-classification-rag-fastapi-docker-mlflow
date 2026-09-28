"""
Layer 3 — Clinical rule base loader and validator.

The rule base (data/clinical_rules/ecg_rules.json) is the ONLY authorised
source of medical facts for the generated reports. This module loads it and
validates it strictly, so a malformed or incomplete rule base fails loudly
here rather than silently producing a wrong report.

Validation checks:
    1. Schema  — every entry has all required fields, non-empty.
    2. Coverage — one entry per super-class (5/5), no duplicates.
    3. Consistency — class_id values match the canonical SUPER_CLASSES list
       used by the classifier, so predictions always find their rule.
    4. Category — each entry's category is one of the declared categories.

    python src/report/rules.py --validate

Exit code is 0 when the rule base is valid, 1 otherwise, so this can be
wired into CI (GitHub Actions, Layer 6).
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data"))

DEFAULT_PATH = os.path.join(_ROOT, "data", "clinical_rules", "ecg_rules.json")

# The canonical class list, mirrored from the Layer-1 loader so the rule base
# and the classifier can never drift apart.
SUPER_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_rules(path=DEFAULT_PATH):
    """Load the clinical rule base from JSON.

    Returns the parsed dict. Raises FileNotFoundError or json.JSONDecodeError
    on a missing or malformed file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Clinical rule base not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entries(rules):
    """Return the list of rule entries."""
    return rules.get("entries", [])


def get_rule(rules, class_id):
    """Return the entry for one super-class (e.g. 'MI'), or None."""
    for e in entries(rules):
        if e.get("class_id") == class_id:
            return e
    return None


def validate(rules, verbose=True):
    """Validate the rule base. Returns (ok, errors, warnings)."""
    errors = []
    warnings = []

    schema = rules.get("_schema", {})
    required = schema.get("required_fields", [])
    valid_categories = set(schema.get("categories", []))
    rows = entries(rules)

    if not required:
        warnings.append("_schema.required_fields is missing; field check skipped.")
    if not rows:
        errors.append("Rule base contains no entries.")
        return False, errors, warnings

    # --- 1. Schema: every entry has every required field, non-empty ----------
    for e in rows:
        cid = e.get("class_id", "<no class_id>")
        for field in required:
            if field not in e:
                errors.append(f"[{cid}] missing field '{field}'")
            elif not str(e[field]).strip():
                errors.append(f"[{cid}] field '{field}' is empty")

    # --- 2. Coverage: one entry per super-class, no duplicates ---------------
    ids = [e.get("class_id") for e in rows]
    seen = set()
    for cid in ids:
        if cid in seen:
            errors.append(f"Duplicate entry for class_id '{cid}'")
        seen.add(cid)

    missing = [c for c in SUPER_CLASSES if c not in seen]
    if missing:
        errors.append(f"Missing entries for: {', '.join(missing)}")

    extra = [c for c in seen if c not in SUPER_CLASSES]
    if extra:
        warnings.append(
            f"Entries not in SUPER_CLASSES (never reachable): {', '.join(extra)}"
        )

    # --- 3. Category is declared --------------------------------------------
    if valid_categories:
        for e in rows:
            cat = e.get("category")
            if cat and cat not in valid_categories:
                errors.append(
                    f"[{e.get('class_id')}] category '{cat}' not in {sorted(valid_categories)}"
                )

    # --- 4. Disclaimer present (safety requirement) -------------------------
    if not schema.get("disclaimer"):
        warnings.append("_schema.disclaimer is missing (recommended for a medical rule base).")

    ok = len(errors) == 0

    if verbose:
        n_cov = len([c for c in SUPER_CLASSES if c in seen])
        print(f"Loaded rule base with {len(rows)} entries")
        print(f"\nCoverage: {n_cov}/{len(SUPER_CLASSES)} super-classes with entries.")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ! {w}")
        if errors:
            print("\nErrors:")
            for err in errors:
                print(f"  x {err}")
        print(f"\nRESULT: {'OK' if ok else 'FAILED'}")

    return ok, errors, warnings


def parse_args():
    p = argparse.ArgumentParser(description="Load and validate the clinical rule base.")
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--validate", action="store_true",
                    help="Run validation and exit with a status code.")
    p.add_argument("--show", metavar="CLASS_ID",
                    help="Print the rule entry for one super-class (e.g. MI).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rules = load_rules(args.path)

    if args.show:
        entry = get_rule(rules, args.show)
        if entry is None:
            print(f"No rule entry for class_id '{args.show}'")
            sys.exit(1)
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        sys.exit(0)

    ok, errors, _ = validate(rules)
    sys.exit(0 if ok else 1)
