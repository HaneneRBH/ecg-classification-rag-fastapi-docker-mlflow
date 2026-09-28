"""
Layer 4 — FastAPI service for the CardioSignal pipeline.

Exposes the ECG classification and grounded reporting pipeline over HTTP,
and keeps a history of predictions in a local SQLite database.

Endpoints
---------
    GET  /                  service info
    GET  /health            liveness + model status
    POST /predict           raw signal or ecg_id  ->  class probabilities
    POST /report            raw signal or ecg_id  ->  probabilities + grounded report
    POST /report/from-class  a class name          ->  grounded report only
    GET  /history           recent predictions from SQLite
    GET  /classes           the five super-classes and their rule cards

Run locally:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.

Design notes
------------
- The model is loaded ONCE at startup (lifespan) rather than per request,
  because loading a Keras checkpoint takes seconds.
- The rule base is loaded and validated at startup too: a malformed rule base
  should stop the service, not silently produce wrong reports.
- Every response carries the medical disclaimer.
- Data never leaves the machine: the model runs locally and the optional LLM
  runs in a local Ollama process.

DISCLAIMER: research and demonstration only, not for clinical diagnosis.
"""

import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data"))
sys.path.insert(0, os.path.join(_ROOT, "src", "report"))
sys.path.insert(0, os.path.join(_ROOT, "src", "pipeline"))

from rules import load_rules, validate, get_rule, SUPER_CLASSES  # noqa: E402
from explain import explain, explain_prediction  # noqa: E402

DB_PATH = os.environ.get("CARDIO_DB", os.path.join(_ROOT, "cardio_history.db"))
CKPT_PATH = os.environ.get(
    "CARDIO_CKPT", os.path.join(_ROOT, "checkpoints", "ecg_1dcnn_best.keras")
)
DATA_ROOT = os.environ.get("CARDIO_DATA_ROOT", "")

# Filled at startup.
STATE = {"model": None, "rules": None, "model_error": None}


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def init_db():
    """Create the history table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            record        TEXT,
            predicted     TEXT    NOT NULL,
            confidence    REAL,
            probabilities TEXT,
            report_source TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(record, predicted, confidence, probabilities, source):
    """Append one prediction to the history table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions "
        "(timestamp, record, predicted, confidence, probabilities, report_source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            record,
            predicted,
            confidence,
            json.dumps(probabilities),
            source,
        ),
    )
    conn.commit()
    conn.close()


def fetch_history(limit=20):
    """Return the most recent predictions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["probabilities"] = json.loads(d["probabilities"]) if d["probabilities"] else {}
        out.append(d)
    return out


# --------------------------------------------------------------------------
# Lifespan: load model + rules once
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Rule base is mandatory: refuse to start with an invalid one.
    rules = load_rules()
    ok, errors, _ = validate(rules, verbose=False)
    if not ok:
        raise RuntimeError(f"Invalid clinical rule base: {errors}")
    STATE["rules"] = rules

    # Model is optional at startup: the report endpoints still work without it.
    try:
        from predict_and_report import load_model
        STATE["model"] = load_model(CKPT_PATH)
    except Exception as exc:
        STATE["model_error"] = f"{type(exc).__name__}: {exc}"

    yield
    STATE["model"] = None


app = FastAPI(
    title="CardioSignal API",
    description=(
        "ECG classification (1D-CNN, TensorFlow) with grounded clinical reports. "
        "Research and demonstration only — not for clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SignalRequest(BaseModel):
    """Either a raw 12-lead signal, or an ecg_id to look up in PTB-XL."""
    signal: Optional[List[List[float]]] = Field(
        None,
        description="Raw ECG as a list of time steps, each with 12 lead values "
                    "(shape n_samples x 12). Typically 1000 x 12 at 100 Hz.",
    )
    ecg_id: Optional[int] = Field(
        None, description="ecg_id from ptbxl_database.csv (requires CARDIO_DATA_ROOT)."
    )
    record_path: Optional[str] = Field(
        None, description="Direct WFDB record path, without extension."
    )
    sampling_rate: int = Field(100, description="100 or 500 Hz.")
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                             description="Probability above which a class is reported.")
    use_llm: bool = Field(False,
                          description="Use the local Ollama LLM. Falls back to the "
                                      "deterministic template if unavailable.")


class ClassReportRequest(BaseModel):
    class_id: str = Field(..., description="One of NORM, MI, STTC, CD, HYP.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_llm: bool = Field(False)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _require_model():
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. {STATE['model_error'] or 'Train it first.'}",
        )
    return STATE["model"]


def _resolve_signal(req: SignalRequest):
    """Turn the request into a preprocessed (1, n, 12) array and a record label."""
    import numpy as np
    from predict_and_report import load_one_ecg, find_record_path
    from preprocess import preprocess

    if req.signal is not None:
        arr = np.asarray(req.signal, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 12:
            raise HTTPException(
                status_code=422,
                detail=f"signal must be (n_samples, 12); got {arr.shape}",
            )
        batch = arr[np.newaxis, ...]
        return preprocess(batch, fs=req.sampling_rate), "inline_signal", None

    if req.record_path is not None:
        return (load_one_ecg(req.record_path, fs=req.sampling_rate),
                os.path.basename(req.record_path), None)

    if req.ecg_id is not None:
        if not DATA_ROOT:
            raise HTTPException(
                status_code=503,
                detail="ecg_id lookup requires the CARDIO_DATA_ROOT environment variable.",
            )
        path, row = find_record_path(DATA_ROOT, req.ecg_id, sr=req.sampling_rate)
        meta = {
            "ecg_id": req.ecg_id,
            "age": None if row.get("age") is None else float(row["age"]),
            "sex": "male" if row.get("sex") == 1 else "female",
            "true_labels": [sc for sc in SUPER_CLASSES if row.get(sc) == 1],
        }
        return load_one_ecg(path, fs=req.sampling_rate), os.path.basename(path), meta

    raise HTTPException(
        status_code=422,
        detail="Provide one of: signal, ecg_id, or record_path.",
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "CardioSignal API",
        "version": "1.0.0",
        "docs": "/docs",
        "disclaimer": STATE["rules"]["_schema"]["disclaimer"] if STATE["rules"] else "",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_error": STATE["model_error"],
        "rules_loaded": STATE["rules"] is not None,
        "data_root_configured": bool(DATA_ROOT),
        "database": os.path.basename(DB_PATH),
    }


@app.get("/classes")
def classes():
    """Return the five super-classes with their clinical rule cards."""
    return {
        "classes": SUPER_CLASSES,
        "rules": STATE["rules"]["entries"],
        "disclaimer": STATE["rules"]["_schema"]["disclaimer"],
    }


@app.post("/predict")
def predict_endpoint(req: SignalRequest):
    """Classify an ECG and return per-class probabilities (no report)."""
    model = _require_model()
    signal, record, meta = _resolve_signal(req)

    preds = model.predict(signal, verbose=0)
    probs = {sc: round(float(p), 4) for sc, p in zip(SUPER_CLASSES, preds[0])}
    primary = max(probs, key=probs.get)

    save_prediction(record, primary, probs[primary], probs, "none")

    out = {
        "record": record,
        "predicted": primary,
        "confidence": probs[primary],
        "probabilities": probs,
        "above_threshold": [sc for sc, p in probs.items() if p >= req.threshold],
    }
    if meta:
        out["metadata"] = meta
    return out


@app.post("/report")
def report_endpoint(req: SignalRequest):
    """Classify an ECG and return the grounded clinical report."""
    model = _require_model()
    signal, record, meta = _resolve_signal(req)

    preds = model.predict(signal, verbose=0)
    probs = [float(p) for p in preds[0]]

    out = explain_prediction(
        probs, threshold=req.threshold,
        use_llm=req.use_llm, rules=STATE["rules"],
    )
    out["record"] = record
    if meta:
        out["metadata"] = meta

    save_prediction(record, out["class_id"], out.get("confidence"),
                    out["probabilities"], out["source"])
    return out


@app.post("/report/from-class")
def report_from_class(req: ClassReportRequest):
    """Generate a grounded report from a class name (no model needed)."""
    if req.class_id not in SUPER_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"class_id must be one of {SUPER_CLASSES}",
        )
    return explain(req.class_id, confidence=req.confidence,
                   use_llm=req.use_llm, rules=STATE["rules"])


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    """Return the most recent predictions stored in SQLite."""
    return {"count": limit, "predictions": fetch_history(limit)}
