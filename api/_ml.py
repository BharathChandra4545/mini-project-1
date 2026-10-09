from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURES = ["G1", "G2", "studytime", "absences", "failures", "age"]
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
DATA_PATH = ROOT / "data" / "cleaned_students.csv"


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model is missing from models/risk_model.pkl.")
    return joblib.load(MODEL_PATH)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}")
    return pd.read_csv(path, sep=None, engine="python")


def validate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise ValueError("CSV input must contain at least one data row.")
    frame = pd.DataFrame(rows)
    frame.columns = frame.columns.astype(str).str.strip()
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[FEATURES].isna().all(axis=1).any():
        raise ValueError("At least one row has no usable prediction feature values.")
    return frame


def predict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = validate_rows(rows)
    model = load_model()
    predictions = model.predict(frame[FEATURES])
    probabilities = model.predict_proba(frame[FEATURES])[:, 1]
    output = frame.copy()
    output["Predicted_Risk"] = predictions.astype(int)
    output["Risk_Probability"] = probabilities
    output["Risk_Status"] = output["Predicted_Risk"].map(
        {1: "Potential support needed", 0: "Not flagged by model"}
    )
    return output.where(pd.notna(output), None).to_dict(orient="records")
