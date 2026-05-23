"""Production-ready pipeline code templates for autonomous execution."""

from __future__ import annotations

from mlagent.pipeline.models import StageName


def get_stage_code(stage: StageName, *, target_column: str = "target", task_type: str = "classification") -> str:
    """Return executable Python for a pipeline stage."""
    generators = {
        StageName.DATA_UNDERSTANDING: _eda_code,
        StageName.DATA_PREPARATION: _prep_code,
        StageName.MODELING: _model_code,
        StageName.EVALUATION: _eval_code,
    }
    return generators[stage](target_column=target_column, task_type=task_type)


def _eda_code(target_column: str, task_type: str) -> str:
    return f'''"""EDA stage — auto-generated."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = Path(__import__("os").environ["MLAGENT_WORKSPACE"])
DATA = WORKSPACE / "data"
raw = DATA / "raw.csv"
df = pd.read_csv(raw)
target = "{target_column}"

summary = {{
    "rows": len(df),
    "columns": list(df.columns),
    "dtypes": {{c: str(t) for c, t in df.dtypes.items()}},
    "missing": df.isnull().sum().to_dict(),
    "target": target,
    "target_distribution": df[target].value_counts().to_dict() if target in df.columns else {{}},
    "task_type": "{task_type}",
}}
(DATA / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("MLAGENT_ARTIFACT:eda_summary.json")

# Basic plots
fig, ax = plt.subplots(figsize=(6, 4))
if target in df.columns and df[target].dtype in ("int64", "float64", "int32"):
    df[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Target distribution")
    fig.savefig(DATA / "target_distribution.png", bbox_inches="tight")
    plt.close()
print("MLAGENT_METRIC:eda_columns=" + str(len(df.columns)))
print("EDA complete.")
'''


def _prep_code(target_column: str, task_type: str) -> str:
    return f'''"""Data preparation — auto-generated."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

WORKSPACE = Path(__import__("os").environ["MLAGENT_WORKSPACE"])
DATA = WORKSPACE / "data"
df = pd.read_csv(DATA / "raw.csv")
target = "{target_column}"

X = df.drop(columns=[target])
y = df[target]
# Encode categoricals
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if "{task_type}" == "classification" and y.nunique() < 20 else None,
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pd.DataFrame(X_train_s, columns=X.columns).to_csv(DATA / "X_train.csv", index=False)
pd.DataFrame(X_test_s, columns=X.columns).to_csv(DATA / "X_test.csv", index=False)
pd.DataFrame(y_train).to_csv(DATA / "y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(DATA / "y_test.csv", index=False)
joblib.dump(scaler, DATA / "scaler.pkl")
print("MLAGENT_ARTIFACT:X_train.csv")
print("MLAGENT_ARTIFACT:X_test.csv")
print("MLAGENT_METRIC:train_rows=" + str(len(X_train)))
print("Preparation complete.")
'''


def _model_code(target_column: str, task_type: str) -> str:
    is_regression = task_type == "regression"
    model_import = (
        "from sklearn.ensemble import GradientBoostingRegressor as Model"
        if is_regression
        else "from sklearn.ensemble import RandomForestClassifier as Model"
    )
    metric_line = (
        'print("MLAGENT_METRIC:train_r2=" + str(score))'
        if is_regression
        else 'print("MLAGENT_METRIC:train_accuracy=" + str(score))'
    )
    score_line = (
        "score = model.score(X_train, y_train)"
    )
    return f'''"""Modeling stage — auto-generated."""
from pathlib import Path
import pandas as pd
import joblib
{model_import}

WORKSPACE = Path(__import__("os").environ["MLAGENT_WORKSPACE"])
DATA = WORKSPACE / "data"
MODELS = WORKSPACE / "models"
MODELS.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA / "X_train.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
model = Model(random_state=42) if "{task_type}" == "regression" else Model(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
{score_line}
joblib.dump(model, MODELS / "model.pkl")
{metric_line}
print("MLAGENT_ARTIFACT:model.pkl")
print("Model training complete.")
'''


def _eval_code(target_column: str, task_type: str) -> str:
    is_regression = task_type == "regression"
    metrics_block = (
        """
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
pred = model.predict(X_test)
report = {
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "mae": float(mean_absolute_error(y_test, pred)),
    "r2": float(r2_score(y_test, pred)),
}
print("MLAGENT_METRIC:r2=" + str(report["r2"]))
print("MLAGENT_METRIC:rmse=" + str(report["rmse"]))
"""
        if is_regression
        else """
from sklearn.metrics import accuracy_score, f1_score, classification_report
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
report = {"accuracy": float(acc), "f1_weighted": float(f1)}
print("MLAGENT_METRIC:accuracy=" + str(acc))
print("MLAGENT_METRIC:f1_weighted=" + str(f1))
"""
    )
    return f'''"""Evaluation stage — auto-generated."""
import json
from pathlib import Path
import pandas as pd
import joblib

WORKSPACE = Path(__import__("os").environ["MLAGENT_WORKSPACE"])
DATA = WORKSPACE / "data"
model = joblib.load(WORKSPACE / "models" / "model.pkl")
X_test = pd.read_csv(DATA / "X_test.csv")
y_test = pd.read_csv(DATA / "y_test.csv").squeeze()
{metrics_block}
(DATA / "evaluation_report.json").write_text(json.dumps(report, indent=2))
(WORKSPACE / "metrics.json").write_text(json.dumps(report, indent=2))
print("MLAGENT_ARTIFACT:evaluation_report.json")
print("Evaluation complete.")
'''
