"""
Injury prediction model.

Trains a multi-output Random Forest to predict each injury severity threshold
(1 day missed, 7 days, 28 days, 42+ days) from play-context features.

On AWS this would be packaged as a SageMaker training job:
  - Input:  s3://nfl-data-lake/processed/injury_features/
  - Output: s3://nfl-models/injury-prediction/model.tar.gz
  - Inference: SageMaker real-time endpoint consumed by the API Gateway
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingestion.data_loader import load_injury_records, load_play_list
from src.processing.feature_engineering import build_injury_feature_set, get_feature_columns

TARGETS = ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]
TARGET_LABELS = {
    "DM_M1":  "≥1 day missed",
    "DM_M7":  "≥7 days missed",
    "DM_M28": "≥28 days missed",
    "DM_M42": "≥42 days missed",
}
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models"


def build_model() -> MultiOutputClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return MultiOutputClassifier(rf)


def train(df: pd.DataFrame, feature_cols: list[str]) -> tuple[MultiOutputClassifier, list[str]]:
    X = df[feature_cols].fillna(0).values
    Y = df[TARGETS].values

    model = build_model()
    model.fit(X, Y)
    return model, feature_cols


def evaluate(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df[feature_cols].fillna(0).values
    Y = df[TARGETS].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for i, target in enumerate(TARGETS):
        y = Y[:, i]
        unique = np.unique(y)
        if len(unique) < 2:
            results[target] = {"note": f"only one class ({unique[0]}) — cannot evaluate"}
            continue
        if y.sum() < 5:
            results[target] = {"note": "insufficient positive samples for CV"}
            continue

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        # guard: if only one class seen in all folds, proba matrix has one column
        y_prob = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y, y_prob)
        except ValueError:
            auc = None

        results[target] = {
            "label": TARGET_LABELS[target],
            "auc_roc": round(auc, 4) if auc else None,
            "report": classification_report(y, y_pred, output_dict=True),
        }

    return results


def feature_importance_summary(model: MultiOutputClassifier, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target, estimator in zip(TARGETS, model.estimators_):
        imp = estimator.feature_importances_
        for feature, score in zip(feature_cols, imp):
            rows.append({"target": target, "feature": feature, "importance": round(score, 4)})
    df = pd.DataFrame(rows)
    return df.sort_values(["target", "importance"], ascending=[True, False])


def save_model_metadata(feature_cols: list[str], eval_results: dict) -> None:
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    meta = {
        "features": feature_cols,
        "targets": TARGETS,
        "evaluation": {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in eval_results.items()
        },
    }
    with open(MODEL_OUTPUT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {MODEL_OUTPUT_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    plays = load_play_list()
    injuries = load_injury_records()

    print(f"Loaded {len(injuries)} injury records, {len(plays):,} plays")

    feature_df = build_injury_feature_set(plays, injuries)
    feature_cols = get_feature_columns(feature_df)

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"\nTarget distribution:")
    for t in TARGETS:
        n = feature_df[t].sum()
        print(f"  {TARGET_LABELS[t]:25s}: {n}/{len(feature_df)} ({100*n/len(feature_df):.0f}%)")

    print("\nRunning 5-fold cross-validation...")
    eval_results = evaluate(feature_df, feature_cols)

    for target, res in eval_results.items():
        label = res.get("label", target)
        auc = res.get("auc_roc", "N/A")
        print(f"\n  {label}")
        print(f"    AUC-ROC: {auc}")
        if "report" in res:
            rep = res["report"]
            print(f"    F1 (weighted): {rep.get('weighted avg', {}).get('f1-score', 'N/A'):.3f}")

    model, cols = train(feature_df, feature_cols)
    print("\nFinal model trained on full dataset.")

    importance = feature_importance_summary(model, cols)
    print("\nTop 5 features for DM_M1 (any injury):")
    print(importance[importance["target"] == "DM_M1"].head(5).to_string(index=False))

    save_model_metadata(cols, eval_results)
