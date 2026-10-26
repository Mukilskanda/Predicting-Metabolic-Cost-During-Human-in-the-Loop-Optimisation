"""Command-line inference with a saved model.

    python scripts/predict.py --csv data/hilo_metabolic_dataset.csv --subject S1
    python scripts/predict.py --demo --subject S2          # leave-one-day-out demo in the terminal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metacost.data import FEATURE_SETS, load_subject
from metacost.evaluation import metrics
from metacost.models import make_model


def demo(subject, model_name):
    d = load_subject(subject)
    X, y = d.features(FEATURE_SETS["All features"]), d.y
    print(f"\nSubject {subject[-1]} | {model_name} | train on 4 days, predict the unseen 5th\n")
    print(f"{'day':>4} {'MSE':>8} {'%err':>6} {'r':>6}")
    for k in range(5):
        tr, te = d.day != k, d.day == k
        p = make_model(model_name).fit(X[tr], y[tr]).predict(X[te])
        m = metrics(y[te], np.ravel(p))
        print(f"{k+1:>4} {m['MSE']:8.4f} {m['pct_error']:5.1f}% {m['pearson_r']:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1", choices=["S1", "S2"])
    ap.add_argument("--csv", help="CSV with the 29 feature columns")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", default="Gaussian Process")
    a = ap.parse_args()
    if a.demo:
        return demo(a.subject, a.model)
    bundle = joblib.load(ROOT / "results" / "models" / f"{a.subject}__Allfeatures__Gaussian.joblib")
    df = pd.read_csv(a.csv)
    if "subject" in df.columns:
        df = df[df.subject == a.subject]
    mu, sd = bundle["model"].predict(df[bundle["features"]].to_numpy(), return_std=True)
    df = df.assign(predicted_cost=mu, ci95_low=mu - 1.96 * sd, ci95_high=mu + 1.96 * sd)
    df.to_csv(a.out, index=False)
    if "metabolic_cost" in df.columns:
        m = metrics(df.metabolic_cost.to_numpy(), mu)
        print(f"(in-sample) MSE={m['MSE']:.4f}  r={m['pearson_r']:.3f}")
    print(f"wrote {len(df)} predictions to {a.out}")


if __name__ == "__main__":
    main()
