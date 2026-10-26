"""Dataset loading and feature-group definitions.

The data come from the Stanford Biomechatronics Lab human-in-the-loop optimisation (HILO)
study used by Krimsky & Ng (CS229, 2018). Each subject walked 5 days x 36 control
conditions (2 min each); one row = the last 30 s of one condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
RAW_MAT = ROOT / "data" / "raw" / "processed_data.mat"

CONDITIONS_PER_DAY = 36
NOISE_FLOOR_MSE = 0.0025  # inherent metabolic-measurement noise (~5 % error), from the paper

# Human-readable names for the 29 raw columns
STEP_FEATURES = [
    "L_peak_force", "L_peak_dorsiflexion", "L_peak_plantarflexion", "L_step_time",
    "R_peak_force", "R_peak_dorsiflexion", "R_peak_plantarflexion", "R_step_time",
    "step_width",
]
EMG_FEATURES = [f"EMG_{side}{i}" for side in ("L", "R") for i in range(1, 9)]
# Zhang et al. (2017) torque-profile parameters; values in the data (~36, ~52 %, ~26 %, ~10 %)
# match peak torque, peak time, rise time and fall time respectively.
CONTROL_FEATURES = ["ctrl_peak_torque", "ctrl_peak_time", "ctrl_rise_time", "ctrl_fall_time"]
ALL_FEATURES = STEP_FEATURES + EMG_FEATURES + CONTROL_FEATURES

FEATURE_SETS = {
    "All features": ALL_FEATURES,
    "Step + EMG": STEP_FEATURES + EMG_FEATURES,
    "Step only": STEP_FEATURES,
    "EMG only": EMG_FEATURES,
}

GROUP_OF = {f: "Step" for f in STEP_FEATURES} | {f: "EMG" for f in EMG_FEATURES} | {
    f: "Control" for f in CONTROL_FEATURES
}

# Subject 1 = higher-variance subject in the paper ("michael"), Subject 2 = "eley"
_MAT_KEYS = {"S1": "michael", "S2": "eley"}


@dataclass
class SubjectData:
    subject: str
    X: pd.DataFrame          # 29 features
    y: np.ndarray            # normalised metabolic cost
    day: np.ndarray          # 0..4 experiment day (for leave-one-day-out CV)
    condition: np.ndarray    # 0..35 condition index within the day

    def features(self, names: list[str]) -> np.ndarray:
        return self.X[names].to_numpy()


def load_subject(subject: str = "S1", drop_invalid: bool = True) -> SubjectData:
    """Load one subject. Rows with a negative normalised metabolic rate are sensor
    artefacts (physically impossible) and are dropped, exactly as in the original study."""
    mat = loadmat(RAW_MAT)
    key = _MAT_KEYS[subject]
    X = np.asarray(mat[f"{key}_data"], dtype=float)
    y = np.asarray(mat[f"{key}_metabolics"], dtype=float).ravel()
    idx = np.arange(len(y))
    day, cond = idx // CONDITIONS_PER_DAY, idx % CONDITIONS_PER_DAY
    keep = y >= 0 if drop_invalid else np.ones_like(y, dtype=bool)
    df = pd.DataFrame(X[keep], columns=ALL_FEATURES)
    return SubjectData(subject, df.reset_index(drop=True), y[keep], day[keep], cond[keep])


def load_all() -> dict[str, SubjectData]:
    return {s: load_subject(s) for s in _MAT_KEYS}


def to_frame(sd: SubjectData) -> pd.DataFrame:
    df = sd.X.copy()
    df.insert(0, "subject", sd.subject)
    df.insert(1, "day", sd.day + 1)
    df.insert(2, "condition", sd.condition + 1)
    df["metabolic_cost"] = sd.y
    return df
