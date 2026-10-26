from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data" / "raw"


def load_processed_data(path=None):
    path = Path(path) if path else DATA_DIR / "processed_data.mat"
    if not path.exists():
        raise FileNotFoundError(f"processed_data.mat not found: {path}")

    data = loadmat(path, squeeze_me=True)
    required = [
        "eley_data", "eley_metabolics",
        "michael_data", "michael_metabolics",
    ]
    missing = [x for x in required if x not in data]
    if missing:
        raise KeyError(f"Missing variables in MAT file: {missing}")

    out = {}
    for subject, key in (("S1", "michael"), ("S2", "eley")):
        X = np.asarray(data[f"{key}_data"], dtype=float)
        y = np.asarray(data[f"{key}_metabolics"], dtype=float).reshape(-1)
        # A negative normalised metabolic rate is physically impossible (sensor artefact).
        # The original MATLAB k_fold_cv.m removes it (bad_idx = find(eley_metabolics < 0)).
        # Keeping it doubles every S2 MSE (LR 0.0172 -> 0.0391), so it is dropped here.
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y >= 0)
        out[subject] = {"X": X[valid], "y": y[valid]}
    return {
        **out,
        "labels": data.get("data_labels"),
    }


if __name__ == "__main__":
    d = load_processed_data()
    for subject, value in d.items():
        if subject in ("S1", "S2"):
            print(f"{subject}: X={value['X'].shape}, y={value['y'].shape}")
