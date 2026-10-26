
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "k_fold_results.mat"

CONFIGS = {
    "all": "All Features",
    "step_emg": "Step + EMG",
    "step_only": "Step Only",
    "emg_only": "EMG Only",
}


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            "Run k_fold_cv.py first so results/k_fold_results.mat exists."
        )

    d = loadmat(RESULT_FILE)
    best = None

    for subject, prefix in (("S1", "michael"), ("S2", "eley")):
        for key, display in CONFIGS.items():
            nn = np.asarray(d[f"{prefix}_{key}_nn_mse"]).reshape(-1)
            lr = float(np.asarray(d[f"{prefix}_{key}_lr_mse"]).reshape(-1)[0])

            neuron = int(np.argmin(nn) + 1)
            mse = float(np.min(nn))

            print(
                f"{subject} | {display:<12} | "
                f"NN MSE={mse:.6f} ({neuron} neurons) | "
                f"LR MSE={lr:.6f}"
            )

            candidate = (mse, subject, key, neuron, lr)
            if best is None or candidate[0] < best[0]:
                best = candidate

    print("\nBEST NEURAL NETWORK")
    print(f"Subject       : {best[1]}")
    print(f"Configuration : {CONFIGS[best[2]]}")
    print(f"Neurons       : {best[3]}")
    print(f"MSE           : {best[0]:.6f}")
    print(f"LR baseline   : {best[4]:.6f}")


if __name__ == "__main__":
    main()
