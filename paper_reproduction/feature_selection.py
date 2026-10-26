
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def forward_stepwise_selection(X, y, n_splits=5, max_features=None, random_state=42):
    n_features = X.shape[1]
    max_features = min(max_features or n_features, n_features)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    remaining = list(range(n_features))
    selected = []
    history = []

    for step in range(max_features):
        scores = []

        for candidate in remaining:
            trial = selected + [candidate]
            fold_scores = []

            for train_idx, test_idx in kf.split(X):
                model = Ridge(alpha=1.0)
                model.fit(X[train_idx][:, trial], y[train_idx])
                pred = model.predict(X[test_idx][:, trial])
                fold_scores.append(mean_squared_error(y[test_idx], pred))

            scores.append(np.mean(fold_scores))

        best_pos = int(np.argmin(scores))
        best_feature = remaining.pop(best_pos)
        best_mse = float(scores[best_pos])

        selected.append(best_feature)
        history.append((step + 1, best_feature, best_mse))

    best_step = int(np.argmin([x[2] for x in history]))
    best_features = selected[:best_step + 1]

    return best_features, history


def main():
    """Forward stepwise selection (paper Sec. II-B.2) on each subject; prints the selection order."""
    from sklearn.preprocessing import StandardScaler
    from data_processing import load_processed_data

    data = load_processed_data()
    names = [str(x) for x in np.ravel(data["labels"])]
    for subject in ("S1", "S2"):
        X = StandardScaler().fit_transform(data[subject]["X"])
        best, history = forward_stepwise_selection(X, data[subject]["y"], max_features=15)
        print(f"\n{subject}: best subset = {len(best)} features")
        for step, feat, mse in history:
            print(f"  {step:2d}. +{names[feat]:<12} CV MSE={mse:.5f}")
        print("  (selection is greedy; see scripts/run_experiments.py E6 for a leakage-free comparison with PCA/PLS)")


if __name__ == "__main__":
    main()
