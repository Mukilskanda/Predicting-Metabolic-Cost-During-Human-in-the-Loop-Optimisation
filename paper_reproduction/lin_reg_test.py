
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_processing import load_processed_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_processed_data()
    X = data["S1"]["X"]
    y = data["S1"]["y"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []

    for train_idx, test_idx in kf.split(X):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        mse_scores.append(mean_squared_error(y[test_idx], pred))
        r2_scores.append(r2_score(y[test_idx], pred))

    print("Linear Regression 5-Fold CV")
    print(f"Mean MSE: {sum(mse_scores)/len(mse_scores):.6f}")
    print(f"Mean R² : {sum(r2_scores)/len(r2_scores):.4f}")


if __name__ == "__main__":
    main()
