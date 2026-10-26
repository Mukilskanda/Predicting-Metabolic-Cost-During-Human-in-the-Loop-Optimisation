# paper_reproduction/ - faithful Python port of the original MATLAB study

One Python script per MATLAB script from Krimsky & Ng (CS229, 2018), kept deliberately simple
so each step can be traced back to the paper. The main `metacost/` package extends this baseline.

| Original MATLAB | Python here | What it does |
|---|---|---|
| `data_processing.m` (output) | `data_processing.py` | Loads `data/raw/processed_data.mat` and drops the one impossible negative target, as `k_fold_cv.m` does |
| `k_fold_cv.m` | `k_fold_cv.py` | 5-fold CV of a 1-hidden-layer tanh network (1-14 neurons) and linear regression, for 4 feature sets |
| `plots_and_tables.m` | `plots.py` | MSE-vs-neurons plots (paper Fig. 1) |
| `model_selection.m` | `model_selection.py` | Picks the best configuration |
| `feature_selection.m` | `feature_selection.py` | Forward stepwise selection (paper Sec. II-B.2) |
| `lin_reg_test.m` | `lin_reg_test.py` | Linear-regression baseline with R² |
| `predict_reduced_model.m` | `predict_reduced_model.py` | Final model + actual-vs-predicted plot (out-of-fold) |

## Run (from the project root)

```bash
python paper_reproduction/k_fold_cv.py
python paper_reproduction/plots.py
python paper_reproduction/model_selection.py
python paper_reproduction/predict_reduced_model.py
python paper_reproduction/feature_selection.py
```

Outputs go to `paper_reproduction/results/` (~2 min in total).

## Results (5-fold CV MSE, random_state = 42)

| Feature set | S1 NN (best n) | S1 Lin. Reg. | S2 NN (best n) | S2 Lin. Reg. |
|---|---|---|---|---|
| All features | 0.0112 (1) | **0.0107** | 0.0184 (11) | 0.0172 |
| Step + EMG | 0.0125 (1) | 0.0124 | 0.0201 (1) | **0.0163** |
| Step only | 0.0159 (2) | 0.0197 | 0.0200 (1) | 0.0189 |
| EMG only | 0.0157 (1) | 0.0178 | 0.0206 (1) | 0.0200 |
| *Paper (MATLAB)* | *0.0089 NN* | *0.0089* | *0.0301 NN* | *0.0168* |

This reproduces the paper's main findings: 1-3 neurons are enough, linear regression is as good as the
network, and EMG beats step data for S1.

## Implementation note: `trainbr`

MATLAB's `trainbr` (Bayesian regularisation) has no scikit-learn equivalent, so this port uses
`MLPRegressor` with a fixed L2 penalty (`alpha = 1e-4`). Its numbers are therefore a Python
re-implementation of the pipeline, not an exact copy of MATLAB's. The main package goes further:
`metacost/bayesian_mlp.py` re-implements `trainbr` itself (Levenberg-Marquardt + MacKay evidence updates).

## Fixes made while merging into the final project

1. **S2 artefact.** The original port kept one sample with a metabolic cost of −0.565, which inflated
   every S2 MSE about 2× (Lin. Reg. 0.0391 → 0.0172 after removing it). It is now dropped, as in the paper.
2. **Scaling leakage.** The scaler was fitted on all the data before CV; it now sits inside each fold's pipeline.
3. **Honest final plot.** `final_prediction.png` now shows out-of-fold predictions (MSE 0.0112), not the
   fit to the training data (training MSE 0.0058 looks better than reality).
4. Forward selection now has a runnable `main()`.

Known limitation, kept on purpose to stay faithful to the paper: the best neuron count is picked using the same
folds it is scored on, so the NN numbers here are slightly optimistic. `metacost/` uses nested CV instead.
