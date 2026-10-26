# Viva / Q&A preparation

Likely panel questions and short, accurate answers. All numbers come from `results/tables/`.

**Q. What exactly is the target?**
Metabolic rate minus that day's standing baseline, divided by that day's normal-walking rate. So 1.0 means "same as walking without the exo" and 0.6 means a 40 % saving.

**Q. Why is MSE so small? Is 0.008 good?**
The target is normalised, with variance 0.066 (S1) and 0.036 (S2). Predicting the mean gives MSE = variance. 0.0079 means RMSE ≈ 0.089, or about 9 % of walking cost, with R² = 0.88. The measurement-noise floor is ≈ 0.0025 (about 5 %), so no model can go much below it.

**Q. How do you know you're not overfitting with 180 samples and 29 features?**
(1) Nested CV: scaling and every hyper-parameter search are fitted inside training folds only. (2) Repeated 5×5 CV, with fold standard deviations reported. (3) A stricter leave-one-day-out test. (4) Regularised models only (LASSO, Ridge, GP marginal likelihood, Bayesian evidence for the NN).

**Q. Why are leave-one-day-out errors larger?**
Random splits put conditions from the same day in both train and test, and those conditions share day-level properties (sensor placement, fatigue, learning stage). That leaks information. A new day is the deployment scenario. Subject 1 still reaches R² ≈ 0.78; Subject 2's day-to-day level shifts (days 3-5) are not visible in its sensors.

**Q. Did you try normalising features per day to remove drift?**
Yes. It made results much worse (e.g. S1 Ridge 0.0145 → 0.085), because the between-day change in cost is real motor learning, and the day-level feature means encode it.

**Q. Explain the Bayesian-regularised network.**
Objective F = β·E_D + α·E_W. We minimise it with Levenberg-Marquardt steps, using the Gauss-Newton Hessian H = βJᵀJ + αI. After each step we re-estimate γ = P − α·tr(H⁻¹), the effective number of parameters, then α = γ/2E_W and β = (N−γ)/2E_D (MacKay's evidence framework). This is what MATLAB `trainbr` does. The Jacobian is closed form for one tanh layer and is unit-tested against finite differences.

**Q. Why does the GP do well?**
Its kernel is linear + RBF + white noise. It captures the mostly linear trend, a smooth non-linearity, and learns the noise level from the data via the marginal likelihood, so no CV is needed for its hyper-parameters. It also returns predictive uncertainty: the 95 % intervals cover 94 % of unseen-day points for S1.

**Q. Why did PCA perform poorly in the paper?**
PCA keeps directions of high *feature* variance, not directions predictive of the target. PLS is supervised: on S2, one PLS component (0.0138) matches PCA with 8 components.

**Q. Which features matter?**
By grouped permutation importance, EMG is the strongest group for both subjects. Right-leg channels R4 and R7 (S1) and R2 (S2) are selected in 93-100 % of 200 LASSO bootstraps. The control parameters add little, which is good for transfer to other devices.

**Q. Can one model work for everyone?**
Not with this data. Training on S1 and testing on S2 gives MSE 0.152, against 0.036 for predicting the mean. Per-subject z-scoring improves this to 0.046, which is still worse than the mean. We would need many subjects plus anthropometrics (height, weight, fitness).

**Q. How would this be used in practice?**
As a fast surrogate inside the optimiser. CMA-ES only needs the *ranking* of candidate controllers within a generation: our within-day Spearman ρ ≈ 0.6 for S1. Or use it as a prior in Bayesian optimisation, with occasional mask measurements to correct a day-level offset.

**Q. What did each member do?** Mukil Skanda A (PES1UG24CS281): extended study - from-scratch Bayesian NN, 13-model zoo, nested and leave-one-day-out validation, calibration, sensor reduction, write-up. Likhith M (PES1UG24CS282): faithful MATLAB-to-Python port of the paper (paper_reproduction/), NN vs linear-regression baseline, forward selection, GitHub repo.

**Q. Where did the data come from?**
The processed feature matrix released by the original authors with their CS229 project (Stanford Biomechatronics Lab HILO study). We did not have the raw EMG or force signals; filtering was done upstream by the lab.

## Follow-up improvements (E10-E12)

**Q. How does your session calibration work?**
On a new day we measure k random conditions with the mask and compute the residuals (measured − predicted). Instead of adding their raw mean, we add w·mean, with w = k / (k + σ²/τ²). σ² is the within-day residual noise and τ² is the variance of true day offsets. Both are estimated by an inner leave-one-day-out loop on the *training* days only, so nothing is learned from the test day except the k calibration points. This is the empirical-Bayes / random-effects (James-Stein) estimator.

**Q. Why shrink?**
With k = 1-3 the raw mean is noisy. When a subject's days are already well predicted (small τ², like S1), a noisy correction adds error. For random spot-checks, shrinkage keeps S1 at or below baseline at every k. For start-of-session calibration it limits the damage (S1 Ridge at k = 8: 0.0143 with shrinkage vs 0.0184 without, baseline 0.0120), but that strategy is biased anyway (next question). S2 has large day shifts, so w ≈ 0.87 and it gets almost the full correction.

**Q. Why spot-checks rather than calibrating at the start of the session?**
Metabolic cost runs systematically high in the first conditions of each session (warm-up, visible in Fig. 1), so the first k conditions give a biased offset. Random spot-checks across the session are unbiased.

**Q. How much does it help, and what does it cost?**
3 mask readings out of 36 conditions, i.e. 92 % fewer. S2 unseen-day MSE: GP 0.0460 → 0.0223, Ridge 0.0362 → 0.0182, LASSO 0.0280 → 0.0171 (≈13 % error). S1: GP 0.0175 → 0.0142.

**Q. Is the sensor-reduction result leakage-free?**
Yes. The channel ranking (order of entry into the LASSO path) is computed inside each training fold by the `TopKSensors` pipeline step. One caveat: the "4 electrodes" figure is read off the curve, so it is slightly optimistic. Choosing k by nested CV would be the fully clean version.

**Q. Which electrodes would you keep?**
Full-data LASSO-path ranking: S1: L3, R4, L7, L5, R1. S2: L2, R2, R8, L3, R3. They differ between people, which again argues for per-person setup.

**Q. Did calibration fix cross-subject transfer?**
No. Offset (or offset+scale) recalibration of the other person's model was no better than simply averaging the k calibration samples, and often worse. From about 5-10 samples, a model trained only on the new person's own data wins (e.g. S2: 0.030 with 10 own samples vs 0.048 for the calibrated transfer).

**Q. Why didn't you add more subjects / IMU / anthropometrics?**
They are not in the released dataset. Simulating them would be fabrication, so they remain future work.

## Paper reproduction layer

**Q. Why are there two implementations?**
`paper_reproduction/` is a faithful 1:1 Python port of the paper's MATLAB scripts. It shows we can reproduce the published baseline (S2 Lin. Reg. 0.0172 vs 0.0168 in the paper). `metacost/` is the extended study that improves on it. Keeping both lets the panel trace every step back to the paper.

**Q. Why is the reproduction's S1 number (0.0107) a bit worse than the paper's 0.0089?**
Different random fold splits on 180 samples (the fold-to-fold spread is about ±0.004), and scikit-learn's `MLPRegressor` with a fixed L2 penalty is not MATLAB's `trainbr`. Our from-scratch `trainbr` and nested CV in `metacost/` reach 0.0079.

**Q. What bugs did you find while merging?**
The first port kept the impossible −0.565 S2 sample, which doubled every S2 MSE (0.0391 → 0.0172 once removed). It also fitted the scaler on all the data before CV, and its final plot showed training-set predictions (MSE 0.0058) instead of out-of-fold ones (0.0112). All three are fixed.
