# Data

`raw/processed_data.mat` is the processed feature matrix released by E. Krimsky and E. Ng with
their Stanford CS229 (2018) project (github.com/eleyng/cs229project). The raw sensor data come
from an ongoing study in the Stanford Biomechatronics Laboratory and are not public.

| key in .mat | shape | meaning |
|---|---|---|
| `michael_data`, `michael_metabolics` | 180×29, 180×1 | Subject 1 |
| `eley_data`, `eley_metabolics` | 180×29, 180×1 | Subject 2 |
| `data_labels` | 1×29 | original column names |

Rows are ordered by day (5 days) and then by control condition (36 per day).
`raw/k_fold_data.mat` (the original authors' MATLAB CV results) is not included in the repository; no code needs it.

`hilo_metabolic_dataset.csv` is the same data in tidy form, with readable column names and
`subject`, `day` and `condition` columns. `scripts/run_experiments.py` regenerates it.
