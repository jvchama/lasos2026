# Universitários + CBIS 2026 Project

Machine-learning analysis of the relationship between the **U-SMILE** lifestyle
domains and **GAD-7** anxiety severity, with SHAP-based explainability.

The project runs the same modelling pipeline twice: once on the full
international (global) dataset, and once per world region / country
(Brazil, Africa, Asia, Europe, North America, Oceania, South America).

## Pipeline

Each notebook follows the same flow:

1. **Load** the regional baseline dataset (`.xlsx` / `.csv`).
2. **Select features** — the 24 U-SMILE items grouped into 7 domains:
   `food`, `subs` (substances), `PA` (physical activity), `stress`,
   `sleep`, `social`, `env` (environment).
3. **Target** — `Severity_GAD7` (GAD-7 anxiety severity class).
4. **Preprocessing** — `KNNImputer` + `StandardScaler` inside a
   `sklearn.pipeline.Pipeline`; train/test split via `ShuffleSplit`
   (5 splits, 20% test, `random_state=50`).
5. **Class balancing** — `SMOTE` (`sampling_strategy="minority"`,
   `k_neighbors=1`) applied to the training fold only.
6. **Models compared**:
   - XGBoost (`XGBClassifier`, `tree_method="hist"`)
   - Random Forest (`RandomForestClassifier`)
   - LightGBM
   - SVM
   - Logistic Regression — **final model used for SHAP explainability**
7. **Evaluation** — precision, recall, F1 across the 5 splits.
8. **Explainability** — SHAP beeswarm and bar plots for the overall model
   and for the "casos graves" (severe cases) subset.

## Repository layout

```
.
├── global/              # International (pooled) analysis
│   ├── global.ipynb
│   └── global_*.png     # SHAP / bar plots
├── brasil/              # Brazil analysis (UniLife baseline CSV)
│   ├── brazil_data.ipynb
│   ├── regions_shap/    # SHAP plots per Brazilian region
│   ├── courses_shap/    # SHAP plots per academic course
│   └── br_*.png
├── africa/              # af_jan26.ipynb  + af_*.png
├── asia/                # as_jan26.ipynb + as_*.png
├── europe/              # eu_jan26.ipynb + eu_*.png
├── north_america/       # na_jan26.ipynb + na_*.png
├── oceania/             # oc_jan26.ipynb + oc_*.png
├── south_america/       # sa_jan26.ipynb + sa_*.png
├── *_jan26.xlsx         # Raw regional baseline exports (Jan 2026)
├── Final baseline database - INT - Enviado *.xlsx   # International baseline
├── INT - Global data - Corrigido IMC - Enviado 24.04.2026.xlsx
└── requirements.txt
```

Each regional folder is self-contained: it holds its own notebook,
the raw `.xlsx` export for that region, and the generated SHAP / bar
figures (`*_bar.png`, `*_beeswarm.png`, `*_beeswarm_g.png`,
`*_severe_beeswarm.png`, `*_severe_beeswarm_g.png`).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` pins `setuptools<82` because `hyperopt` imports
> `pkg_resources`, which was removed in setuptools 82+.

Then open the notebooks in Jupyter / VS Code and run them in order.
The Brazil notebook expects the UniLife CSV
`03_unilife_baseline_database_3 [labels_var_categoricas] - banco .csv`
inside `brasil/`; the regional notebooks read their `*_jan26.xlsx`.

## Key dependencies

- `pandas`, `numpy` — data handling
- `scikit-learn`, `scipy` — preprocessing, splitting, metrics, SVM, LR
- `imbalanced-learn` — SMOTE
- `xgboost`, `lightgbm` — gradient-boosted trees
- `shap` — model explainability
- `hyperopt` — (optional) hyperparameter search
- `matplotlib` — plotting

## Outputs

Generated figures (already committed) follow this naming convention:

| Suffix                       | Content                                            |
| --------------------------- | -------------------------------------------------- |
| `_bar.png`                   | SHAP global bar plot (mean \|SHAP\| per feature)  |
| `_beeswarm.png`              | SHAP beeswarm, full sample                         |
| `_beeswarm_g.png`            | SHAP beeswarm, grouped by U-SMILE domain           |
| `_severe_beeswarm.png`       | SHAP beeswarm, severe-cases subset                 |
| `_severe_beeswarm_g.png`     | SHAP beeswarm, severe cases grouped by domain      |

Brazil additionally produces per-region and per-course SHAP plots under
`brasil/regions_shap/` and `brasil/courses_shap/`, plus decision plots
for class 0 / class 1.

## Notes

- The final explainability model for Global analysisis **Logistic Regression**; the tree-based
  models (XGBoost, Random Forest, LightGBM) and SVM are used as baselines
  for comparison of precision / recall / F1. Brazillian cutout is LightGBM, check article for
  explanation. 
- SMOTE is fit on the training fold only to avoid leakage.
