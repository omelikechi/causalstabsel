# Causal Stability Selection (CausalStabSel)

> Stable feature selection for heterogeneous treatment effects, with false discovery control

CausalStabSel extends [integrated path stability selection (IPSS)](https://github.com/omelikechi/ipss) to causal inference: instead of selecting features associated with a response `y`, it selects features associated with the *conditional average treatment effect* (CATE), `tau(X) = E[Y(1) - Y(0) | X]`. On each subsample, a CATE estimator (DR-learner, T-learner, X-learner, or causal forest) produces pseudo-outcomes, which are then passed to IPSS's stability selection machinery to produce q-values and expected-false-positive (E(FP)) bounds for each feature.

## Installation
```
pip install causalstabsel
```
The `causal_forest` CATE estimator requires [`econml`](https://github.com/py-why/EconML), which is not installed by default:
```
pip install causalstabsel[causal_forest]
```

## Usage
```python
from causalstabsel import causalstabsel

# load n-by-p feature matrix X, n-by-1 response vector y, and n-by-1 binary treatment indicator

# run causal stability selection
output = causalstabsel(X, y, treatment)

# select features based on target FDR
target_fdr = 0.1
q_values = output['q_values']
selected_features = [idx for idx, q_value in q_values.items() if q_value <= target_fdr]
print(f'Selected features (target FDR = {target_fdr}): {selected_features}')
```

### Outputs
`output = causalstabsel(X, y, treatment)` is a dictionary containing:
- `efp_scores`: Dictionary whose keys are feature indices and values are their efp scores (dict of length `p`).
- `q_values`: Dictionary whose keys are feature indices and values are their q-values (dict of length `p`).
- `runtime`: Runtime of the algorithm in seconds (float).
- `selected_features`: Indices of features selected by CausalStabSel; empty list if `target_fp` and `target_fdr` are not specified (list of ints).
- `stability_paths`: Estimated selection probabilities at each parameter (array of shape `(n_alphas, p)`).

If `delta` is a list rather than a single value, `output` instead maps each `delta` to its own result dictionary of the form above.

### Selecting features
Each feature (column of `X`) is assigned:
- a **q-value**: the minimum false discovery rate (FDR) at which the feature is selected
- an **efp score**: the minimum expected number of false positives (E(FP)) at which the feature is selected

To select features:
- **Control FDR** by choosing all features with `q_value ≤ target_fdr`
  _Example: Selecting features with `q_value ≤ 0.1` controls the FDR at level 0.1_
- **Control E(FP)** by choosing all features with `efp_score ≤ target_fp`
  _Example: Selecting features with `efp_score ≤ 2` controls the E(FP) at level 2_

## CATE estimators
`cate_estimator` selects the method used to estimate pseudo-outcomes on each subsample (str; default `'drlearner_gb'`):
- `'drlearner_gb'`, `'drlearner_rf'`, `'drlearner_ridge'`: Doubly-robust (DR) learner, with gradient boosting (XGBoost), random forest, or ridge regression nuisance models.
- `'tlearner_gb'`, `'tlearner_rf'`, `'tlearner_ridge'`: T-learner.
- `'xlearner_gb'`, `'xlearner_rf'`, `'xlearner_ridge'`: X-learner.
- `'causal_forest'`: Causal forest via [`econml`](https://github.com/py-why/EconML)'s `CausalForestDML` (requires the `causal_forest` extra, see Installation).

`cate_args` passes estimator-specific keyword arguments (dict; default `None`, which resolves to each estimator's own defaults).

`propensity_estimator` selects how treatment propensities are estimated where needed (used by the DR- and X-learners; default `None`, which assumes a constant propensity of 0.5, appropriate for randomized experiments). Options:
- `'logistic_regression'`: Logistic regression (scikit-learn).
- `'rf'`: Random forest (scikit-learn).
- `'gb'` or `'xgb'`: Gradient boosting (XGBoost).
- A float in `[0, 1]`: Use this constant propensity for all samples.
- Custom: a function `propensity_estimator(x_train, t_train, x_test) -> array of propensities for x_test`.

### Smoothing
Cross-fit CATE estimates are noisy at the sample level, so by default CausalStabSel smooths them with a secondary regression model before running the base selector:
- `smooth_cate`: Whether to smooth CATE pseudo-outcomes (bool; default `None`, which resolves to `False` if `selector` is `'l1'`/`'adaptive_lasso'` and `True` otherwise).
- `smoother`: Regression model class used for smoothing when `smooth_cate=True` (default `None`, which resolves to `xgb.XGBRegressor`). Must implement `.fit(X, y)` / `.predict(X)`.
- `smoother_args`: Arguments passed to `smoother` (dict; default `None`, which resolves to `{'n_estimators': 20, 'max_depth': 3}` for the default smoother).

## Full list of `causalstabsel` arguments

### Required arguments:
- `X`: Features (array of shape `(n,p)`), where `n` is the number of samples and `p` is the number of features.
- `y`: Response (array of shape `(n,)` or `(n, 1)`).
- `treatment`: Binary treatment indicator (array of shape `(n,)`, values in `{0, 1}`).

### Optional arguments:
- `selector`: Base algorithm used to score smoothed CATE estimates (str; default `'gb'`). See the [IPSS](https://github.com/omelikechi/ipss) documentation for all options, including custom feature importance functions.
- `selector_args`: Arguments for the base algorithm (dict; default `None`).
- `cate_estimator`, `cate_args`, `propensity_estimator`: See CATE estimators above.
- `smooth_cate`, `smoother`, `smoother_args`: See Smoothing above.
- `preselect`: Preselect/filter features prior to subsampling (bool; default `True`).
- `preselector`, `preselector_args`: Preselection method and its arguments; see the [IPSS](https://github.com/omelikechi/ipss) documentation.
- `target_fp`: Target number of false positives to control (positive float; default `None`).
- `target_fdr`: Target false discovery rate (FDR) (positive float; default `None`).
- `B`: Number of subsampling steps (int; default `100`).
- `n_alphas`: Number of values in the regularization or threshold grid (int; default `None`, resolves to an IPSS-determined default).
- `ipss_function`: Function applied to selection probabilities (str; default `'h3'`). Options `'h1'`, `'h2'`, `'h3'`; see [IPSS](https://github.com/omelikechi/ipss).
- `cutoff`: Maximum value of the theoretical integral bound `I(Lambda)` (positive float; default `0.05`).
- `delta`: Defines probability measure; see [IPSS](https://github.com/omelikechi/ipss) (float, list of floats, or `None`; default `None` resolves to `2`). If a list, `causalstabsel` returns one result dictionary per value.
- `subsample_size`: Size of each subsample used for cross-fitting (int, float in `(0, 1)`, or `None`; default `None` resolves to `n // 2`, and values above `n // 2` are capped there since two disjoint subsamples of this size are drawn per step).
- `standardize_X`: Scale features to have mean 0, standard deviation 1 (bool; default `None`).
- `center_y`: Center response to have mean 0 (bool; default `None`).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).
