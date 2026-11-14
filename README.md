# Causal Stability Selection (CausalStabSel)

> Stable effect modifier discovery with false discovery control

## Associated paper

- **Causal stability selection**  
  [arXiv](https://arxiv.org/abs/2605.09300)

## Installation
```
pip install causalstabsel
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
`cate_estimator`: method used to estimate pseudo-outcomes (str; default `'drlearner_gb'`):
- `'drlearner_gb'`, `'drlearner_rf'`, `'drlearner_ridge'`: Doubly-robust (DR) learner, with gradient boosting (XGBoost), random forest, or ridge regression nuisance models.
- `'tlearner_gb'`, `'tlearner_rf'`, `'tlearner_ridge'`: T-learner.
- `'xlearner_gb'`, `'xlearner_rf'`, `'xlearner_ridge'`: X-learner.
- `'causal_forest'`: Causal forest via [`econml`](https://github.com/py-why/EconML)'s `CausalForestDML`. `econml` is not installed by default; install it with `pip install causalstabsel[causal_forest]`.

`cate_args`: estimator-specific keyword arguments (dict; default `None`, which resolves to each estimator's own defaults).

`propensity_estimator`: how treatment propensities are estimated (used by the DR- and X-learners; default `None`, which uses logistic regression for the DR-learners and a constant propensity of 0.5 for the X-learners). Options:
- `'logistic_regression'`: Logistic regression (scikit-learn).
- `'rf'`: Random forest (scikit-learn).
- `'gb'` or `'xgb'`: Gradient boosting (XGBoost).
- A float in `[0, 1]`: Use this constant propensity for all samples.
- Custom: a function `propensity_estimator(x_train, t_train, x_test) -> array of propensities for x_test`.

### Smoothing
Cross-fit CATE estimates are noisy at the sample level, so by default causalstabsel smooths them with a secondary regression model before running the base selector:
- `smooth_cate`: Whether to smooth CATE pseudo-outcomes (bool; default `None`, which resolves to `False` if `selector` is `'l1'`/`'adaptive_lasso'` and `True` otherwise).
- `smoother`: Regression model class used for smoothing when `smooth_cate=True` (default `None`, which resolves to XGBoost's `XGBRegressor`). Must implement `.fit(X, y)` / `.predict(X)`.
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
- `random_state`: Seed for preselection, subsampling, and every model fit (int, `RandomState`, or `None`; default `None` uses numpy's global random state, so `np.random.seed` also works). Results are the same for any `n_jobs`.

## Reference
* Bargagli-Stoffi, F.J., Melikechi, O. 2026. Causal Stability Selection. arXiv preprint arXiv:2605.09300 </b> [<a href="https://arxiv.org/abs/2605.09300">preprint</a>]

```
@article{bargagli2026causal,
  title={Causal Stability Selection},
  author={Bargagli-Stoffi, Falco J and Melikechi, Omar},
  journal={arXiv preprint arXiv:2605.09300},
  year={2026}
}
```
