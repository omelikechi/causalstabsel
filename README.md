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
- `stability_paths`: Estimated selection probabilities at each parameter (array of shape `(k, p)`, with `k <= n_alphas`).

### Selecting features
Each feature (column of `X`) is assigned:
- a **q-value**: the minimum false discovery rate (FDR) at which the feature is selected
- an **efp score**: the minimum expected number of false positives (E(FP)) at which the feature is selected

To select features:
- **Control FDR** by choosing all features with `q_value ≤ target_fdr`  
  _Example: Selecting features with `q_value ≤ 0.1` controls the FDR at level 0.1 (approximately)_
- **Control E(FP)** by choosing all features with `efp_score ≤ target_fp`  
  _Example: Selecting features with `efp_score ≤ 2` controls the E(FP) at level 2_

## Main arguments
- `cate_estimator`: Method used to estimate conditional average treatment effects (default `'drlearner_gb'`). Options are DR-, T-, and X-learners with gradient boosting, random forest, or ridge models (`'drlearner_gb'`, `'tlearner_rf'`, `'xlearner_ridge'`, ...) and `'causal_forest'`.
- `selector`: Base algorithm for selecting effect modifiers (default `'gb'`). `'gb'` and `'rf'` capture nonlinear effect modification; `'l1'` and `'adaptive_lasso'` capture linear effect modification. See [IPSS](https://github.com/omelikechi/ipss).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).

The other arguments rarely need changing. See [ARGUMENTS.md](https://github.com/omelikechi/causalstabsel/blob/main/ARGUMENTS.md) for the full list.

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
