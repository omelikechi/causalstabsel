# Changelog

## 0.3.0

Requires ipss 2.0.0. Results differ from 0.2.0 for the same call.

### Changed
- **IPSS scoring.** efp scores and q-values use the corrected IPSS 2.0.0 scoring. Under 0.2.0, they were too small, and the FDR could exceed its target several-fold.
- **Default `ipss_function` is `'h4'`** (was `'h3'`), which recovers much of the power lost to the corrected scoring.
- **Default `delta` is 1.5** (was 2).
- **Default preselector for `selector='gb'` is gb** (was dcor), through ipss 2.0.0.
- **`jitter`** (new) is passed to IPSS.
- **Subsamples** have exactly `subsample_size` samples, with the treatment proportions of the full sample.
- **`cate_args` and `smoother_args`** are added to the defaults instead of replacing them.

### Fixed
- The ridge CATE estimators use `cate_args`.
- Warning filters are no longer changed for the rest of the Python session.
- Every fit gets its own seed; a `random_state` in `selector_args`, `cate_args`, or `smoother_args` is ignored with a warning.
- Invalid arguments raise clear errors before any fitting.
