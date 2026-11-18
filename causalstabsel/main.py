# Causal stability selection

import time
import warnings

from ipss.helpers import (is_integer, is_number, return_null_result, stratified_halves, validate_scores,
	validate_targets)
from ipss.processing import postprocess_ipss, preprocess_ipss
from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import check_random_state

from .cate_estimators import cate_estimators, validate_propensity_estimator, xgb_regressor

#----------------------------------------------------------------
# Causal stability selection
#----------------------------------------------------------------
def causalstabsel(
	# data
	X, y, treatment, 
	# base selector for ipss
	selector='gb', selector_args=None, 
	# cate estimation method
	cate_estimator='drlearner_gb', cate_args=None,
	# smooth cate
	smooth_cate=None, smoother=None, smoother_args=None,
	# propensity estimation
	propensity_estimator=None, 
	# preselection
	preselect=True, preselector=None, preselector_args=None,
	# nominal false discovery values (optional)
	target_fp=None, target_fdr=None,
	# additional ipss parameters 
	B=100, n_alphas=None, ipss_function='h4', cutoff=0.05, delta=1.5,
	# subsample size
	subsample_size=None, 
	# standardize data
	standardize_X=None, center_y=None, 
	# parallel processing
	n_jobs=1,
	# random seed (int, RandomState, or None); None uses numpy's global random state
	random_state=None,
	# jitter categorical features before fitting (see ipss); None jitters for the 'gb' and 'rf' selectors
	jitter=None
	):

	# start timer
	start = time.time()
	rng = check_random_state(random_state)

	# validate arguments before any fitting
	validate_targets(target_fp, target_fdr)
	treatment = validate_treatment(treatment, np.shape(X)[0] if np.ndim(X) >= 1 else 0)
	if isinstance(cate_estimator, str):
		if cate_estimator not in cate_estimators:
			raise ValueError(f'Unrecognized cate_estimator: {cate_estimator!r}. Must be one of '
				f'{sorted(cate_estimators)} or a custom callable.')
		cate_estimator = cate_estimators[cate_estimator]
	elif not callable(cate_estimator):
		raise ValueError(f'cate_estimator must be a string or a callable, but got {cate_estimator!r}.')
	validate_propensity_estimator(propensity_estimator)
	if not (is_number(cutoff) and cutoff > 0):
		raise ValueError(f'cutoff must be a positive number, but got {cutoff!r}.')
	if smooth_cate is not None and not isinstance(smooth_cate, (bool, np.bool_)):
		raise ValueError(f'smooth_cate must be True, False, or None, but got {smooth_cate!r}.')
	for name, args in (('selector_args', selector_args), ('cate_args', cate_args), ('smoother_args', smoother_args),
			('preselector_args', preselector_args)):
		if args is not None and not isinstance(args, dict):
			raise ValueError(f'{name} must be a dict or None, but got {args!r}.')
	if smoother is not None and not callable(smoother):
		raise ValueError(f'smoother must be a callable that returns a model with fit and predict, but got {smoother!r}.')
	if subsample_size is not None and not (is_integer(subsample_size) and subsample_size >= 1) and not (
			isinstance(subsample_size, (float, np.floating)) and 0 < subsample_size < 1):
		raise ValueError(f'subsample_size must be a positive integer or a fraction in (0, 1), but got {subsample_size!r}.')

	# every fit gets its own seed: a seed in any argument dict would give every fit the same one
	selector_args = drop_random_state(selector_args, 'selector_args')
	cate_args = drop_random_state(cate_args, 'cate_args')
	smoother_args = drop_random_state(smoother_args, 'smoother_args')

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector, preselector_args,
		B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression=True, random_state=rng,
		jitter=jitter)

	if prep is None:
		return return_null_result(np.shape(X)[1], time.time() - start)
	
	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B, delta = prep['alphas'], prep['n_alphas'], prep['B'], prep['delta']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']

	n, p = X.shape

	# default smoother settings: no smoothing for the l1 selectors (which have a regularization grid), smoothing otherwise
	if smooth_cate is None:
		smooth_cate = alphas is None
	if smooth_cate and smoother is None:
		smoother = xgb_regressor
		smoother_args = {'n_estimators': 20, 'max_depth': 3, **(smoother_args or {})}
	smoother_args = smoother_args or {}

	# subsample size: a fraction of n, or a number of samples; at most n // 2
	if subsample_size is None:
		subsample_size = n // 2
	elif not is_integer(subsample_size):
		subsample_size = int(subsample_size * n)
	subsample_size = min(int(subsample_size), n // 2)
	if subsample_size < 1:
		raise ValueError('subsample_size is too small: each subsample must contain at least one sample.')

	# one seed per subsampling step, drawn here so that results do not depend on n_jobs
	seeds = rng.randint(np.iinfo(np.int32).max, size=B)

	# estimate selection probabilities
	results = Parallel(n_jobs=n_jobs)( 
		delayed(causal_selection)(
			X, y, treatment, cate_estimator, propensity_estimator, alphas,
			selector_function, selector_args, subsample_size,
			cate_args, smooth_cate, smoother, smoother_args, seed=seeds[b]
		)
		for b in range(B)
	)

	results = np.array(results)

	# postprocess estimated selection probabilities (delta is a number, or a list of numbers)
	if np.ndim(delta) == 0:
		ipss_output = postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
			ipss_function, target_fdr, target_fp)
		ipss_output['runtime'] = time.time() - start
	else:
		ipss_output = {}
		for idx, d in enumerate(delta):
			ipss_output[d] = postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, d, 
				ipss_function, target_fdr, target_fp)
			if idx == 0:
				ipss_output['runtime'] = time.time() - start

	return ipss_output

# treatment as a float vector of 0s and 1s with at least 2 samples in each arm
def validate_treatment(treatment, n):
	try:
		t = np.asarray(treatment, dtype=np.float64)
	except (TypeError, ValueError):
		raise ValueError('treatment must contain only 0 (control) and 1 (treated).') from None
	if t.ndim == 2 and 1 in t.shape:
		t = t.ravel()
	if t.ndim != 1 or t.size != n:
		raise ValueError(f'treatment must be a vector with one value per sample ({n}), but has shape {t.shape}.')
	if not np.all(np.isin(t, [0, 1])):
		raise ValueError('treatment must contain only 0 (control) and 1 (treated), with no missing values.')
	n_treated = int(np.sum(t == 1))
	if min(n_treated, n - n_treated) < 2:
		raise ValueError(f'Each treatment arm needs at least 2 samples, but there are {n_treated} treated and '
			f'{n - n_treated} control samples.')
	return t

# remove a fixed seed from an argument dict; an emptied dict becomes None so that defaults still apply
def drop_random_state(args, name):
	if not args or 'random_state' not in args:
		return args
	warnings.warn(f"{name}['random_state'] is ignored so that every fit gets its own seed; "
		'use random_state instead.', UserWarning)
	return {key: value for key, value in args.items() if key != 'random_state'} or None

# subsampler for estimating selection probabilities
def causal_selection(*args, seed=None):
	'''Seeds numpy's global random state for one subsampling step. The subsampling and the
	CATE, smoother, and selector fits all draw from it, and in parallel runs each worker
	process otherwise has its own unseeded state. The caller's state and warning filters are restored afterward.'''
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		if seed is None:
			return _causal_selection(*args)
		state = np.random.get_state()
		np.random.seed(seed)
		try:
			return _causal_selection(*args)
		finally:
			np.random.set_state(state)

def _causal_selection(
	X, y, treatment, cate_estimator, propensity_estimator, alphas,
	selector_function, selector_args, subsample_size,
	cate_args, smooth_cate, smoother, smoother_args
	):

	n, p = X.shape
	indices = np.arange(n)

	# two disjoint subsamples of equal size, each with the treatment proportions of the full sample
	halves = stratified_halves(treatment, subsample_size, np.random)

	if alphas is None:
		indicators = np.empty((2, p))
	else:
		indicators = np.empty((len(alphas), 2, p))

	for half, test_idx in enumerate(halves):

		# out-of-bag training data
		train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

		# smooth estimate
		if smooth_cate:
			tau_train = validate_scores(cate_estimator(
				X[train_idx], y[train_idx], treatment[train_idx],
				X[train_idx], propensity_estimator, cate_args
			), (train_idx.size,), 'the CATE estimator')
			meta = smoother(**smoother_args)
			meta.fit(X[train_idx], tau_train)
			tau_hat = validate_scores(meta.predict(X[test_idx]), (test_idx.size,), 'the smoother')

		# no smoothing (standard cross-fit)
		else:
			tau_hat = validate_scores(cate_estimator(
				X[train_idx], y[train_idx], treatment[train_idx],
				X[test_idx], propensity_estimator, cate_args
			), (test_idx.size,), 'the CATE estimator')

		# selection
		if alphas is None:
			indicators[half, :] = validate_scores(selector_function(X[test_idx], tau_hat, **selector_args), (p,),
				'the selector')
		else:
			indicators[:, half, :] = validate_scores(selector_function(
				X[test_idx], tau_hat, alphas, **selector_args
			), (len(alphas), p), 'the selector')

	return indicators


