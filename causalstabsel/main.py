# Causal stability selection

import time
import warnings

from ipss.helpers import return_null_result
from ipss.processing import postprocess_ipss, preprocess_ipss
from joblib import Parallel, delayed
import numpy as np
import xgboost as xgb

from .cate_estimators import cate_estimators

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
	preselect=True, preselector_args=None, 
	# nominal false discovery values (optional)
	target_fp=None, target_fdr=None,
	# additional ipss parameters 
	B=100, n_alphas=None, ipss_function='h3', cutoff=0.05, delta=None,
	# subsample size
	subsample_size=None, 
	# standardize data
	standardize_X=None, center_y=None, 
	# parallel processing
	n_jobs=1
	):

	# start timer
	start = time.time()

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector_args,
		B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression=True)

	if prep is None:
		return return_null_result(X.shape[1], time.time() - start)
	
	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B = prep['alphas'], prep['n_alphas'], prep['B']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']

	n, p = X.shape

	# default smoother settings
	if smooth_cate is None:
		# no smoothing if linear, smoothing otherwise
		if selector in ['l1', 'adaptive_lasso']:
			smooth_cate = False
		else:
			smooth_cate = True
			# default smoother
			if smoother is None:
				smoother = xgb.XGBRegressor
			if smoother_args is None:
				smoother_args = dict(n_estimators=20, max_depth=3)

	# cate estimator
	if isinstance(cate_estimator, str):
		cate_estimator = cate_estimators[cate_estimator]

	# subsample size
	if subsample_size is None or subsample_size > n // 2:
		subsample_size = n // 2
	elif 0 < subsample_size < 1:
		subsample_size = int(subsample_size * n)

	# stratify samples by treatment
	treated = np.where(treatment == 1)[0]
	control = np.where(treatment == 0)[0]

	# estimate selection probabilities
	results = Parallel(n_jobs=n_jobs)( 
		delayed(causal_selection)(
			X, y, treatment, cate_estimator, propensity_estimator, alphas,
			selector_function, selector_args, subsample_size,
			cate_args, smooth_cate, smoother, smoother_args,
			treated, control
		)
		for _ in range(B)
	)

	results = np.array(results)

	# postprocess estimated selection probabilities
	if delta is None:
		delta = 2

	if isinstance(delta, (int, float)):
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

# subsampler for estimating selection probabilities
def causal_selection(
	X, y, treatment, cate_estimator, propensity_estimator, alphas,
	selector_function, selector_args, subsample_size,
	cate_args, smooth_cate, smoother, smoother_args,
	treated, control
	):

	warnings.filterwarnings('ignore')

	n, p = X.shape
	indices = np.arange(n)

	# shuffle within strata
	treated = np.random.permutation(treated)
	control = np.random.permutation(control)
	n_treated = len(treated)

	# proportional allocation
	n1_treated = int(round(subsample_size * n_treated / n))
	n1_control = subsample_size - n1_treated
	half1 = np.concatenate([treated[:n1_treated], control[:n1_control]])
	half2 = np.concatenate([treated[n1_treated:2*n1_treated], control[n1_control:2*n1_control]])
	np.random.shuffle(half1)
	np.random.shuffle(half2)
	halves = [half1, half2]

	if alphas is None:
		indicators = np.empty((2, p))
	else:
		indicators = np.empty((len(alphas), 2, p))

	for half, test_idx in enumerate(halves):

		# out-of-bag training data
		train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

		# smooth estimate
		if smooth_cate:
			tau_train = cate_estimator(
				X[train_idx], y[train_idx], treatment[train_idx],
				X[train_idx], propensity_estimator, cate_args
			)
			meta = smoother(**smoother_args)
			meta.fit(X[train_idx], tau_train)
			tau_hat = meta.predict(X[test_idx])

		# no smoothing (standard cross-fit)
		else:
			tau_hat = cate_estimator(
				X[train_idx], y[train_idx], treatment[train_idx],
				X[test_idx], propensity_estimator, cate_args
			)

		# selection
		if alphas is None:
			indicators[half, :] = selector_function(X[test_idx], tau_hat, **selector_args)
		else:
			indicators[:, half, :] = selector_function(
				X[test_idx], tau_hat, alphas, **selector_args
			)

	return indicators


