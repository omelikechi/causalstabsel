# Causal IPSS

import time

from ipss.helpers import return_null_result
from ipss.processing import postprocess_ipss, preprocess_ipss
from joblib import Parallel, delayed
import numpy as np

from .cate_estimators import *

#----------------------------------------------------------------
# Causal IPSS
#----------------------------------------------------------------
def causalstabsel(X, y, treatment, cate_estimator='xlearner_gb', selector='gb', selector_args=None, preselect=True, 
		preselector_args=None, target_fp=None, target_fdr=None, B=200, n_alphas=None, ipss_function='h3', cutoff=0.05, 
		delta=None, standardize_X=None, center_y=None, propensity_estimator=None, n_jobs=1):

	# start timer
	start = time.time()

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector_args, 
		B, n_alphas, ipss_function, delta, standardize_X, center_y)

	if prep is None:
		return return_null_result(X.shape[1])
	
	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B = prep['alphas'], prep['n_alphas'], prep['B']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']

	n, p = X.shape

	# estimate selection probabilities
	results = Parallel(n_jobs=n_jobs)(
		delayed(causal_selection)(X, y, treatment, cate_estimator, propensity_estimator, alphas, selector_function, **selector_args)
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

# subsampler for estimating selection probabilities (causal inference)
def causal_selection(X, y, treatment, cate_estimator, propensity_estimator, alphas, selector_function, **kwargs):

	cate_estimators = {
		'causal_forest':causal_forest,
		'tlearner_gb':tlearner_gb,
		'tlearner_rf':tlearner_rf,
		'tlearner_ridge':tlearner_ridge,
		'xlearner_gb':xlearner_gb,
		'xlearner_rf':xlearner_rf,
		'xlearner_ridge':xlearner_ridge
	}
	if isinstance(cate_estimator, str):
		cate_estimator = cate_estimators[cate_estimator]

	n, p = X.shape
	treated_idx = np.where(treatment == 1)[0]
	control_idx = np.where(treatment == 0)[0]

	# stratify samples according to treatment
	np.random.shuffle(treated_idx)
	np.random.shuffle(control_idx)
	n_treat_split = len(treated_idx) // 2
	n_control_split = len(control_idx) // 2

	half1 = np.concatenate([treated_idx[:n_treat_split], control_idx[:n_control_split]])
	half2 = np.concatenate([treated_idx[n_treat_split:], control_idx[n_control_split:]])
	halves = [half1, half2]

	if alphas is None:
		indicators = np.empty((2, p))
	else:
		indicators = np.empty((len(alphas), 2, p))

	for half, (train_idx, test_idx) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
		tau_hat = cate_estimator(X[train_idx, :], y[train_idx], treatment[train_idx], X[test_idx, :], propensity_estimator)

		if alphas is None:
			indicators[half, :] = np.array(selector_function(X[test_idx, :], tau_hat, **kwargs))
		else:
			indicators[:, half, :] = np.array(selector_function(X[test_idx, :], tau_hat, alphas, **kwargs))

	return indicators











