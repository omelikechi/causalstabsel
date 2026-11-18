# Functions for estimating conditional average treatment effects (CATEs)

from ipss.helpers import is_number, validate_scores
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb

# XGBoost models with a new random seed for each fit (XGBoost otherwise always uses seed 0)
def xgb_regressor(**kwargs):
	if kwargs.get('random_state') is None:
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	return xgb.XGBRegressor(**kwargs)

def xgb_classifier(**kwargs):
	if kwargs.get('random_state') is None:
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	return xgb.XGBClassifier(**kwargs)

# random forests likewise get an explicit seed for each fit
def rf_regressor(**kwargs):
	if kwargs.get('random_state') is None:
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	return RandomForestRegressor(**kwargs)

def rf_classifier(**kwargs):
	if kwargs.get('random_state') is None:
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	return RandomForestClassifier(**kwargs)

################################################################

# Conditional average treatment effect (CATE) estimators

################################################################

#--------------------------------
# Causal forest
#--------------------------------
def causal_forest(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	from econml.dml import CausalForestDML
	cate_args = {'model_y':'linear', 'model_t':'linear', 'max_depth':3, 'n_estimators':24, **(cate_args or {})}
	if cate_args.get('random_state') is None:
		cate_args = {**cate_args, 'random_state': np.random.randint(np.iinfo(np.int32).max)}
	cf = CausalForestDML(discrete_treatment=True, **cate_args)
	cf.fit(y_train, t_train, X=x_train)
	return cf.effect(x_test)

#--------------------------------
# DR-learners
#--------------------------------
def drlearner_gb(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'max_depth':3, 'colsample_bynode':1/3, 'n_estimators':25, **(cate_args or {})}
	ps_estimator = propensity_estimator if propensity_estimator is not None else 'logistic_regression'
	e_hat = np.clip(estimate_propensity_scores(x_train, t_train, x_train, ps_estimator), 0.05, 0.95)
	mu1_model = xgb_regressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = xgb_regressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	mu1_hat = mu1_model.predict(x_train)
	mu0_hat = mu0_model.predict(x_train)
	gamma = (mu1_hat - mu0_hat
		+ t_train * (y_train - mu1_hat) / e_hat
		- (1 - t_train) * (y_train - mu0_hat) / (1 - e_hat))
	return xgb_regressor(**cate_args).fit(x_train, gamma).predict(x_test)

def drlearner_rf(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'n_estimators':100, 'max_features':'sqrt', 'min_samples_leaf':5, **(cate_args or {})}
	ps_estimator = propensity_estimator if propensity_estimator is not None else 'logistic_regression'
	e_hat = np.clip(estimate_propensity_scores(x_train, t_train, x_train, ps_estimator), 0.05, 0.95)
	mu1_model = rf_regressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = rf_regressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	mu1_hat = mu1_model.predict(x_train)
	mu0_hat = mu0_model.predict(x_train)
	gamma = (mu1_hat - mu0_hat
		+ t_train * (y_train - mu1_hat) / e_hat
		- (1 - t_train) * (y_train - mu0_hat) / (1 - e_hat))
	return rf_regressor(**cate_args).fit(x_train, gamma).predict(x_test)

def drlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = cate_args or {}
	ps_estimator = propensity_estimator if propensity_estimator is not None else 'logistic_regression'
	e_hat = np.clip(estimate_propensity_scores(x_train, t_train, x_train, ps_estimator), 0.05, 0.95)
	mu1_model = Ridge(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = Ridge(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	mu1_hat = mu1_model.predict(x_train)
	mu0_hat = mu0_model.predict(x_train)
	gamma = (mu1_hat - mu0_hat
		+ t_train * (y_train - mu1_hat) / e_hat
		- (1 - t_train) * (y_train - mu0_hat) / (1 - e_hat))
	return Ridge(**cate_args).fit(x_train, gamma).predict(x_test)

#--------------------------------
# T-learners
#--------------------------------
def tlearner_gb(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'max_depth':3, 'colsample_bynode':1/3, 'n_estimators':25, **(cate_args or {})}
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1]
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0]
	mu1_model = xgb_regressor(**cate_args).fit(X_treat, y_treat)
	mu0_model = xgb_regressor(**cate_args).fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

def tlearner_rf(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'n_estimators':100, 'max_features':'sqrt', 'min_samples_leaf':5, **(cate_args or {})}
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1].astype(float)
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0].astype(float)
	mu1_model = rf_regressor(**cate_args).fit(X_treat, y_treat)
	mu0_model = rf_regressor(**cate_args).fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

def tlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = cate_args or {}
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1]
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0]
	mu1_model = Ridge(**cate_args).fit(X_treat, y_treat)
	mu0_model = Ridge(**cate_args).fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

#--------------------------------
# X-learners
#--------------------------------
def xlearner_gb(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'max_depth':3, 'colsample_bynode':1/3, 'n_estimators':25, **(cate_args or {})}
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = xgb_regressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = xgb_regressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = xgb_regressor(**cate_args).fit(x_train[t_train == 1], d1)
	tau0_model = xgb_regressor(**cate_args).fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

def xlearner_rf(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = {'n_estimators':100, 'max_features':'sqrt', 'min_samples_leaf':5, **(cate_args or {})}
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = rf_regressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = rf_regressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = rf_regressor(**cate_args).fit(x_train[t_train == 1], d1)
	tau0_model = rf_regressor(**cate_args).fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

def xlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	cate_args = cate_args or {}
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = Ridge(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = Ridge(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = Ridge(**cate_args).fit(x_train[t_train == 1], d1)
	tau0_model = Ridge(**cate_args).fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

# estimate propensity scores
def estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator=None):
	if propensity_estimator is None:
		return np.full(len(x_test), 0.5)
	if is_number(propensity_estimator):
		if 0 <= propensity_estimator <= 1:
			return np.full(len(x_test), float(propensity_estimator))
		else:
			raise ValueError('Propensity constant must be in [0,1].')
	if propensity_estimator == 'logistic_regression':
		model = LogisticRegression().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if propensity_estimator == 'rf':
		model = rf_classifier().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if propensity_estimator in ['gb', 'xgb']:
		model = xgb_classifier().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if callable(propensity_estimator):
		e = validate_scores(propensity_estimator(x_train, t_train, x_test), (len(x_test),), 'the propensity estimator')
		if np.any((e < 0) | (e > 1)):
			raise ValueError('the propensity estimator returned values outside [0, 1].')
		return e
	raise ValueError('Unknown method for estimating propensity.')

# check a propensity estimator before any fitting
def validate_propensity_estimator(propensity_estimator):
	if propensity_estimator is None or callable(propensity_estimator):
		return
	if is_number(propensity_estimator):
		if not 0 <= propensity_estimator <= 1:
			raise ValueError(f'A constant propensity must be in [0, 1], but got {propensity_estimator!r}.')
		return
	if isinstance(propensity_estimator, str) and propensity_estimator in ('logistic_regression', 'rf', 'gb', 'xgb'):
		return
	raise ValueError(f"Unrecognized propensity_estimator: {propensity_estimator!r}. Must be None, 'logistic_regression', "
		"'rf', 'gb', 'xgb', a number in [0, 1], or a custom callable.")

#----------------------------------------------------------------
# All methods
#----------------------------------------------------------------
cate_estimators = {
	'causal_forest': causal_forest,
	'drlearner_gb': drlearner_gb,
	'drlearner_rf': drlearner_rf,
	'drlearner_ridge': drlearner_ridge,
	'tlearner_gb': tlearner_gb,
	'tlearner_rf': tlearner_rf,
	'tlearner_ridge': tlearner_ridge,
	'xlearner_gb': xlearner_gb,
	'xlearner_rf': xlearner_rf,
	'xlearner_ridge': xlearner_ridge
}
