# Functions for estimating conditional average treatment effects (CATEs)

from econml.dml import CausalForestDML
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb

#----------------------------------------------------------------
# Conditional average treatment effect (CATE) estimators
#----------------------------------------------------------------
def causal_forest(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	if cate_args is None:
		cate_args = {'model_y':'linear', 'model_t':'linear', 'max_depth':3, 'n_estimators':24}
	cf = CausalForestDML(discrete_treatment=True, **cate_args)
	cf.fit(y_train, t_train, X=x_train)
	return cf.effect(x_test)

def tlearner_gb(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	if cate_args is None:
		# cate_args = {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':50}
		cate_args = {}
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1]
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0]
	mu1_model = xgb.XGBRegressor(**cate_args).fit(X_treat, y_treat)
	mu0_model = xgb.XGBRegressor(**cate_args).fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

def tlearner_rf(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	if cate_args is None:
		cate_args = {'n_estimators':100, 'max_features':1/3}
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1].astype(float)
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0].astype(float)
	mu1_model = RandomForestRegressor(**cate_args).fit(X_treat, y_treat)
	mu0_model = RandomForestRegressor(**cate_args).fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

def tlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	X_treat, y_treat = x_train[t_train == 1], y_train[t_train == 1]
	X_ctrl, y_ctrl = x_train[t_train == 0], y_train[t_train == 0]
	mu1_model = Ridge().fit(X_treat, y_treat)
	mu0_model = Ridge().fit(X_ctrl, y_ctrl)
	mu1 = mu1_model.predict(x_test)
	mu0 = mu0_model.predict(x_test)
	return mu1 - mu0

def xlearner_gb(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	if cate_args is None:
		cate_args = {'max_depth':3, 'colsample_bynode':1/3, 'n_estimators':25}
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = xgb.XGBRegressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = xgb.XGBRegressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = xgb.XGBRegressor(**cate_args).fit(x_train[t_train == 1], d1)
	tau0_model = xgb.XGBRegressor(**cate_args).fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

def xlearner_rf(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	if cate_args is None:
		cate_args = {'n_estimators':100, 'max_features':'sqrt', 'min_samples_leaf':5}
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = RandomForestRegressor(**cate_args).fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = RandomForestRegressor(**cate_args).fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = RandomForestRegressor(**cate_args).fit(x_train[t_train == 1], d1)
	tau0_model = RandomForestRegressor(**cate_args).fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

def xlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator=None, cate_args=None):
	w = estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator)
	mu1_model = Ridge().fit(x_train[t_train == 1], y_train[t_train == 1])
	mu0_model = Ridge().fit(x_train[t_train == 0], y_train[t_train == 0])
	d1 = y_train[t_train == 1] - mu0_model.predict(x_train[t_train == 1])
	d0 = mu1_model.predict(x_train[t_train == 0]) - y_train[t_train == 0]
	tau1_model = Ridge().fit(x_train[t_train == 1], d1)
	tau0_model = Ridge().fit(x_train[t_train == 0], d0)
	tau1 = tau1_model.predict(x_test)
	tau0 = tau0_model.predict(x_test)
	return w * tau0 + (1 - w) * tau1

# estimate propensity scores
def estimate_propensity_scores(x_train, t_train, x_test, propensity_estimator=None):
	if propensity_estimator is None:
		return np.full(len(x_test), 0.5)
	if isinstance(propensity_estimator, (int, float)):
		if 0 <= propensity_estimator <= 1:
			return np.full(len(x_test), float(propensity_estimator))
		else:
			raise ValueError('Propensity constant must be in [0,1].')
	if propensity_estimator == 'logistic_regression':
		model = LogisticRegression().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if propensity_estimator == 'rf':
		model = RandomForestClassifier().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if propensity_estimator in ['gb', 'xgb']:
		model = xgb.XGBClassifier().fit(x_train, t_train)
		return model.predict_proba(x_test)[:,1]
	if callable(propensity_estimator):
		return propensity_estimator(x_train, t_train, x_test)
	raise ValueError('Unknown method for estimating propensity.')





	