# CausalStabSel tests
import unittest
import warnings

import numpy as np

from causalstabsel import causalstabsel
from causalstabsel.cate_estimators import drlearner_ridge


def make_data(n=120, p=8, seed=0, binary_column=False):
	rng = np.random.default_rng(seed)
	X = rng.normal(size=(n, p))
	if binary_column:
		X[:, -1] = rng.integers(0, 2, n)
	t = rng.integers(0, 2, n)
	y = X[:, 0] + 2 * t * X[:, 1] + rng.normal(size=n)
	return X, y, t


fast = dict(B=4, cate_estimator='drlearner_ridge', selector='l1', preselect=False)


class ValidationTests(unittest.TestCase):
	def setUp(self):
		self.X, self.y, self.t = make_data()

	def test_bad_treatment(self):
		bad = {'length': self.t[:-1], 'values': 2 * self.t, 'one arm': np.zeros_like(self.t),
			'missing': np.where(np.arange(120) == 0, np.nan, self.t), 'text': np.array(['a'] * 120)}
		for name, t in bad.items():
			with self.subTest(name), self.assertRaises(ValueError):
				causalstabsel(self.X, self.y, t, **fast)

	def test_column_treatment_accepted(self):
		out = causalstabsel(self.X, self.y, self.t[:, None], **fast, random_state=0)
		self.assertEqual(len(out['efp_scores']), 8)

	def test_bad_arguments(self):
		cases = [dict(cate_estimator='bogus'), dict(propensity_estimator='bogus'), dict(propensity_estimator=1.5),
			dict(target_fp=1, target_fdr=0.1), dict(subsample_size=0), dict(subsample_size=1.5), dict(delta='a'),
			dict(jitter='yes'), dict(smoother=3)]
		for kwargs in cases:
			with self.subTest(kwargs), self.assertRaises(ValueError):
				causalstabsel(self.X, self.y, self.t, **{**fast, **kwargs})

	def test_bad_cate_output(self):
		def wrong_shape(x_train, y_train, t_train, x_test, propensity_estimator, cate_args):
			return np.zeros(len(x_test) + 1)

		def not_finite(x_train, y_train, t_train, x_test, propensity_estimator, cate_args):
			return np.full(len(x_test), np.nan)

		for estimator in (wrong_shape, not_finite):
			with self.subTest(estimator.__name__), self.assertRaisesRegex(ValueError, 'CATE estimator'):
				causalstabsel(self.X, self.y, self.t, **{**fast, 'cate_estimator': estimator})

	def test_bad_propensity_output(self):
		with self.assertRaisesRegex(ValueError, 'propensity'):
			causalstabsel(self.X, self.y, self.t, **fast, propensity_estimator=lambda a, b, c: np.full(len(c), 2.0))


class SubsampleTests(unittest.TestCase):
	def test_equal_halves_with_odd_arms(self):
		X, y, t = make_data(n=41)
		t[:] = 0
		t[:11] = 1
		sizes = []

		def recording(x_train, y_train, t_train, x_test, propensity_estimator, cate_args):
			sizes.append((len(x_test), int(41 - len(x_train))))
			return drlearner_ridge(x_train, y_train, t_train, x_test, propensity_estimator, cate_args)

		causalstabsel(X, y, t, **{**fast, 'cate_estimator': recording}, random_state=0)
		self.assertTrue(all(size == (20, 20) for size in sizes))


class RandomnessTests(unittest.TestCase):
	def setUp(self):
		self.X, self.y, self.t = make_data(binary_column=True)

	def test_reproducible_across_n_jobs(self):
		a = causalstabsel(self.X, self.y, self.t, B=6, random_state=1)
		b = causalstabsel(self.X, self.y, self.t, B=6, random_state=1, n_jobs=2)
		c = causalstabsel(self.X, self.y, self.t, B=6, random_state=2)
		self.assertEqual(a['efp_scores'], b['efp_scores'])
		self.assertNotEqual(a['efp_scores'], c['efp_scores'])

	def test_warning_filters_restored(self):
		before = list(warnings.filters)
		causalstabsel(self.X, self.y, self.t, **fast, random_state=0)
		self.assertEqual(list(warnings.filters), before)

	def test_jitter_passed_through(self):
		default = causalstabsel(self.X, self.y, self.t, B=6, random_state=1)
		off = causalstabsel(self.X, self.y, self.t, B=6, random_state=1, jitter=False)
		on = causalstabsel(self.X, self.y, self.t, B=6, random_state=1, jitter=True)
		self.assertEqual(default['efp_scores'], on['efp_scores'])
		self.assertNotEqual(default['efp_scores'], off['efp_scores'])

	def test_default_delta(self):
		a = causalstabsel(self.X, self.y, self.t, **fast, random_state=0)
		b = causalstabsel(self.X, self.y, self.t, **fast, random_state=0, delta=1.5)
		self.assertEqual(a['efp_scores'], b['efp_scores'])


class ReviewFixTests(unittest.TestCase):
	def setUp(self):
		self.X, self.y, self.t = make_data()

	def test_bad_arguments_fail_before_fitting(self):
		bad = [{'cutoff': 0}, {'cutoff': 'a'}, {'smooth_cate': 'no'}, {'cate_args': [1]}, {'smoother_args': 'a'},
			{'selector_args': 3}]
		for kwargs in bad:
			with self.assertRaises(ValueError):
				causalstabsel(self.X, self.y, self.t, **{**fast, **kwargs})

	def test_selector_aliases_smooth_alike(self):
		for public, internal in (('l1', 'lasso'), ('adaptive_lasso', 'adaptive_lasso_regressor')):
			a = causalstabsel(self.X, self.y, self.t, **{**fast, 'selector': public}, random_state=0)
			b = causalstabsel(self.X, self.y, self.t, **{**fast, 'selector': internal}, random_state=0)
			self.assertEqual(a['efp_scores'], b['efp_scores'])

	def test_null_result_for_list_input(self):
		X = [[1.0] * 8 for _ in range(120)]
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			out = causalstabsel(X, self.y, self.t, **fast)
		self.assertEqual(len(out['efp_scores']), 8)

	def test_numpy_constant_propensity(self):
		a = causalstabsel(self.X, self.y, self.t, propensity_estimator=np.int64(1), random_state=0, **fast)
		b = causalstabsel(self.X, self.y, self.t, propensity_estimator=1, random_state=0, **fast)
		self.assertEqual(a['efp_scores'], b['efp_scores'])

	def test_ridge_uses_cate_args(self):
		x_train, y_train, t_train = self.X[:80], self.y[:80], self.t[:80]
		args = (x_train, y_train, t_train, self.X[80:], 0.5)
		self.assertFalse(np.allclose(drlearner_ridge(*args), drlearner_ridge(*args, cate_args={'alpha': 1e4})))
		with self.assertRaises(TypeError):
			drlearner_ridge(*args, cate_args={'nonsense': 1})


if __name__ == '__main__':
	unittest.main()
