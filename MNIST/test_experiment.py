"""Meaningful protocol checks. Run: python -m unittest -v."""
import unittest
import numpy as np
from mnist_experiment import (select_pixels, mask_pixels, delta_log_odds,
                              coalition_predictor, explain_kernel, explain_lime)


class MaskingTests(unittest.TestCase):
    def test_positive_difference_and_global_budget(self):
        x = np.ones((28, 28))
        a = np.arange(784, dtype=float)
        b = np.full(784, 500.)
        indices = select_pixels(x, a, b)
        self.assertEqual(len(indices), 157)
        np.testing.assert_array_equal(indices, np.arange(783, 626, -1))
        masked = mask_pixels(x, indices)
        self.assertEqual(np.count_nonzero(masked == 0), 157)
        self.assertTrue(np.all(x == 1))

    def test_fewer_than_budget_and_absent_pixels(self):
        x = np.array([1., 1., 1., 0.])
        indices = select_pixels(x, [5, -4, 0, 100], [0, 0, 0, 0], fraction=1)
        np.testing.assert_array_equal(indices, [0])

    def test_ties_are_deterministic(self):
        indices = select_pixels(np.ones(10), np.ones(10), np.zeros(10), .2)
        np.testing.assert_array_equal(indices, [0, 1])

    def test_log_odds_direction_and_softmax_invariance(self):
        before = np.zeros(10)
        after = np.zeros(10)
        before[8], before[3] = 10, 2
        after[8], after[3] = 1, 4
        self.assertEqual(delta_log_odds(before, after), 11.)
        self.assertEqual(delta_log_odds(before + 1000, after - 1000), 11.)
        def ratio(z):
            p = np.exp(z - z.max()); p /= p.sum()
            return np.log(p[3] / p[8])
        self.assertAlmostEqual(delta_log_odds(before, after), ratio(after) - ratio(before))

    def test_coalition_mapping_preserves_original_pixel_values(self):
        x = np.zeros((28, 28))
        x.flat[[5, 200]] = [.2, .8]
        class RecordingPredictor:
            def probabilities(self, a):
                self.seen = a
                return np.tile(np.arange(10), (len(a), 1))
        p = RecordingPredictor()
        active, f = coalition_predictor(x, p, [8, 3])
        result = f(np.array([[0, 0], [1, 0], [1, 1]]))
        np.testing.assert_array_equal(active, [5, 200])
        np.testing.assert_allclose(p.seen[:, 5], [0, .2, .2])
        np.testing.assert_allclose(p.seen[:, 200], [0, 0, .8])
        np.testing.assert_array_equal(result, [[8, 3]] * 3)


class ExplainerTests(unittest.TestCase):
    def test_exact_kernel_shap_and_lime_on_known_linear_game(self):
        weights = np.array([[.1, -.2], [.3, .4], [-.1, .1], [.2, -.1]])
        baseline = np.array([.2, .1])
        f = lambda masks: np.asarray(masks) @ weights + baseline
        scores, diagnostics = explain_kernel(f, 4, 100, 7)
        np.testing.assert_allclose(scores, weights, atol=1e-10)
        self.assertLess(diagnostics['local_accuracy_max_abs_error'], 1e-10)
        scores, _ = explain_lime(f, 4, 1000, 7, ridge_alpha=0.)
        np.testing.assert_allclose(scores, weights, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
