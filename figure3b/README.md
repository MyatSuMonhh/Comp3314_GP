# Figure 3B: sparse-feature SHAP experiment

This folder contains a self-contained conceptual reproduction of Figure 3B from Lundberg and Lee (2017). The experiment explains one prediction from a 100-input decision tree that uses only three active features. It compares:

- Kernel SHAP-style weighted regression with sparse feature selection and a debiased refit
- Permutation-based Shapley sampling for one target feature
- A LIME-style local weighted surrogate

The paper describes the dense and sparse experiment designs, the three methods, 200 replicate estimates, and 10th/90th percentile bands. Its main text does not provide enough detail to recover the exact original tree, explained input, random seeds, and all estimator settings. This implementation therefore recreates the experimental structure with a transparent synthetic tree. Do not describe its numerical results as an exact reproduction of the original figure.

## Run

Install the only external dependency and run from this directory:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce_fig3b.py
```

The default run uses 200 repetitions and query budgets of 200, 400, 600, 800, and 1,000. For a quick local run:

```bash
python3 reproduce_fig3b.py --repeats 20 --budgets 100 200 400
```

## Outputs

The script writes files under `results/`:

- `figure3b.svg`: estimates and 10th/90th percentile bands versus model-evaluation budget
- `figure3b_estimates.csv`: summary statistics for each method and budget
- `run_summary.txt`: model setup, exact reference value, and end-budget results

The exact reference is computed by enumerating the coalitions of the three active features. The other 97 features are dummy inputs: the tree ignores them. Each estimator receives all 100 feature positions.

The implementation uses the same independent Bernoulli background for every estimator. Kernel SHAP draws coalitions in proportion to the Shapley kernel and uses a lasso screen followed by a weighted least-squares refit. The sampling estimator averages marginal contributions along random permutations. The LIME-style surrogate uses Bernoulli feature masks, a cosine-distance locality weight, feature screening, and a weighted refit. These are explicit approximations, not byte-for-byte replications of the 2017 software implementations.

## Paper reference

Scott M. Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model Predictions.” NeurIPS 2017. arXiv:1705.07874v2. Figure 3 compares Kernel SHAP (debiased lasso), Shapley sampling, and LIME, plotting one feature's estimates for dense and sparse decision trees with 200 repetitions and 10th/90th percentile ranges.
