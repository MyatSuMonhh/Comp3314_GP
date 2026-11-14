#!/usr/bin/env python3
"""Conceptual reproduction of Figure 3B with a sparse 100-input tree."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from statistics import mean

import numpy as np

N_FEATURES = 100
ACTIVE = (0, 1, 2)
TARGET = 0
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def tree_predict(row: np.ndarray) -> float:
    """A small depth-3 decision tree using only features 0, 1, and 2."""
    a, b, c = (int(row[j] >= 0.5) for j in ACTIVE)
    # Each path uses the three active binary features.  The remaining 97
    # columns are deliberately ignored.
    leaves = {
        (0, 0, 0): 0.00, (0, 0, 1): 0.40,
        (0, 1, 0): 0.20, (0, 1, 1): 0.70,
        (1, 0, 0): 0.30, (1, 0, 1): 0.90,
        (1, 1, 0): 0.80, (1, 1, 1): 1.70,
    }
    return leaves[(a, b, c)]


def value_function(mask: int, x_active=(1, 1, 1)) -> float:
    """E[f(X) | features in mask fixed to x], with independent Bernoulli(0.5) background."""
    fixed = [j for j in ACTIVE if (mask >> j) & 1]
    missing = [j for j in ACTIVE if j not in fixed]
    outputs = []
    for bits in itertools.product((0, 1), repeat=len(missing)):
        active = [0, 0, 0]
        for j in fixed:
            active[j] = x_active[j]
        for j, bit in zip(missing, bits):
            active[j] = bit
        row = np.zeros(N_FEATURES, dtype=float)
        row[:3] = active
        outputs.append(tree_predict(row))
    return float(np.mean(outputs))


def exact_shapley(target=TARGET) -> tuple[float, float, float, float]:
    """Exact reference by enumerating the 2^3 active-feature coalitions."""
    m = len(ACTIVE)
    phis = []
    for feature in ACTIVE:
        phi = 0.0
        others = [j for j in ACTIVE if j != feature]
        for size in range(len(others) + 1):
            weight = (math.factorial(size) * math.factorial(m - size - 1)
                      / math.factorial(m))
            for subset in itertools.combinations(others, size):
                mask = sum(1 << j for j in subset)
                phi += weight * (value_function(mask | (1 << feature))
                                 - value_function(mask))
        phis.append(phi)
    baseline = value_function(0)
    prediction = value_function(sum(1 << j for j in ACTIVE))
    return float(phis[ACTIVE.index(target)]), baseline, prediction, float(sum(phis))


def sample_kernel_masks(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample coalitions with probability proportional to the SHAP kernel."""
    sizes = np.arange(1, N_FEATURES)
    size_probs = 1.0 / (sizes * (N_FEATURES - sizes))
    size_probs /= size_probs.sum()
    masks = np.zeros((n, N_FEATURES), dtype=float)
    for i, size in enumerate(rng.choice(sizes, size=n, p=size_probs)):
        chosen = rng.choice(N_FEATURES, size=int(size), replace=False)
        masks[i, chosen] = 1.0
    return masks


def mask_to_int(row: np.ndarray) -> int:
    """Only active features can affect the synthetic tree."""
    return sum((1 << j) for j in ACTIVE if row[j] >= 0.5)


def evaluate_masks(masks: np.ndarray) -> np.ndarray:
    return np.fromiter((value_function(mask_to_int(row)) for row in masks),
                       dtype=float, count=len(masks))


def soft_threshold(value: float, alpha: float) -> float:
    if value > alpha:
        return value - alpha
    if value < -alpha:
        return value + alpha
    return 0.0


def lasso_cd(X: np.ndarray, y: np.ndarray, alpha: float,
             max_iter: int = 80) -> np.ndarray:
    """Small coordinate-descent lasso; intercept is fixed by the SHAP baseline."""
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    residual = y.copy()
    norms = np.sum(X * X, axis=0) / max(n, 1)
    for _ in range(max_iter):
        largest = 0.0
        for j in range(p):
            if norms[j] < 1e-12:
                continue
            residual += X[:, j] * beta[j]
            rho = float(X[:, j] @ residual) / n
            updated = soft_threshold(rho, alpha) / norms[j]
            residual -= X[:, j] * updated
            largest = max(largest, abs(updated - beta[j]))
            beta[j] = updated
        if largest < 1e-5:
            break
    return beta


def weighted_refit(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.maximum(weights, 0.0))
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=1e-9)[0]


def constrained_refit(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                      total_contribution: float) -> np.ndarray:
    """Weighted least squares with exact local accuracy: sum(beta) = delta f."""
    gram = X.T @ (weights[:, None] * X)
    rhs = X.T @ (weights * y)
    gram = gram + np.eye(gram.shape[0]) * 1e-9
    ones = np.ones(X.shape[1], dtype=float)
    beta0 = np.linalg.solve(gram, rhs)
    direction = np.linalg.solve(gram, ones)
    denominator = float(ones @ direction)
    if abs(denominator) < 1e-12:
        beta0 += (total_contribution - float(beta0.sum())) / len(beta0)
        return beta0
    return beta0 - direction * ((float(beta0.sum()) - total_contribution) / denominator)


def screened_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                 target: int, baseline: float, prediction: float,
                 top_k: int = 10, lasso_alpha: float = 0.025) -> float:
    """Screen, lasso-select, then refit selected coefficients with local accuracy."""
    centered_x = X - np.average(X, axis=0, weights=weights)
    centered_y = y - np.average(y, weights=weights)
    denom = np.sqrt(np.sum(weights[:, None] * centered_x ** 2, axis=0) + 1e-12)
    scores = np.abs((weights[:, None] * centered_x).T @ centered_y) / denom
    candidate = np.argsort(scores)[-min(top_k, X.shape[1]):]
    Xc = X[:, candidate]
    # Center around the exact empty-coalition value, as in the SHAP additive form.
    yc = y - baseline
    lasso = lasso_cd(Xc, yc, lasso_alpha)
    selected_local = np.flatnonzero(np.abs(lasso) > 1e-5)
    if len(selected_local) == 0:
        selected_local = np.array([int(np.argmax(np.abs(lasso)))])
    selected = candidate[selected_local]
    coefs = constrained_refit(X[:, selected], yc, weights, prediction - baseline)
    where = np.flatnonzero(selected == target)
    return float(coefs[where[0]]) if len(where) else 0.0


def kernel_estimate(rng: np.random.Generator, budget: int,
                    baseline: float, prediction: float) -> float:
    masks = sample_kernel_masks(rng, budget)
    y = evaluate_masks(masks)
    # Since masks are sampled proportional to the kernel, equal sample weights
    # estimate the kernel-weighted regression objective.
    return screened_fit(masks, y, np.ones(budget), TARGET, baseline, prediction)


def shapley_sampling_estimate(rng: np.random.Generator, budget: int) -> float:
    # Each random permutation needs two coalition-value evaluations for the
    # target feature's marginal contribution.
    n_pairs = max(1, budget // 2)
    marginal = np.empty(n_pairs, dtype=float)
    others = np.array([j for j in range(N_FEATURES) if j != TARGET])
    for k in range(n_pairs):
        order = rng.permutation(others)
        # Every feature before the target in the permutation forms S.
        rank = int(rng.integers(0, N_FEATURES))
        subset = order[:rank]
        mask = sum(1 << int(j) for j in subset)
        marginal[k] = value_function(mask | (1 << TARGET)) - value_function(mask)
    return float(marginal.mean())


def lime_estimate(rng: np.random.Generator, budget: int,
                  baseline: float, prediction: float) -> float:
    # LIME-style binary perturbations around the all-present interpretable input.
    masks = rng.binomial(1, 0.5, size=(budget, N_FEATURES)).astype(float)
    y = evaluate_masks(masks)
    cosine_distance = 1.0 - np.sqrt(np.maximum(masks.mean(axis=1), 1e-12))
    width = 0.75
    weights = np.exp(-(cosine_distance ** 2) / (width ** 2))

    # Local weighted feature selection: rank single-feature associations, retain
    # at most 10, then fit a weighted linear surrogate with an intercept.
    mx = np.average(masks, axis=0, weights=weights)
    my = np.average(y, weights=weights)
    xc, yc = masks - mx, y - my
    scores = np.abs((weights[:, None] * xc).T @ yc)
    candidates = np.argsort(scores)[-min(10, N_FEATURES):]
    design = np.column_stack([np.ones(budget), masks[:, candidates]])
    coefs = weighted_refit(design, y, weights)
    where = np.flatnonzero(candidates == TARGET)
    return float(coefs[1 + where[0]]) if len(where) else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def write_svg(path: Path, summaries: dict, budgets: list[int],
              exact: float) -> None:
    width, height = 980, 600
    left, right, top, bottom = 92, 36, 48, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin = min(-0.2, exact - 0.2)
    ymax = max(1.0, exact + 0.2)
    colors = {'Kernel SHAP (sparse WLS)': '#167D79',
              'Shapley sampling': '#D9476D',
              'LIME-style surrogate': '#3579B9'}
    def px(x): return left + (x - min(budgets)) / max(1, (max(budgets)-min(budgets))) * plot_w
    def py(y): return top + (ymax-y)/(ymax-ymin)*plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#243247}.small{font-size:13px}.label{font-size:16px}.title{font-size:22px;font-weight:bold}</style>',
             f'<text x="{left}" y="28" class="title">Figure 3B conceptual reproduction: sparse 100-input tree</text>']
    for tick in np.linspace(ymin, ymax, 6):
        parts.append(f'<line x1="{left}" y1="{py(tick):.2f}" x2="{left+plot_w}" y2="{py(tick):.2f}" stroke="#e4e8ee"/>')
        parts.append(f'<text x="{left-12}" y="{py(tick)+4:.2f}" text-anchor="end" class="small">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{py(exact):.2f}" x2="{left+plot_w}" y2="{py(exact):.2f}" stroke="#475569" stroke-dasharray="7 5" stroke-width="2"/>')
    parts.append(f'<text x="{left+plot_w-4}" y="{py(exact)-8:.2f}" text-anchor="end" class="small">Exact SHAP = {exact:.3f}</text>')
    for method,color in colors.items():
        stats=[summaries[(method,b)] for b in budgets]
        upper=' '.join(f'{px(b):.2f},{py(z["p90"]):.2f}' for b,z in zip(budgets,stats))
        lower=' '.join(f'{px(b):.2f},{py(z["p10"]):.2f}' for b,z in reversed(list(zip(budgets,stats))))
        parts.append(f'<polygon points="{upper} {lower}" fill="{color}" opacity="0.14"/>')
        points=' '.join(f'{px(b):.2f},{py(z["median"]):.2f}' for b,z in zip(budgets,stats))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for b,z in zip(budgets,stats):
            parts.append(f'<circle cx="{px(b):.2f}" cy="{py(z["median"]):.2f}" r="4" fill="{color}"/>')
    for b in budgets:
        parts.append(f'<line x1="{px(b):.2f}" y1="{top+plot_h}" x2="{px(b):.2f}" y2="{top+plot_h+5}" stroke="#526176"/>')
        parts.append(f'<text x="{px(b):.2f}" y="{top+plot_h+26}" text-anchor="middle" class="small">{b}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#526176"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#526176"/>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" class="label">Model evaluations (coalition-value queries)</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" class="label">Estimated SHAP value for feature 0</text>')
    ly=top+plot_h+52
    x=left
    for method,color in colors.items():
        parts.append(f'<line x1="{x}" y1="{ly}" x2="{x+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+30}" y="{ly+5}" class="small">{method}</text>')
        x+=250
    parts.append(f'<text x="{left}" y="{height-5}" class="small">Shaded bands: 10th–90th percentile across repetitions. Synthetic tree, not original paper data.</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--budgets', type=int, nargs='+', default=[200, 400, 600, 800, 1000])
    parser.add_argument('--seed', type=int, default=3314)
    args = parser.parse_args()
    if args.repeats < 1 or any(b < 20 for b in args.budgets):
        parser.error('repeats must be positive and every budget must be at least 20')
    budgets = sorted(set(args.budgets))
    exact, baseline, prediction, summed = exact_shapley()
    methods = {
        'Kernel SHAP (sparse WLS)': kernel_estimate,
        'Shapley sampling': shapley_sampling_estimate,
        'LIME-style surrogate': lime_estimate,
    }
    raw = {(name,b): [] for name in methods for b in budgets}
    master = np.random.SeedSequence(args.seed)
    children = master.spawn(args.repeats)
    for r, child in enumerate(children):
        method_seeds = child.spawn(len(methods) * len(budgets))
        si = 0
        for name, fn in methods.items():
            for budget in budgets:
                rng = np.random.default_rng(method_seeds[si]); si += 1
                if name == 'Kernel SHAP (sparse WLS)':
                    val = fn(rng, budget, baseline, prediction)
                elif name == 'Shapley sampling':
                    val = fn(rng, budget)
                else:
                    val = fn(rng, budget, baseline, prediction)
                raw[(name,budget)].append(float(val))
        if (r+1) % max(1, args.repeats//10) == 0 or r+1 == args.repeats:
            print(f'Completed {r+1}/{args.repeats} repetitions')

    summaries={}
    for key, vals in raw.items():
        summaries[key]={'mean':mean(vals),'median':percentile(vals,50),
                        'p10':percentile(vals,10),'p90':percentile(vals,90),
                        'rmse':float(np.sqrt(np.mean((np.asarray(vals)-exact)**2)))}
    RESULTS.mkdir(parents=True,exist_ok=True)
    csv_path=RESULTS/'figure3b_estimates.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['method','budget_queries','repeats','mean','median','p10','p90','rmse_vs_exact'])
        for method in methods:
            for budget in budgets:
                s=summaries[(method,budget)]
                writer.writerow([method,budget,args.repeats,*[f'{s[k]:.8f}' for k in ('mean','median','p10','p90','rmse')]])
    write_svg(RESULTS/'figure3b.svg',summaries,budgets,exact)
    with (RESULTS/'run_summary.txt').open('w',encoding='utf-8') as f:
        f.write('Figure 3B conceptual reproduction\n')
        f.write('Tree uses active features 0, 1, 2; features 3-99 are unused dummies.\n')
        f.write('All features are independently Bernoulli(0.5); explained active input is (1,1,1).\n')
        f.write(f'Baseline: {baseline:.8f}\nPrediction: {prediction:.8f}\n')
        f.write('Active-feature SHAP values: ' + ', '.join(f'{v:.8f}' for v in (
            exact_shapley(0)[0], exact_shapley(1)[0], exact_shapley(2)[0])) + '\n')
        f.write(f'Target feature: {TARGET}; exact SHAP reference: {exact:.8f}\n')
        f.write(f'Attribution sum: {summed:.8f}; prediction minus baseline: {prediction-baseline:.8f}\n')
        f.write(f'Repetitions: {args.repeats}; budgets: {budgets}\n\n')
        for method in methods:
            s=summaries[(method,budgets[-1])]
            f.write(f'{method} at budget {budgets[-1]}: median={s["median"]:.6f}, p10={s["p10"]:.6f}, p90={s["p90"]:.6f}, RMSE={s["rmse"]:.6f}\n')
    print(f'Exact target SHAP: {exact:.6f}; baseline={baseline:.6f}; prediction={prediction:.6f}')
    print(f'Wrote {csv_path}, {RESULTS / "figure3b.svg"}, and {RESULTS / "run_summary.txt"}')


if __name__ == '__main__':
    main()
