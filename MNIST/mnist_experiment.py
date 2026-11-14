"""Partial reproduction of Lundberg & Lee (2017), Figure 5.

Original experiment orchestration, CNN, masking, metric, plots and tests.
Attribution algorithms reuse SHAP KernelExplainer and LIME LimeBase.
See README.md for protocol differences and source acknowledgements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".cache" / "numba"))

import numpy as np
import torch
from torch import nn

ARCHITECTURE = "conv16-pool-conv32-pool-dense128-dense10-v1"
PROFILES = {"smoke": (2, 512), "quick": (20, 5000), "paper-budget": (20, 50000)}
PAPER = "https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No result rows to write")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DigitCNN(nn.Module):
    """Two convolutions and two dense layers; newly trained, not paper weights."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_mnist(root):
    from torchvision.datasets import MNIST
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    return (train.data[:, None].float() / 255, train.targets,
            test.data[:, None].float() / 255, test.targets)


class Predictor:
    def __init__(self, model, device, batch_size=256):
        self.model = model.eval()
        self.device = device
        self.batch_size = batch_size
        self.evaluations = 0

    def logits(self, images):
        a = np.asarray(images, dtype=np.float32).reshape(-1, 1, 28, 28)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(a), self.batch_size):
                x = torch.from_numpy(a[i:i+self.batch_size]).to(self.device)
                chunks.append(self.model(x).cpu().numpy())
        self.evaluations += len(a)
        return np.concatenate(chunks)

    def probabilities(self, images):
        a = self.logits(images).astype(np.float64)
        a -= a.max(axis=1, keepdims=True)
        a = np.exp(a)
        return a / a.sum(axis=1, keepdims=True)


def train_model(args):
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"{checkpoint} exists; use --overwrite to replace it")
    seed_all(args.seed)
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    x, y, xt, yt = load_mnist(args.data)
    order = torch.randperm(len(x))
    val_idx, fit_idx = order[:5000], order[5000:]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x[fit_idx], y[fit_idx]),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best, best_state = [], -1.0, None
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            seen += len(xb)
        pred = Predictor(model, device, args.batch_size)
        val_acc = float((pred.logits(x[val_idx].numpy()).argmax(1) == y[val_idx].numpy()).mean())
        history.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                        "validation_accuracy": val_acc})
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    test_acc = float((Predictor(model, device).logits(xt.numpy()).argmax(1) == yt.numpy()).mean())
    metadata = {"architecture": ARCHITECTURE, "seed": args.seed,
                "epochs": args.epochs, "learning_rate": args.learning_rate,
                "batch_size": args.batch_size, "device": str(device), "threads": args.threads,
                "fit_examples": 55000, "validation_examples": 5000,
                "selection": "highest validation accuracy, first epoch on ties",
                "test_accuracy": test_acc, "history": history,
                "elapsed_seconds": time.perf_counter() - started,
                "torch_version": str(torch.__version__)}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint)
    write_json(checkpoint.with_suffix(".json"), metadata)
    print(f"Saved {checkpoint}; test accuracy={test_acc:.4%}", flush=True)


def coalition_predictor(image, predictor, classes):
    """Only nonzero pixels vary. Other pixels equal their zero reference already."""
    active = np.flatnonzero(image.reshape(-1) != 0)

    def predict(masks):
        masks = np.asarray(masks)
        images = np.zeros((len(masks), 784), dtype=np.float32)
        images[:, active] = masks * image.reshape(-1)[active]
        return predictor.probabilities(images)[:, classes]

    return active, predict


def explain_kernel(predict, m, samples, seed):
    import shap
    np.random.seed(seed)
    explainer = shap.KernelExplainer(predict, np.zeros((1, m)), link="identity")
    values = explainer.shap_values(np.ones((1, m)), nsamples=samples,
                                  l1_reg=0., silent=True)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    scores = np.asarray(values, dtype=float)[0]
    if scores.shape != (m, 2):
        raise ValueError(f"Unexpected Kernel SHAP output shape {scores.shape}")
    residual = (scores.sum(0) + np.asarray(explainer.expected_value)
                - predict(np.ones((1, m)))[0])
    return scores, {"local_accuracy_max_abs_error": float(np.abs(residual).max()),
                    "actual_coalitions": int(explainer.nsamplesRun),
                    "link": "identity", "l1_reg": 0.}


def explain_lime(predict, m, samples, seed, kernel_width=.25, ridge_alpha=1.):
    """Official LIME weighted fit, adapted to binary single-foreground-pixel masks."""
    from lime.lime_base import LimeBase
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(samples, m), dtype=np.int8)
    masks[0] = 1
    outputs = predict(masks)
    # Cosine distance to the all-present image, as in image LIME.
    distances = 1. - np.sqrt(masks.sum(1, dtype=float) / m)
    kernel = lambda d: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    lime = LimeBase(kernel, random_state=seed)
    scores = np.zeros((m, 2))
    diagnostics = {"kernel_width": kernel_width, "ridge_alpha": ridge_alpha,
                   "feature_selection": "none", "weighted_r2": [],
                   "local_accuracy_error": [], "actual_coalitions": samples}
    for j in range(2):
        intercept, explanation, r2, local = lime.explain_instance_with_data(
            masks, outputs, distances, j, m, feature_selection="none",
            model_regressor=Ridge(alpha=ridge_alpha, fit_intercept=True),
        )
        for feature, score in explanation:
            scores[feature, j] = score
        diagnostics["weighted_r2"].append(float(r2))
        diagnostics["local_accuracy_error"].append(float(np.ravel(local)[0] - outputs[0, j]))
    return scores, diagnostics


def select_pixels(image, source_scores, target_scores, fraction=.2):
    """Author's positive-difference rule with a deterministic exact top-k tie break."""
    diff = np.asarray(source_scores).ravel() - np.asarray(target_scores).ravel()
    eligible = np.flatnonzero((diff > 0) & (np.asarray(image).ravel() != 0))
    rank = eligible[np.argsort(-diff[eligible], kind="stable")]
    return rank[:int(round(fraction * np.asarray(image).size))]


def mask_pixels(image, indices):
    out = np.asarray(image).copy()
    out.reshape(-1)[indices] = 0.
    return out


def delta_log_odds(before, after, source=8, target=3):
    return float((after[target] - after[source]) - (before[target] - before[source]))


def outcome(before, after, source, target):
    def prob(a):
        a = np.exp(np.asarray(a, float) - np.max(a))
        return a / a.sum()
    p, q = prob(before), prob(after)
    return {"p_source_before": float(p[source]), "p_target_before": float(p[target]),
            "p_source_after": float(q[source]), "p_target_after": float(q[target]),
            "prediction_before": int(before.argmax()), "prediction_after": int(after.argmax()),
            "delta_log_odds": delta_log_odds(before, after, source, target),
            "target_beats_source": int(after[target] > after[source]),
            "top1_is_target": int(after.argmax() == target)}


def run_experiment(args):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    count, samples = PROFILES[args.profile]
    count = args.images if args.images is not None else count
    samples = args.samples if args.samples is not None else samples
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty. Use a new --output directory.")
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["metadata"]["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    predictor = Predictor(model, device, args.batch_size)
    _, _, xt, yt = load_mnist(args.data)
    source_ids = np.flatnonzero(yt.numpy() == args.source)
    logits = predictor.logits(xt[source_ids].numpy())
    eligible = source_ids[logits.argmax(1) == args.source]
    if count > len(eligible):
        raise ValueError(f"Only {len(eligible)} correctly classified source images")
    ids = np.random.default_rng(args.seed).choice(eligible, count, replace=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "numpy", "scipy", "shap", "lime", "scikit-learn", "matplotlib"]}
    manifest = {"paper": PAPER, "scope": "partial reproduction, newly trained CNN, Kernel SHAP and adapted image LIME",
                "profile": args.profile, "source": args.source, "target": args.target,
                "samples_per_method_per_image": samples, "seed": args.seed,
                "image_ids": ids.tolist(), "image_selection": "random true source class correctly classified as source",
                "mask_fraction_of_entire_image": args.mask_fraction,
                "mask_pixel_cap": round(args.mask_fraction * 784),
                "mask_rule": "descending phi_source-phi_target, strictly positive only; stable pixel-index tie break",
                "background": "one all-zero image", "explanation_output": "two columns of original ten-way probabilities, unrenormalized",
                "lime": {"kernel_width": args.lime_width, "ridge_alpha": 1., "feature_selection": "none",
                         "features": "one binary feature per nonzero pixel", "distance": "cosine to all-present mask"},
                "metric": "delta log(p_target/p_source), computed exactly using pre-softmax logits",
                "random_repeats": args.random_repeats,
                "random_rule": "for each method/image, same actual removed pixel count; choose uniformly among nonzero pixels",
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "training": checkpoint["metadata"], "device": str(device), "threads": args.threads,
                "versions": versions,
                "differences_from_paper": ["new CNN and weights", "new fixed sample of correctly classified 8s",
                  "Original/New DeepLIFT omitted", "LIME adaptation uses official LimeBase with specified modern settings",
                  "exact top-k tie break instead of author's strict threshold", "random matched controls added",
                  "sampling budget may differ, see samples_per_method_per_image"]}
    write_json(out / "manifest.json", manifest)
    rows, random_rows, diagnostics = [], [], []
    for ordinal, idx in enumerate(ids):
        image = xt[idx].numpy().reshape(28, 28)
        active, predict = coalition_predictor(image, predictor, [args.source, args.target])
        before = predictor.logits(image)[0]
        saved = {"image": image, "test_id": np.array(idx)}
        for method, function in [("Kernel SHAP", explain_kernel), ("LIME", explain_lime)]:
            local_seed = args.seed + 1009 * (int(idx) + 1)
            t = time.perf_counter()
            start_evals = predictor.evaluations
            extra = {"kernel_width": args.lime_width} if method == "LIME" else {}
            scores, diag = function(predict, len(active), samples, local_seed, **extra)
            seconds = time.perf_counter() - t
            evals = predictor.evaluations - start_evals
            full = np.zeros((784, 2))
            full[active] = scores
            indices = select_pixels(image, full[:, 0], full[:, 1], args.mask_fraction)
            masked = mask_pixels(image, indices)
            after = predictor.logits(masked)[0]
            row = {"test_id": int(idx), "method": method, "removed_pixels": len(indices),
                   "active_pixels": len(active), "explanation_seed": local_seed,
                   "requested_samples": samples, "model_image_evaluations": evals,
                   "explanation_seconds": seconds, **outcome(before, after, args.source, args.target)}
            rows.append(row)
            diagnostics.append({"test_id": int(idx), "method": method, **diag})
            key = "kernel_shap" if method == "Kernel SHAP" else "lime"
            saved[key + "_scores"] = full.reshape(28, 28, 2)
            saved[key + "_masked"] = masked
            saved[key + "_indices"] = indices
            rng = np.random.default_rng(local_seed + 1)
            random_images = [mask_pixels(image, rng.choice(active, len(indices), replace=False))
                             for _ in range(args.random_repeats)]
            random_logits = predictor.logits(np.stack(random_images))
            controls = []
            for repeat, random_after in enumerate(random_logits):
                control = {"test_id": int(idx), "matched_to": method, "repeat": repeat,
                           "removed_pixels": len(indices), **outcome(before, random_after, args.source, args.target)}
                random_rows.append(control)
                controls.append(control)
            random_row = dict(row, method="Random matched to " + method,
                              model_image_evaluations=0, explanation_seconds=0.)
            random_row["prediction_after"] = -1
            for k in ["p_source_after", "p_target_after", "delta_log_odds", "target_beats_source", "top1_is_target"]:
                random_row[k] = float(np.mean([c[k] for c in controls]))
            rows.append(random_row)
            print(f"[{ordinal+1}/{count}] id={idx} {method}: delta={row['delta_log_odds']:.3f}, "
                  f"removed={len(indices)}, target_top1={bool(row['top1_is_target'])}, {seconds:.1f}s", flush=True)
        np.savez_compressed(out / f"image_{idx}.npz", **saved)
        write_csv(out / "metrics.csv", rows)
        write_csv(out / "random_trials.csv", random_rows)
        write_json(out / "diagnostics.json", diagnostics)
    summarize(out)
    plot_results(out)


def summarize(out):
    out = Path(out)
    with (out / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    summaries = []
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        values = np.array([float(r["delta_log_odds"]) for r in group])
        summaries.append({"method": method, "n_images": len(group),
            "mean_delta_log_odds": float(values.mean()), "median_delta_log_odds": float(np.median(values)),
            "target_beats_source_rate": float(np.mean([float(r["target_beats_source"]) for r in group])),
            "top1_target_rate": float(np.mean([float(r["top1_is_target"]) for r in group])),
            "mean_removed_pixels": float(np.mean([int(r["removed_pixels"]) for r in group])),
            "mean_explanation_seconds": float(np.mean([float(r["explanation_seconds"]) for r in group]))})
    write_json(out / "summary.json", summaries)
    write_csv(out / "summary.csv", summaries)
    return summaries


def plot_results(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    first = manifest["image_ids"][0] 
    data = np.load(out / f"image_{first}.npz")
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.8), layout="constrained")
    for r, (key, name) in enumerate([("kernel_shap", "Kernel SHAP"), ("lime", "LIME")]):
        scores = data[key + "_scores"]
        vmax = max(float(np.abs(scores).max()), 1e-10)
        for c in range(4):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c in (0, 3):
                ax.imshow(data["image"] if c == 0 else data[key + "_masked"], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(data["image"], cmap="gray", vmin=0, vmax=1)
                vals = scores[:, :, c - 1]
                alpha = .85 * np.abs(vals) / vmax
                ax.imshow(vals, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=alpha)
            if r == 0:
                ax.set_title(["Input", f"Explain {manifest['source']}", f"Explain {manifest['target']}", "Masked"][c])
            if c == 0:
                ax.set_ylabel(name)
    fig.suptitle(f"Our partial reproduction: test image {first}\n"
                 "Red = positive, blue = negative attribution; shared scale within each method", fontsize=12)
    fig.savefig(out / "figure5a_partial.png", dpi=200)
    plt.close(fig)
    with (out / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    methods = ["Kernel SHAP", "Random matched to Kernel SHAP", "LIME", "Random matched to LIME"]
    labels = ["Kernel SHAP", "Random\n(Kernel count)", "LIME", "Random\n(LIME count)"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    arrays = [np.array([float(r["delta_log_odds"]) for r in rows if r["method"] == m]) for m in methods]
    ax.boxplot(arrays, tick_labels=labels, showfliers=False)
    rng = np.random.default_rng(0)
    for i, values in enumerate(arrays):
        ax.scatter(i + 1 + rng.uniform(-.08, .08, len(values)), values, s=20, alpha=.65)
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set_ylabel("Change in log[p(target) / p(source)]")
    ax.set_title(f"Our partial reproduction: {manifest['source']} to {manifest['target']}, n={len(arrays[0])}\n"
                 f"{manifest['samples_per_method_per_image']:,} samples per explainer/image")
    fig.savefig(out / "figure5b_partial.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--data", default="data")
        p.add_argument("--checkpoint", default="checkpoints/cnn.pt")
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=256)
    train = sub.add_parser("train", help="Train a CNN with a held-out validation set")
    common(train)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run", help="Run explainers, matched random controls and plot")
    common(run)
    run.add_argument("--profile", choices=PROFILES, default="quick")
    run.add_argument("--images", type=int)
    run.add_argument("--samples", type=int)
    run.add_argument("--output", default="results/quick")
    run.add_argument("--source", type=int, default=8)
    run.add_argument("--target", type=int, default=3)
    run.add_argument("--mask-fraction", type=float, default=.2)
    run.add_argument("--random-repeats", type=int, default=30)
    run.add_argument("--lime-width", type=float, default=.25)
    plot = sub.add_parser("plot", help="Regenerate summaries/figures from saved results")
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        train_model(args)
    elif args.command == "run":
        if not 0 < args.mask_fraction <= 1 or args.random_repeats < 1:
            parser.error("Require 0 < mask-fraction <= 1 and random-repeats >= 1")
        if not 0 <= args.source < 10 or not 0 <= args.target < 10 or args.source == args.target:
            parser.error("Source/target must be distinct digits")
        if (args.images is not None and args.images < 1) or (args.samples is not None and args.samples < 2):
            parser.error("Require images >= 1 and samples >= 2")
        if args.lime_width <= 0:
            parser.error("--lime-width must be positive")
        run_experiment(args)
    else:
        summarize(args.output)
        plot_results(args.output)


if __name__ == "__main__":
    main()
