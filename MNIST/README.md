# MNIST pixel masking: partial reproduction of SHAP Figure 5

COMP3314 group project. This project implements the assigned MNIST experiment
from **Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions**,
NIPS 30, pp. 4765–4774.

**Scope:** a newly trained CNN, Kernel SHAP, adapted single-pixel image LIME,
and matched random masking controls. This is a **partial reproduction**, not
an exact replication of the paper's model, four methods, or numerical values.
No original-paper results are substituted for our own results.

## Setup

Python 3.11 or 3.12 is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

The first training command downloads the official MNIST train/test data using
torchvision. An internet connection is needed for installation and download.
The scripts keep data, model weights, and caches in project-relative folders.
Training and inference run on CPU by default, or CUDA when available.
Pass `--device cpu` for a consistent CPU execution setting.

## Run

Train once, then reuse exactly the same checkpoint for all comparisons:

```bash
python mnist_experiment.py train --epochs 3 --checkpoint checkpoints/cnn.pt
```

Training uses 55,000 examples, a seeded 5,000-example validation split, Adam,
and cross-entropy. It chooses the checkpoint with the best validation accuracy
and evaluates test accuracy only after that choice. Inputs stay in [0, 1].
The architecture has two 3×3 convolutions (16 and 32 channels), each followed by
ReLU and 2×2 max pooling, then dense layers of size 128 and 10. There is no
claim that these layers or weights equal the original pretrained CNN.

Choose a profile:

```bash
python mnist_experiment.py run --profile smoke --output results/smoke

python mnist_experiment.py run --profile quick --output results/quick

python mnist_experiment.py run --profile paper-budget --output results/50k
```

`paper-budget` means 20 images and 50,000 samples per explainer/image. It does
not mean the rest of the original protocol has been fully reproduced. Runtime
depends on hardware and the number of nonzero pixels. The script prints per-image
progress and explanation time. Do not extrapolate performance from the smoke run.

Useful overrides:

```bash
python mnist_experiment.py run --profile quick --images 5 --samples 2000 \
  --seed 2026 --random-repeats 30 --device cpu --threads 4 \
  --output results/custom

python mnist_experiment.py plot --output results/quick
```

The source and target default to 8 and 3 and can be changed using `--source` and
`--target`. A nonempty result directory is rejected to prevent mixing different
runs. Use a new directory. Training similarly rejects an existing checkpoint
unless explicitly given `--overwrite`.

## Experimental protocol

1. Fix the checkpoint and seed. Draw 20 MNIST test images without replacement
   from true 8s that the model correctly predicts as 8. Save these IDs in
   `manifest.json` **before** computing explanations. No success-based filtering.
2. Explain original ten-way softmax probabilities for 8 and 3. The two outputs
   remain unrenormalized: this does not turn the classifier into a binary model.
3. Use one all-zero reference image. Each nonzero pixel is one independently
   toggleable feature. Pixels already equal to the zero reference do not vary.
   The background is fixed, not sampled from training data, and this should not
   be described as exact conditional-expectation SHAP with correlated pixels.
4. Run the official `shap.KernelExplainer`, explicitly setting `link="identity"`
   and `l1_reg=0.0` to avoid silent default top-10 feature selection. Requested
   sample budgets refer to coalition samples per explained image. Real model
   evaluations, including baseline/diagnostic calls, are recorded separately.
5. Run the official LIME `LimeBase` regression on Bernoulli binary foreground-pixel
   masks. This custom adapter uses cosine distance to the all-present mask,
   kernel `sqrt(exp(-d² / 0.25²))`, Ridge alpha 1, and no feature selection. It
   fits both output columns using the same perturbations. It is **adapted image
   LIME**, not the default superpixel-based `LimeImageExplainer` workflow.
6. For each method calculate `difference = attribution_8 - attribution_3`.
   Set to zero the highest-scoring pixels with **strictly positive** difference,
   up to `round(0.20 * 784) = 157` pixels. The denominator is the entire image,
   not just the foreground. Fewer than 157 pixels may qualify. Stable ascending
   pixel index breaks ties. Negative contributions and absolute values are not
   used to choose deletion pixels.
7. Compare predictions before and after deletion. All methods use the same
   model, image IDs, zero replacement, and deletion cap. The random controls
   match each method's **actual** deletion count on each image. Each of 30 random
   repeats chooses uniformly from that image's nonzero pixels.

### Metric

The main metric is the increase in log odds of 3 relative to 8:

```text
delta = log[p(3 | masked) / p(8 | masked)]
      - log[p(3 | original) / p(8 | original)]
```

For a softmax classifier this equals

```text
delta = (logit_3(masked) - logit_8(masked))
      - (logit_3(original) - logit_8(original))
```

Computing it with logits avoids unstable logarithms and probability clipping.
Positive values mean a shift toward 3 relative to 8, not necessarily that 3
becomes the overall predicted class. Both `target_beats_source` and
`top1_is_target` are reported.

Random trials are averaged **within each image** before plotting or summarizing.
Thus a plot with n=20 has 20 data points per method, not 600 independent samples.
Raw random trials are also saved. For their average rows, `prediction_after=-1`
means a categorical mean prediction is undefined; top-1 success is a rate.

## Outputs

| File | Meaning |
| --- | --- |
| `manifest.json` | Fixed IDs, seed, model hash, training record, protocol, package versions and deviations |
| `metrics.csv` | Per-image method results and per-image means of matched random controls |
| `random_trials.csv` | Individual random-deletion trials |
| `diagnostics.json` | SHAP additivity residual and LIME fit diagnostics |
| `summary.csv`, `summary.json` | Mean/median delta, deletion counts, success rates and time |
| `image_ID.npz` | Image, full two-class attribution maps, deleted indices and masked images |
| `figure5a_partial.png` | Input, explain 8, explain 3 and masked image for the first preselected image |
| `figure5b_partial.png` | Delta distributions with one point per image |

Positive attribution is red and negative attribution blue. Both class maps
share the same color scale within a method. A new scale is used for each method,
so color intensity should not be used as a cross-method magnitude comparison.
The displayed example is always the first preselected ID, not the best result.

## Delivered run: actual results

The included `results/50k` run uses seed 2026, 20 images, 50,000 requested
coalition samples per method/image, and 30 matched random repeats per image.
The included CNN achieved **98.32%** test accuracy. All six unit/analytic tests
passed. The saved random controls and mask budgets were also checked.

| Method | Mean delta log-odds | Median delta | Overall top-1 becomes 3 |
| --- | ---: | ---: | ---: |
| Kernel SHAP | 15.320 | 15.700 | 14/20 (70%) |
| Random matched to Kernel count | 6.481 | 6.209 | 7.00% over random trials |
| Adapted LIME | 14.442 | 14.887 | 7/20 (35%) |
| Random matched to LIME count | 6.495 | 6.577 | 6.17% over random trials |

Both learned masking rules made 3 more probable than 8 on all 20 images,
but this did not always make 3 the most probable of all ten classes.
Their mean removed-pixel counts were 107.4 (Kernel SHAP) and 110.3 (LIME).
The matched random controls delete exactly the corresponding count on each image.
The largest Kernel SHAP additivity residual was approximately 2.81e-18;
this is a numerical constraint check, not evidence of exact attribution accuracy.

In this run, attribution-guided deletion had a larger shift than the matched
random controls, and Kernel SHAP had a higher target top-1 rate than LIME.
These are descriptive results for this model and sample, not a statistical
claim that either algorithm always performs better. Sampling, reference choice,
LIME regularization and differing actual deletion counts limit direct comparisons.
These numerical values should not be compared as exact replications of the
original paper's boxplot heights.

## Comparison with the original paper

| Item | Original Figure 5 | This implementation |
| --- | --- | --- |
| Model | Specific pretrained CNN from DeepLIFT experiment | Newly trained PyTorch CNN |
| Methods | Original DeepLIFT, New DeepLIFT, Kernel SHAP, LIME | Kernel SHAP, adapted LIME, matched random controls |
| Sampling | 50,000 for SHAP and LIME | Smoke 512, quick 5,000, paper-budget 50,000 |
| Quantitative sample | 20 random images | 20 preselected correctly classified test 8s by default |
| Masking | 20% of image, to move from 8 to 3 | Up to 157 strictly positive class-difference pixels |
| Selection tie handling | Related author notebook uses a strict score threshold | Exact stable top-k tie break |
| LIME details | Modified single-pixel digit segmentation | Explicit documented binary foreground-pixel adapter |

Compare qualitative behavior under the new model, not exact boxplot values.
Do not conclude that SHAP always wins. Look at paired per-image differences,
random controls, LIME fit quality and SHAP residuals, and retain failures.
Small samples and masking-induced distribution shift limit interpretation.
This project does not implement original/new DeepLIFT or label modern
DeepExplainer as either original-paper variant.

## Contributions and reused code

**This project's implementation:** CNN and training/validation procedure,
binary-pixel adapter, deterministic sample selection, class-difference masking,
stable log-odds metric, matched random controls, saved diagnostics, figures,
configuration manifests, and analytic/unit tests.

**Reused algorithms via library calls:** official SHAP KernelExplainer, official
LIME LimeBase, scikit-learn Ridge, PyTorch, and torchvision MNIST. We do not
claim to have implemented Kernel SHAP or LIME from scratch. The masking rule
and logit metric follow the authors' experiment; the Python implementation here
was written for this project rather than copied from their notebook.

### Sources

- [SHAP paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf), Figure 5 and Section 5.3.
- [SHAP official source](https://github.com/shap/shap) and [KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).
- Ribeiro, Singh & Guestrin (2016), [LIME official source](https://github.com/marcotcr/lime), especially `lime_base.py` and `lime_image.py`.
- Shrikumar, Greenside & Kundaje (2017), [DeepLIFT paper](https://arxiv.org/abs/1704.02685), and [author MNIST notebook](https://github.com/kundajelab/deeplift/blob/master/examples/mnist/MNIST_replicate_figures.ipynb). The notebook confirms `n_to_erase=157`, positive source-minus-target scores, and differences of pre-softmax class scores. The notebook itself is not redistributed here.
- [torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).

## Submission

Publish the source, README, dependency file, tests, and representative result
files in the group's public GitHub repository if claiming reproduction bonus.
The repository URL is not created by this script. Do not commit downloaded
MNIST or local virtual environments. The small checkpoint can be distributed as
a release asset; its SHA256 appears in each result manifest. Tell readers how to
train it again. Document any changes to the commands or settings used in your
submitted run.
