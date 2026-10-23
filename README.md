# Gravity Spy — LIGO glitch spectrogram classification

Classifying LIGO **Gravity Spy** glitch spectrograms into **22 classes**. This is
ordinary supervised image classification: the model only ever sees a spectrogram
PNG, never the underlying strain time series.

Three models are compared on one shared pipeline and one shared test set:

| Model | File | Owner | Status |
|---|---|---|---|
| `custom_cnn` — 5-block CNN, from scratch | `models/custom_cnn.py` | Student 1 | implemented, trained |
| `transfer_cnn` — EfficientNetV2-S, ImageNet-pretrained | `models/transfer_cnn.py` | Student 2 | implemented |
| `transformer` — Swin-Tiny, ImageNet-pretrained | `models/transformer.py` | Student 3 | implemented |

Everything except the model files is shared infrastructure. **A model is added
by writing one file and adding one line to `models/registry.py`, and nothing
else changes.** To run the official three-model comparison, follow
[RUNBOOK.md](RUNBOOK.md). See [Contributing a
model](#contributing-a-model).

---

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Getting the data](#getting-the-data)
- [Reproducing the results](#reproducing-the-results)
- [The interface contract](#the-interface-contract)
- [Contributing a model](#contributing-a-model)
- [Repository layout](#repository-layout)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Reading the results](#reading-the-results)
- [Troubleshooting](#troubleshooting)
- [Dataset citation and licence](#dataset-citation-and-licence)

---

## Quick start

```bash
python data/inspect_data.py          # verify the data is wired up correctly
python train.py --model custom_cnn   # train
python evaluate.py --model custom_cnn # all metrics + figures
python compare.py                    # aggregate across models
```

**The full write-up of the data, setup and results is in the notebook**
[`notebooks/gravity_spy_project.ipynb`](notebooks/gravity_spy_project.ipynb), saved with
all outputs visible. It loads the trained results from `outputs/`; set `RETRAIN = True`
at the top to retrain all three models first.

```bash
jupyter notebook notebooks/gravity_spy_project.ipynb
```

No data yet? The whole pipeline runs on synthetic data in a couple of minutes:

```bash
python scripts/smoke_test.py --all-models
```

---

## Installation

Python **3.10–3.12**.

**1. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
on macOS/Linux.

**2. Install PyTorch from the index that matches your GPU.** Do this *before*
`requirements.txt`; plain `pip install torch` pulls a wheel that may not have
kernels for your card.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

That `cu128` index is for **RTX 50-series (Blackwell, sm_120)**, including the
RTX 5070. For RTX 30/40-series use `cu124`; for no GPU use `cpu`.

**3. Install everything else**

```bash
pip install -r requirements.txt
```

**4. Check the GPU is visible**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False` on a machine with an NVIDIA GPU, you almost certainly
installed torch from the default PyPI index in step 2. Reinstall with the
`--index-url` above.

> Verified working on an RTX 5070: `torch 2.11.0+cu128`, `timm 1.0.29`,
> Python 3.12.

---

## Getting the data

**Gravity Spy training set v1.1** — Zenodo DOI
[10.5281/zenodo.1476551](https://doi.org/10.5281/zenodo.1476551), licensed
**CC BY 4.0**. **7,966** labelled glitches across 22 classes, pre-split
**5,587 train / 1,200 validation / 1,179 test**. Each glitch is rendered at four
*duration views* (0.5 / 1.0 / 2.0 / 4.0 seconds); we use the **2.0 s** view by
default.

> Papers on Gravity Spy often quote **8,583** glitches. That is the *original*
> release ([Zenodo record 1476156](https://zenodo.org/records/1476156)). The
> "Updated Gravity Spy Data Set" used here (record 1476551, v1.1) contains 7,966.
> Both the HDF5 hierarchy and the metadata CSV in this release agree on 7,966, so
> the difference is the dataset version, not a loading error.

The data is not in this repository — fetch it from Zenodo. Two formats are
offered and the pipeline reads either:

| File | Size | Need it? |
|---|---|---|
| `trainingsetv1d1.h5` | 3.1 GB | **Yes** — pre-cropped `1×140×170` float32 arrays |
| `trainingset_v1d1_metadata.csv` | 5.4 MB | Optional, recommended — the split as a table |
| `trainingsetv1d1.tar.gz` | 5.5 GB | **No** — the same glitches as raw PNGs. This pipeline reads the `.h5` only. |

**You only need the 3.1 GB `.h5`** (plus the 5.4 MB CSV if you want it). Zenodo
reports 8.6 GB as the total across all three files; do not download the tarball
as well, it is the same data in a bulkier form.

The HDF5 file is smaller despite storing float32 because the PNGs are complete
matplotlib renders — axes, tick labels and a colourbar — at roughly ten times the
pixel count, nearly all of which would have to be cropped away again.

Zenodo publishes an MD5 for each file; check your download against the value on
the record page, since a 3.1 GB transfer that silently truncates will otherwise
surface much later as a confusing indexing error.

Download **`trainingsetv1d1.h5`** (3.1 GB) into `gravity-spy/dataset/`. Arrays
are already cropped to `1x140x170` and scaled to [0, 1], so no preprocessing is
needed. Optionally also grab `trainingset_v1d1_metadata.csv` (5.4 MB) -- the
pipeline does not need it (the HDF5 encodes the same split), but it is handy for
the report.

```
gravity-spy/
  dataset/
    trainingsetv1d1.h5
```

You do **not** need 3.1 GB of RAM: `h5py` reads lazily, so the loader pulls one
~95 KB array per sample and never holds more than a batch.

Check your download against the MD5 on the Zenodo record page -- a transfer that
silently truncates would otherwise surface much later as a confusing error.

### Pointing the code at the data

The default is `gravity-spy/dataset/`. Override in any of three ways
(highest priority first):

```bash
python train.py --model custom_cnn --data-root /path/to/dataset   # CLI flag
export GRAVITYSPY_DATA=/path/to/dataset                            # env var
# or edit DEFAULT_DATA_ROOT in config.py
```

### Verify before training

```bash
python data/inspect_data.py
```

This prints per-class counts per split, asserts that no `gravityspy_id` appears
in more than one split, and writes `outputs/data_inspection/`
(`class_distribution.png`, `examples_per_class.png`, `class_counts.csv`). Add
`--show-augmented` to see exactly what the training augmentation does to an
image.

If it reports **0 indexed samples**, the group layout in your download differs
from the expected one:

```bash
python data/inspect_data.py --dump-h5-structure
```

---

## Reproducing the results

Everything is driven by a single global seed (default `42`) applied to Python,
NumPy, torch CPU and torch CUDA, with cuDNN forced into deterministic mode. Each
run writes its exact `config.json` and an `env.json` of library versions and GPU
details, so a result can always be traced back to what produced it.

```bash
# 1. sanity-check the data
python data/inspect_data.py

# 2. train each model
python train.py --model custom_cnn
python train.py --model transfer_cnn --lr 1e-4 --epochs 25
python train.py --model transformer  --lr 3e-5 --epochs 20 --weight-decay 0.05

# 3. evaluate each on the SAME test set
python evaluate.py --model custom_cnn
python evaluate.py --model transfer_cnn
python evaluate.py --model transformer

# 4. aggregate
python compare.py
```

Useful flags (`python train.py --help` for the full list):

| Flag | Effect |
|---|---|
| `--list-models` | Show registered models and their input sizes |
| `--duration 4.0` | Use a different duration view |
| `--no-class-weights` | Ablate the class-weighted loss |
| `--no-augment` | Ablate augmentation |
| `--horizontal-flip` | Enable time-reversal augmentation (off by default) |
| `--no-pretrained` | Build a backbone with random weights |
| `--seed 1` | Different seed, for a variance estimate |
| `--drop-blank-images` | Exclude the 134 blank images (see below). All three models must match |

### Outputs

```
outputs/<model>/
  config.json                 resolved config for the run
  env.json                    library versions, GPU, timestamp
  history.json                per-epoch losses and validation metrics
  checkpoint_best.pt          weights at best validation macro-F1
  checkpoint_last.pt          weights after the final epoch
  training_curves.png         train/val loss + val accuracy and macro-F1
  metrics.json                every metric (compare.py reads this)
  per_class_metrics.md/.csv   per-class precision/recall/F1 table
  predictions.csv             per-sample true/predicted label + confidence
  confusion_matrix.png        row-normalised
  examples_correct.png        grid of correct predictions
  examples_incorrect.png      grid of the most confident mistakes

outputs/comparison/
  comparison.md / .csv
  model_comparison.png
```

---

## The interface contract

**Every model file exposes exactly one public function:**

```python
def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module:
    ...
```

**and is registered with exactly one line in `models/registry.py`:**

```python
register("my_model", my_model.build_model, input_size=(224, 224),
         description="What it is.")
```

That is the whole contract. `train.py`, `evaluate.py` and `compare.py` look
models up by string name and never import a model module directly, so
`--model <name>` works for **any** registered model with zero other changes.

Requirements on your `build_model`:

- **Return raw logits** of shape `(batch, num_classes)`. Do **not** apply
  softmax — `nn.CrossEntropyLoss` applies log-softmax internally. Adding your own
  will not crash; it will quietly train a worse model, which is much harder to
  notice.
- **Honour `pretrained`.** With `pretrained=False` the same architecture must
  build with random weights, so the "does pretraining help?" ablation is fair.
- **Accept 3-channel input** at the size you declared in the registry. The shared
  pipeline already replicates the greyscale spectrogram across 3 channels and
  resizes for you.

### Why `input_size` lives in the registry rather than in `build_model`

Models disagree about what input size they can accept, and the disagreement is
architectural:

- A plain CNN with global pooling is size-agnostic. Ours therefore runs at the
  spectrograms' **native 140×170** and never pays for upsampled pixels that carry
  no extra information.
- **ViT-B/16** splits the image into fixed 16×16 patches and adds one *learned*
  positional embedding per patch. Change the input size and the patch count
  changes, so the pretrained embeddings no longer line up and must be
  interpolated.
- **Swin-T** uses shifted-window attention with a 7×7 window and needs the
  feature map to stay divisible by that window at every stage.

If input size were a `build_model` argument, every model would be obliged to
honour whatever it was handed — which for the transformers means positional
embedding surgery inside what is supposed to be a one-file contribution.
Declaring it as registry metadata inverts the dependency: each model states what
it wants, and the shared pipeline resizes to match.

---

## Contributing a model

*How `transfer_cnn` and `transformer` were added, and how to add another model.*

**You edit exactly one file**, plus one `register(...)` line in `models/registry.py`.

### Step 1 — implement `build_model`

Open your file (`models/transfer_cnn.py` or `models/transformer.py`). Its
docstring contains the full contract, a worked sketch, recommended `timm`
backbone names, and the specific pitfalls for that model family. Delete the
`raise NotImplementedError` and implement:

```python
import timm

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",   # or swin_tiny_patch4_window7_224.ms_in1k
        pretrained=pretrained,
        num_classes=num_classes,
    )
```

Passing `num_classes` makes `timm` swap the pretrained 1000-way ImageNet head for
a correctly initialised `num_classes`-way one. Do **not** build the model and
then reassign `.fc` or `.classifier` by hand — the attribute name differs between
architectures, and `num_classes` handles it uniformly.

### Step 2 — check it satisfies the interface

```bash
python scripts/smoke_test.py --all-models
```

This trains and evaluates your model on synthetic data. It proves the plumbing
works; it says nothing about accuracy.

### Step 3 — train and evaluate for real

```bash
python train.py --model transfer_cnn --lr 1e-4 --epochs 25
python evaluate.py --model transfer_cnn
python compare.py
```

### Step 4 — commit

Commit **only your model file** and your `outputs/<your_model>/` directory. If
you find yourself needing to change `train.py`, `evaluate.py`, `data/` or
`utils/`, stop and raise it with the group — that is a sign the contract needs
changing for everyone, not a local fix.

### The three mistakes that actually happen

1. **Learning rate.** The `config.py` default (`3e-4`) is tuned for the
   from-scratch CNN and is far too high for fine-tuning. It washes out the
   pretrained features in the first few hundred steps and you end up no better
   than training from scratch. Use `1e-4` for the CNN backbone, `3e-5` for the
   transformer. If your macro-F1 is inexplicably poor, **check this first.**
2. **Batch size.** 224×224 with a 21M+ parameter backbone uses much more memory
   than the baseline. On CUDA out-of-memory, drop `--batch-size` to 32 or 16
   before changing anything else.
3. **Epochs.** Fine-tuning converges much faster than training from scratch;
   15–25 epochs is usually plenty. The default 60 mostly wastes time and
   overfits.

If you want per-model behaviour such as freezing the backbone for the first few
epochs, implement it **inside your model file** (e.g. a `freeze_backbone()`
method). Do not add hooks to `train.py` — that is shared code.

---

## Repository layout

```
gravity-spy/
├── README.md
├── requirements.txt
├── config.py               paths, seed, 22-class list, duration, all defaults
├── data/
│   ├── dataset.py          Dataset: 2.0s view per glitch, grey→3ch, from HDF5
│   ├── transforms.py       preprocessing + domain-safe augmentation
│   └── inspect_data.py     per-class counts per split + example grids
├── models/
│   ├── registry.py         name → build fn; build_model(name, num_classes, pretrained)
│   ├── custom_cnn.py       fully implemented — the reference to copy
│   ├── transfer_cnn.py     STUB for teammate
│   └── transformer.py      STUB for teammate
├── train.py                model-agnostic loop, --model <name>, early stopping
├── evaluate.py             checkpoint → test set → all metrics + figures
├── compare.py              aggregate metrics.json across models
├── utils/
│   ├── seeding.py          global seed, cuDNN determinism, worker seeding
│   ├── metrics.py          metric suite + class weights
│   ├── plotting.py         all figures
│   └── env_info.py         library/GPU version capture
├── notebooks/
│   └── gravity_spy_project.ipynb   dataset, setup and results, outputs visible
├── splits/                 exact train/validation/test lists + blank-image list
├── scripts/
│   └── smoke_test.py       end-to-end run on synthetic data
└── outputs/                per-model results and figures (checkpoints git-ignored)
```

---

## Design decisions worth knowing

### The split is taken from `sample_type`, never generated

The dataset ships a canonical `train`/`validation`/`test` assignment in its
`sample_type` field, and the code uses it verbatim. **Images are never split
randomly.** This is a correctness requirement, not a preference: each glitch is
rendered at four durations, and those four images are four views of the *same
physical event*. A per-image random split would put views of one glitch on both
sides of the train/test boundary, and the model would be scored partly on events
it had already memorised — inflating the test score with no visible symptom.

`sample_type` is assigned per glitch (all four views share one `gravityspy_id`
and one `sample_type`), so honouring it makes the leak structurally impossible.
`data/dataset.py` additionally asserts that no `gravityspy_id` appears in more
than one split, so a malformed download fails loudly instead of silently
producing a suspiciously good number.

All three models use the **same test set**, and `compare.py` refuses to build a
table from models evaluated on different splits.

### 134 images in the published dataset are blank

**1.68% of the 2.0 s views (134 of 7,966) contain no information at all** — every
pixel sits at the floor value `0.0832`. They are not evenly spread:

| Class | Blank | Share of class |
|---|---|---|
| `Paired_Doves` | 8 / 27 | **29.6%** |
| `Extremely_Loud` | 108 / 447 | **24.2%** |
| `No_Glitch` | 9 / 150 | 6.0% |
| `None_of_the_Above` | 2 / 81 | 2.5% |
| others (`Blip`, `Koi_Fish`, `Low_Frequency_Burst`, `Whistle`) | 7 | <0.5% |

Split-wise: 97 train, 18 validation, 19 test.

This is a defect in the **published dataset**, not a loading error — the arrays
really are constant on disk. Changing `--duration` does not help: **107 of the
134 are blank at all four duration views**, and only 27 have data at 4.0 s.

**They are kept by default.** Two reasons:

1. Dropping them would mean our test set is no longer the canonical Gravity Spy
   test set, so our numbers would not be comparable with published results.
2. It would only be the *same* test set across all three of our models if all
   three of us remembered to pass the same flag.

Blanks appear in train *and* test, and 108 of 134 are `Extremely_Loud`, so a
model that learns "blank → `Extremely_Loud`" scores genuinely well on the
canonical benchmark. That is a real regularity in the dataset everyone publishes
against, not leakage — but it *is* a degenerate shortcut, and it is worth saying
so in the write-up rather than leaving it as an unexplained bump in
`Extremely_Loud`'s F1.

The cost lands on `Paired_Doves`: 8 of its 27 images are blank and pixel-identical
to `Extremely_Loud`'s blanks, so those are unclassifiable in principle. **Expect
systematic `Paired_Doves` → `Extremely_Loud` confusion in the confusion matrix**;
that is the artefact, not a modelling failure.

To measure the effect as an ablation:

```bash
python train.py --model custom_cnn --drop-blank-images
python evaluate.py --model custom_cnn
```

`drop_blank_images` is recorded in `metrics.json`, and `compare.py` **refuses**
to build a table from models that disagree about it — otherwise you would be
comparing scores on two different test sets.

`python data/inspect_data.py` reports the blank counts on every run.

### Augmentation is spectrogram-valid

A spectrogram is not a photograph. Its axes mean different, non-interchangeable
things: horizontal is **time**, vertical is **frequency**, intensity is energy.

**Not used, and deliberately so:**

- **Vertical flip** — maps low frequencies to high. Several classes are *defined*
  by frequency: `Violin_Mode` is a specific harmonic resonance,
  `Low_Frequency_Burst` is named for its band, `1080Lines` and `1400Ripples` are
  named for their frequency in Hz. Flipping vertically turns one class into a
  physically impossible artefact.
- **Rotation / shear / perspective** — any rotation that is not a multiple of 90°
  mixes the time and frequency axes together, producing an image no
  interferometer could record. Even 90° transposes them.
- **Horizontal flip** — this is *time reversal*. Harmless for symmetric glitches
  (`Blip`, `Koi_Fish`) but it destroys asymmetric ones: a `Chirp` sweeps *upward*
  in frequency as the binary inspirals, and reversed it sweeps downward, which is
  not a chirp. **Off by default**, exposed as `--horizontal-flip` so the effect
  can be measured rather than assumed.

**Used (train split only):**

- **Small time shifts** (±8% of width) — the glitch is not guaranteed to sit
  exactly at the window centre, so this is genuine nuisance variation. Implemented
  as a shift with edge padding, not a circular roll, which would teleport energy
  from the end of the window to the start and could fabricate a second glitch.
- **SpecAugment-style time and frequency masking** (Park et al., 2019,
  [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)) — the standard
  augmentation for spectrogram inputs. Forces classification from partial
  evidence rather than one narrow band. Masks are filled with the per-image mean,
  not zero: after normalisation zero is a meaningful intensity that the model
  could learn to read as "this is a mask".
- **Mild brightness/contrast** (±15%) — corresponds to glitch loudness and dynamic
  range, which vary with detector state and are not class-defining.

Validation and test see only resize + normalise, so their numbers are comparable
across models and runs.

### Greyscale replicated to 3 channels

The spectrograms are single-channel. They are replicated across 3 identical
channels so ImageNet-pretrained backbones accept the input unmodified — no
surgery on the first convolution, and the pretrained first-layer filters keep
working as trained. ImageNet normalisation statistics are used for the same
reason (`imagenet_norm=True` in `config.py`; the from-scratch CNN is indifferent).

### Class imbalance handled in the loss

`Blip` alone is 1,274 of the 5,587 training images (23%), while `Paired_Doves`
has 19 — a **67:1** imbalance. The loss is class-weighted by default
(`--no-class-weights` to ablate).

The default scheme is **inverse square root** (`1/√n_c`), not plain inverse
frequency. Plain `1/n_c` is the textbook choice but is too aggressive here: it
turns the 67:1 class imbalance into a 67:1 weight ratio, so gradients from
nineteen `Paired_Doves` images start to dominate every batch and validation
macro-F1 oscillates. `1/√n_c` compresses that to about 8:1 — still a strong
correction toward the tail, but stable.
Weights are normalised to mean 1.0 so switching schemes does not also change the
effective learning rate.

### Reproducibility

A single seed (`config.seed`, default 42) seeds Python, NumPy, torch CPU and
torch CUDA, plus per-DataLoader-worker seeding so augmentation draws are
independent across workers. cuDNN is set deterministic with autotuning off —
autotuning picks whichever convolution algorithm times fastest on the day, and
different algorithms sum in different orders, so results drift in the last few
bits and then diverge over hundreds of epochs. Every run saves its `config.json`
and `env.json`.

---

## Results so far

`custom_cnn`, trained on the canonical split, evaluated on the canonical
1,179-image test set (seed 42, RTX 5070, ~6 min):

| Metric | Value |
|---|---|
| **macro-F1** | **0.9332** |
| Accuracy | 0.9686 |
| Balanced accuracy | 0.9267 |
| weighted-F1 | 0.9684 |
| macro precision / recall | 0.9511 / 0.9267 |
| top-3 accuracy | 0.9958 |
| Parameters | 2,359,542 |
| Epochs | 57 (best at 45, early-stopped) |

Accuracy is 96.9%, right in the expected 97–98% band, and 3.5 points above
macro-F1 — the imbalance gap in one number.

**Weakest classes:** `Paired_Doves` 0.667, `None_of_the_Above` 0.750,
`No_Glitch` 0.818, `Wandering_Line` 0.833.

**`Paired_Doves` is entirely explained by the blank-image defect.** Two of its
four test images contain no data; both were predicted `Extremely_Loud`, and both
of the *real* `Paired_Doves` images were classified correctly at 0.96 and 1.00
confidence. Excluding blank images from scoring:

| | macro-F1 | Accuracy | Balanced acc. |
|---|---|---|---|
| Canonical test set (reported) | 0.9332 | 0.9686 | 0.9267 |
| Excluding the 19 blank test images | 0.9502 | 0.9707 | 0.9511 |

So **0.0169 of macro-F1 is lost to the dataset defect, not to the model**, and
`Paired_Doves` F1 goes 0.667 → 1.000. Report the canonical number; cite the
second as analysis.

**Genuine confusions worth discussing** (i.e. not blank-image artefacts):
`Low_Frequency_Burst` ↔ `Low_Frequency_Lines` (7 errors both ways),
`Low_Frequency_Lines` → `No_Glitch` (3), `Repeating_Blips` → `Blip` (2). These
are physically sensible — the confused pairs occupy the same frequency band and
differ mainly in duration and repetition structure.

## Reading the results

**Expect accuracy around 97–98%. Do not lead with it.** The class distribution
is severely skewed, so a model can score very highly on accuracy while doing
badly on several rare classes, because those classes barely move the average.

**Macro-F1 is the headline metric.** It weights all 22 classes equally, so
neglecting `Paired_Doves` is penalised as heavily as neglecting `Blip`. It is
what early stopping selects on, what `compare.py` sorts by, and what should lead
the write-up.

Two derived numbers worth quoting:

- **weighted-F1 − macro-F1.** A large gap means the tail classes are being
  neglected — the weighted average is dominated by the classes with most support.
- **Balanced accuracy** (mean per-class recall) — a second imbalance-aware view
  that is easier to explain to a non-specialist than macro-F1.

`evaluate.py` prints the five weakest classes by F1 at the end of every run.
That list, plus `confusion_matrix.png` and `examples_incorrect.png` (the model's
*most confident* mistakes, which are the informative ones), is usually where the
interesting discussion in a write-up comes from.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | torch installed from default PyPI. Reinstall with the `--index-url` in [Installation](#installation). |
| `no kernel image is available for execution` | torch built for an older architecture than your GPU. On RTX 50-series you need the `cu128` build. |
| `trainingsetv1d1.h5 not found` | `--data-root` points somewhere without the file. See [Getting the data](#getting-the-data). |
| `Indexed 0 glitches at duration 2.0s` | The HDF5 group layout differs from `<label>/<split>/<id>/2.0.png`. Check the download is the v1.1 file. |
| `Split leakage: N gravityspy_id(s) appear in multiple splits` | The download is malformed or was manually reorganised. Do not train on it — re-download. |
| `CUDA out of memory` | Lower `--batch-size` (32, then 16). AMP is already on by default. |
| Training is slower than expected | `deterministic=True` disables cuDNN autotuning. That is the intended trade for a reproducible report. |
| Windows DataLoader hangs or errors | Use `--num-workers 0`. Windows spawns rather than forks workers, which is stricter about what can be pickled. |

---

## Dataset citation and licence

**Gravity Spy training set v1.1**, Zenodo, DOI
[10.5281/zenodo.1476551](https://doi.org/10.5281/zenodo.1476551).
Licensed **CC BY 4.0** — free to use with attribution.

Please cite the dataset and the accompanying papers:

> Bahaadini, S., Noroozi, V., Rohani, N., Coughlin, S., Zevin, M., Smith, J. R.,
> Kalogera, V., & Katsaggelos, A. (2018). *Machine learning for Gravity Spy:
> Glitch classification and dataset.* Information Sciences, 444, 172–186.

> Zevin, M., Coughlin, S., Bahaadini, S., et al. (2017). *Gravity Spy:
> Integrating advanced LIGO detector characterization, machine learning, and
> citizen science.* Classical and Quantum Gravity, 34(6), 064003.

The augmentation strategy follows:

> Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le,
> Q. V. (2019). *SpecAugment: A simple data augmentation method for automatic
> speech recognition.* [arXiv:1904.08779](https://arxiv.org/abs/1904.08779).
