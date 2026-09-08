# Gravity Spy — Glitch Spectrogram Classification

Classify LIGO **Gravity Spy** glitch spectrograms into **22 classes** with PyTorch.
This is ordinary image classification: each model sees a single spectrogram image.

The repo is built for a **3-person project**. One person (the maintainer) built the
shared pipeline and a from-scratch reference model. Two teammates each add one model
by writing **a single file** — they never touch shared code. See
[CONTRIBUTING](#contributing-adding-a-model-teammates) below.

---

## 1. What's in here

```
gravity-spy/
  config.py               # single source of truth: paths, seed, 22 classes, hyperparams
  data/
    dataset.py            # PyTorch Dataset: 2.0s view per glitch, grey->3ch, from .h5 or PNG
    transforms.py         # preprocessing + domain-safe train augmentation
    inspect_data.py       # per-class counts, h5-tree probe, example grid
    make_synthetic.py     # tiny fake dataset (run the pipeline without the 8.6GB download)
  models/
    registry.py           # THE CONTRACT: name -> build_model(num_classes, pretrained)
    custom_cnn.py         # reference model — small 5-block CNN, from scratch (implemented)
    transfer_cnn.py       # STUB for a teammate: EfficientNetV2-S / ConvNeXt-Tiny via timm
    transformer.py        # STUB for a teammate: Swin-Tiny / ViT-B/16 via timm
  train.py                # model-agnostic training loop  (--model <name>)
  evaluate.py             # test-set metrics + figures     (--model <name>)
  compare.py              # aggregate all models' metrics into one table
  utils/                  # seed, reproducibility, metrics, plotting
  outputs/<model>/        # per-model checkpoints, figures, metrics.json
  requirements.txt
```

---

## 2. Setup

Verified with **Python 3.12**. Create and activate a virtual environment first:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
```

Then install PyTorch — **pick ONE** of the following, then install the rest.

**Option A — CPU only** (fine for testing/smoke runs; too slow for full training):
```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Option B — NVIDIA GPU incl. RTX 50-series (e.g. RTX 5070)** — recommended for the full run.
RTX 50-series ("Blackwell") needs a **CUDA 12.8** build of PyTorch. The newest torch on
the cu128 index is 2.11.0 (paired with torchvision 0.26.0):
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Then confirm the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```
You should see `True  NVIDIA GeForce RTX 5070`. If it prints `False`, update your
NVIDIA driver (Blackwell needs a recent driver + CUDA 12.8), or use the exact command
the selector at <https://pytorch.org/get-started/locally/> gives for your setup.

> `requirements.txt` pins the CPU builds for reproducibility; the `pip install ... --index-url`
> line above overrides torch/torchvision with the right build, so run it **before**
> `pip install -r requirements.txt` (or just ignore the torch lines in the file).

---

## 3. Get the data

**Source:** Gravity Spy Training Set on Zenodo — <https://zenodo.org/records/1486046>.
Licence **CC BY 4.0**.

Follow these steps exactly:

1. Open <https://zenodo.org/records/1486046> in a browser.
2. In the **Files** section, download **these two files** (click each filename):
   - **`trainingsetv1d1.h5`** — 3.1 GB (the pre-cropped `1×140×170` image arrays — this is what we use)
   - **`trainingset_v1d1_metadata.csv`** — 5.4 MB (the labels + the train/validation/test split)

   > You do **not** need `trainingsetv1d1.tar.gz` (5.5 GB) — that's the raw PNGs, an
   > alternative to the `.h5`. Skip it unless you specifically want the PNG loader.
3. Create the folder `gravity-spy/dataset/` if it doesn't exist, and move both
   downloaded files into it so you have **exactly** this:

   ```
   gravity-spy/dataset/trainingsetv1d1.h5
   gravity-spy/dataset/trainingset_v1d1_metadata.csv
   ```

   (Filenames must match — don't rename them. If you keep them somewhere else,
   either edit `DATA_ROOT` in `config.py` or pass `--h5-path` / `--metadata-csv`
   to the scripts.)
4. Verify the files are found before training:

   ```bash
   python -m data.inspect_data --h5-tree --no-grid
   ```

   You should see per-class counts (train 5587 / val 1200 / test 1179) and the HDF5
   layout printed. If you get a "file not found" error, re-check step 3.

**Real .h5 layout** (auto-detected by the loader):
`f[label][sample_type][gravityspy_id]["2.0.png"] -> (1,140,170) float32 in [0,1]`.

### No data yet? Run on synthetic data
```bash
python -m data.make_synthetic --out dataset_synth
python train.py    --model custom_cnn --h5-path dataset_synth/trainingsetv1d1.h5 \
                   --metadata-csv dataset_synth/trainingset_v1d1_metadata.csv --epochs 3 --num-workers 0
```

---

## 4. Reproduce the results (exact steps)

```bash
# 0. (optional) look at the data first
python -m data.inspect_data --h5-tree      # counts + h5 layout + example grid

# 1. train the reference model  (writes outputs/custom_cnn/)
python train.py --model custom_cnn

# 2. evaluate on the test set   (writes metrics.json + figures)
python evaluate.py --model custom_cnn

# 3. compare all evaluated models
python compare.py
```

Do the same with `--model transfer_cnn` / `--model transformer` once those are
implemented. **All three models use the exact same commands** — only the name changes.

### Full training run on the RTX 5070

Once the GPU build is installed (Setup Option B) and `torch.cuda.is_available()` is
`True`, `train.py` automatically uses the GPU — no flag needed. For the full run:

```bash
python train.py --model custom_cnn --num-workers 4
```

- No `--max-batches` here → it trains on **all** data for the full `config.EPOCHS`
  (40) with early stopping. On the RTX 5070 expect well under a minute per epoch.
- `--num-workers 4` speeds up data loading. **On Windows**, if the run hangs at the
  start (spawn issue), drop it to `--num-workers 0`.
- Then `python evaluate.py --model custom_cnn` and `python compare.py` as above.

**Reproducibility.** A single seed (`config.SEED`, default 42) seeds Python/NumPy/torch
and forces cuDNN determinism. Every run writes `outputs/<model>/run_config.json` with the
exact library versions, command, and config used.

**Windows note.** `NUM_WORKERS` defaults to `0` (spawn-based multiprocessing can hang on
Windows). On Linux/GPU pass `--num-workers 4` for faster loading.

**Duration.** Uses the **2.0 s** view by default; change with `--duration {0.5,1.0,2.0,4.0}`
or `config.DURATION`.

---

## 5. Reading the metrics (important)

Overall **accuracy will look high (~97–98%)** — the classes are extremely imbalanced
(Blip ≈ 1821 samples, Paired_Doves ≈ 27; a **67× ratio**), so predicting the common
classes already scores well. **Macro-F1 is the headline metric**: it averages per-class
F1 with equal weight, so rare classes count fully. `evaluate.py` prints both and saves:

- `metrics.json` — overall + per-class precision/recall/F1 + confusion matrix
- `per_class_metrics.csv` — the per-class table
- `confusion_matrix.png` — row-normalized (per-class recall)
- `examples_correct.png` / `examples_incorrect.png` — example predictions
- `training_curves.png` — train/val loss + val accuracy/macro-F1

Handling imbalance: class-weighted loss is **on by default** (`--no-class-weights` to
ablate). The same **test split** (`sample_type == 'test'`) is used for every model.

### Correctness rules baked in
- **Split** is the dataset's own `sample_type` column — never a random split. The 4
  duration-views of a glitch share a `gravityspy_id` and stay in the same split.
- **Grey → 3 channels** so ImageNet-pretrained backbones accept the input.
- **Augmentation (train only)** is spectrogram-valid: small time shifts, SpecAugment
  time/frequency masking, mild brightness/contrast. **No vertical flips, no rotations**
  (they break the physical time/frequency axes). Horizontal flip is off by default.

---

## 6. The interface contract

Every model file exposes exactly:

```python
def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module: ...
```

and registers itself in `models/registry.py` under a string name (via the `@register`
decorator). `train.py` and `evaluate.py` take `--model <name>` and work for **any**
registered model with zero other changes.

---

## <a name="contributing"></a>7. CONTRIBUTING: adding a model (teammates)

**Team workflow (this project).** Official training runs happen on the maintainer's
RTX 5070 so all three models share identical hardware, torch version, and seed — a
cleaner comparison for the report. So as a teammate you:
- **write and push your model file only** (`models/transfer_cnn.py` or `models/transformer.py`);
- you do **not** need a GPU or the dataset to develop — you can smoke-test on the synthetic
  data (Section 3) on CPU;
- the maintainer pulls your file and runs `train.py` / `evaluate.py` for your model on the
  5070, producing the official `outputs/<model>/metrics.json` and figures.

(You *can* also train on your own machine or Google Colab if you want to check results
early — the fixed test set makes any numbers comparable — but the reported run is the one
on the 5070.)

You add a model with **one file + one decorator line**. You do **not** edit `config.py`,
`data/`, `train.py`, `evaluate.py`, `registry.py`, or `utils/`.

1. Open your assigned stub — `models/transfer_cnn.py` or `models/transformer.py`.
   (Both already register their name and satisfy the contract; they just raise
   `NotImplementedError` until you fill them in.)
2. Implement `build_model` — about 3 lines with `timm`:
   ```python
   import timm
   def build_model(num_classes, pretrained):
       return timm.create_model(
           "tf_efficientnetv2_s",         # or convnext_tiny / swin_tiny_patch4_window7_224 / vit_base_patch16_224
           pretrained=pretrained,
           num_classes=num_classes,       # timm swaps the classifier head for you
       )
   ```
   `timm` models expect 3-channel, ImageNet-normalized, 224×224 input — exactly what the
   shared transforms produce.
3. Delete the `raise NotImplementedError`.
4. Train, evaluate, compare — same commands as the reference model:
   ```bash
   python train.py    --model transfer_cnn
   python evaluate.py --model transfer_cnn
   python compare.py
   ```

**Writing a model from scratch instead?** Copy `models/custom_cnn.py` as a template: define
an `nn.Module`, then a `@register("your_name")`-decorated `build_model`. To wire a brand-new
name in, add one import line to `registry._import_models()`.

Checklist: ✅ file exposes `build_model(num_classes, pretrained)` · ✅ decorated with a unique
`@register("name")` · ✅ returns an `nn.Module` producing `(batch, 22)` logits · ✅ no shared
code edited.
```
