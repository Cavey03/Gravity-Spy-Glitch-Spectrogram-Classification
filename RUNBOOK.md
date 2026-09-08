# RUNBOOK — running the official model comparison

For the maintainer (Cavey) who runs the official training on the RTX 5070 after
teammates push their model files. Teammate instructions are in
[README.md](README.md) → CONTRIBUTING.

All commands run from the repo root (`gravity-spy/`).

---

## Step 1 — Pull the teammates' model files
```bash
git pull origin main
```
Brings in `models/transfer_cnn.py` and `models/transformer.py`. Does **not** touch
your local dataset or checkpoints (they are gitignored).

## Step 2 — Sanity-check both models build
```bash
python -c "from models.registry import available_models, build_model; import config; print(available_models()); [build_model(m, config.NUM_CLASSES, pretrained=False) for m in ['transfer_cnn','transformer']]; print('both build OK')"
```
Catches a leftover `raise NotImplementedError`, a syntax error, or a bad backbone
name before you spend GPU time.

## Step 3 — Confirm GPU + data are ready
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m data.inspect_data --no-grid --h5-tree
```
Expect `GPU: True NVIDIA GeForce RTX 5070` and counts train=5587 / val=1200 / test=1179.
The dataset is already at `dataset/trainingsetv1d1.h5` — nothing to re-download.

## Step 4 — Train + evaluate each model on the 5070
```bash
python train.py    --model transfer_cnn --num-workers 4 --lr 1e-4
python evaluate.py --model transfer_cnn --num-workers 4

python train.py    --model transformer  --num-workers 4 --lr 1e-4
python evaluate.py --model transformer  --num-workers 4
```
`--lr 1e-4` because pretrained backbones fine-tune (a gentler LR than the
from-scratch baseline's default 3e-4). Justify this choice in the report — it is a
permitted, deliberate per-architecture difference, not an unfair advantage.

Your baseline (`custom_cnn`) is already trained; re-run it only if you change the
data or config:
```bash
python train.py --model custom_cnn --num-workers 4
python evaluate.py --model custom_cnn --num-workers 4
```

## Step 5 — Build the 3-model comparison
```bash
python compare.py
```
Produces `outputs/comparison.csv`, `comparison.md`, `comparison_macro_f1.png`.

## Step 6 — Commit the results back
```bash
git add outputs/
git commit -m "Add trained results for all three models + comparison"
git push origin main
```
Only small result files are committed; `*.pt` checkpoints stay local (gitignore).

---

## Where everything lands

| What | Location | Committed to git? |
|---|---|---|
| Dataset (already present) | `dataset/trainingsetv1d1.h5`, `dataset/trainingset_v1d1_metadata.csv` | No (gitignored) |
| Pretrained weights (first run downloads ~80–110 MB) | timm cache: `C:\Users\<you>\.cache\huggingface\` | No (cache) |
| Per-model results | `outputs/<model>/` — `checkpoint_best.pt`, `metrics.json`, `per_class_metrics.csv`, `confusion_matrix.png`, `training_curves.png`, `examples_correct.png`, `examples_incorrect.png`, `run_config.json`, `history.json` | Yes, except `*.pt` |
| Comparison | `outputs/comparison.csv`, `comparison.md`, `comparison_macro_f1.png` | Yes |

`metrics.json` is the file `compare.py` reads. `macro_f1` is the headline metric
(accuracy is inflated by the 67× class imbalance).

---

## Troubleshooting

- **Out-of-memory / Windows worker hang on the transformer** (it is ~28M params +
  attention, much heavier than the baseline): drop to `--num-workers 0` and/or
  `--batch-size 32`. Everything else stays the same.
- **A teammate's model errors on input shape:** their backbone likely isn't
  224×224-native. The recommended picks (EfficientNetV2-S, Swin-Tiny) are — the
  shared transforms output 3-channel 224×224.
- **`compare.py` shows only some models:** it only lists models that have been
  evaluated. Run `evaluate.py --model <name>` for any missing one.
- **Reproducing a past run:** every run's exact versions + config are in
  `outputs/<model>/run_config.json`.

## Reference results (baseline)
`custom_cnn`, full run on RTX 5070: **test macro-F1 0.840, accuracy 0.906**
(best val macro-F1 0.870 @ epoch 19, early-stopped at 26).
