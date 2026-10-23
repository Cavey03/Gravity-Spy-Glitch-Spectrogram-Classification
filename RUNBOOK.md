# RUNBOOK — the official three-model comparison

Step-by-step for whoever runs the official training (the RTX 5070 machine).
All commands run from the repo root.

---

## Step 1 — Get the latest code
```bash
git pull origin main
```
The dataset and checkpoints are gitignored, so this never touches them.

## Step 2 — Check all three models build and train
```bash
python train.py --list-models
python scripts/smoke_test.py --all-models
```
The smoke test trains and evaluates every model for 1–2 epochs on synthetic data,
so a broken model fails here in a minute instead of after an hour of GPU time.
The first run downloads the pretrained weights (~85 MB EfficientNet, ~110 MB Swin).

## Step 3 — Check the GPU and data
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python data/inspect_data.py --show-augmented
```
Expect `True NVIDIA GeForce RTX 5070`, and train=5587 / validation=1200 /
test=1179 with 134 blank images reported.

## Step 4 — Train and evaluate each model
```bash
python train.py    --model custom_cnn
python evaluate.py --model custom_cnn

python train.py    --model transfer_cnn --lr 1e-4 --epochs 25
python evaluate.py --model transfer_cnn

python train.py    --model transformer  --lr 3e-5 --epochs 20 --weight-decay 0.05
python evaluate.py --model transformer
```
The pretrained models use a lower learning rate than the from-scratch baseline
because fine-tuning must not wash out the pretrained weights. This is a permitted,
justified per-architecture difference (spec §6) — say so in the report.

Everything else is identical for all three: the same split, seed (42), class
weighting, augmentation and early stopping on validation macro-F1.

**Do not pass `--drop-blank-images` for one model and not the others.**
`compare.py` refuses to compare runs that disagree on it.

## Step 5 — Build the comparison
```bash
python compare.py
```

## Step 6 — Commit the results
```bash
git add outputs/
git commit -m "Add trained results for all three models"
git push origin main
```
Checkpoints (`*.pt`) stay local via `.gitignore`; only the small result files and
figures are committed.

---

## Where everything lands

| What | Location | In git? |
|---|---|---|
| Dataset | `dataset/trainingsetv1d1.h5` | No |
| Pretrained weights | `C:\Users\<you>\.cache\huggingface\` | No |
| Per-model results | `outputs/<model>/` — `metrics.json`, `per_class_metrics.md/.csv`, `predictions.csv`, `confusion_matrix.png`, `training_curves.png`, `examples_correct.png`, `examples_incorrect.png`, `config.json`, `env.json`, `history.json` | Yes |
| Checkpoints | `outputs/<model>/checkpoint_best.pt`, `checkpoint_last.pt` | No |
| Comparison | `outputs/comparison/` — `comparison.md`, `comparison.csv`, `model_comparison.png` | Yes |

`config.json` and `env.json` together record exactly how each run was made:
every setting, library versions, GPU model and memory (spec §7.3).

---

## Troubleshooting

- **CUDA out-of-memory on the transformer:** `--batch-size 32`, then 16.
- **Windows DataLoader hangs:** `--num-workers 0`.
- **`compare.py` shows fewer than three models:** it lists only evaluated
  models. Run `evaluate.py --model <name>` for the missing one.
- **`compare.py` refuses to compare:** two models were run with different
  `--drop-blank-images` settings or on different splits. Re-run to match.

## Reference result

`custom_cnn` on the RTX 5070: **test macro-F1 0.9332, accuracy 0.9686**
(best validation macro-F1 0.9489 at epoch 45, early-stopped at epoch 57).
This reproduces exactly with seed 42.
