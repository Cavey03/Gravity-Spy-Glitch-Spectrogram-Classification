"""
compare.py — Aggregate the three models' metrics into one comparison.

Scans outputs/<model>/metrics.json for every model that has been evaluated and
produces a side-by-side table. Works with 1, 2, or 3 models present -- missing
models are simply skipped with a note, so you can compare as teammates finish.

    python compare.py

Outputs (into outputs/):
    comparison.csv    the table, spreadsheet-friendly
    comparison.md     the same table as Markdown (drop straight into the report)
    comparison_macro_f1.png   bar chart of macro-F1 per model (the headline)

Reminder: MACRO-F1 is the headline metric under heavy imbalance; accuracy is
shown too but is inflated by the dominant classes.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Columns pulled from each metrics.json, in display order.
_COLUMNS = [
    ("model", "model_name"),
    ("test_n", "n_samples"),
    ("accuracy", "overall.accuracy"),
    ("macro_f1", "overall.macro_f1"),          # <- headline
    ("weighted_f1", "overall.weighted_f1"),
    ("macro_precision", "overall.macro_precision"),
    ("macro_recall", "overall.macro_recall"),
    ("val_macro_f1", "checkpoint.best_val_macro_f1"),
]


def _dig(d: dict, dotted: str):
    """Fetch a nested value by 'a.b.c' path; return None if absent."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect_rows() -> list[dict]:
    """Read every outputs/<model>/metrics.json into a flat row dict."""
    rows = []
    for metrics_path in sorted(config.OUTPUTS_ROOT.glob("*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        row = {col: _dig(data, path) for col, path in _COLUMNS}
        rows.append(row)
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    headers = [c for c, _ in _COLUMNS]
    widths = {h: max(len(h), max((len(_fmt(r[h])) for r in rows), default=0)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print("\n" + line)
    print("-" * len(line))
    # Sort by macro_f1 descending (None last).
    for r in sorted(rows, key=lambda r: (r["macro_f1"] is not None, r["macro_f1"] or 0),
                    reverse=True):
        print("  ".join(_fmt(r[h]).ljust(widths[h]) for h in headers))


def save_csv(rows: list[dict], out_path: Path) -> None:
    headers = [c for c, _ in _COLUMNS]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: _fmt(r[h]) for h in headers})


def save_markdown(rows: list[dict], out_path: Path) -> None:
    headers = [c for c, _ in _COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in sorted(rows, key=lambda r: (r["macro_f1"] is not None, r["macro_f1"] or 0),
                    reverse=True):
        lines.append("| " + " | ".join(_fmt(r[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_bar_chart(rows: list[dict], out_path: Path) -> None:
    """Bar chart of macro-F1 (and accuracy for context) per model."""
    named = [r for r in rows if r["macro_f1"] is not None]
    if not named:
        return
    models = [r["model"] for r in named]
    macro = [r["macro_f1"] for r in named]
    acc = [r["accuracy"] for r in named]

    x = range(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 3, 4))
    ax.bar([i - width / 2 for i in x], macro, width, label="macro-F1 (headline)")
    ax.bar([i + width / 2 for i in x], acc, width, label="accuracy", alpha=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Model comparison (test set)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    rows = collect_rows()
    if not rows:
        print("No metrics found. Run evaluate.py for at least one model first:\n"
              "    python evaluate.py --model custom_cnn")
        return

    found = {r["model"] for r in rows}
    expected = {"custom_cnn", "transfer_cnn", "transformer"}
    missing = expected - found
    print(f"Found metrics for: {sorted(found)}")
    if missing:
        print(f"(Not yet evaluated: {sorted(missing)} -- table will fill in as "
              "teammates finish.)")

    print_table(rows)

    out = config.OUTPUTS_ROOT
    save_csv(rows, out / "comparison.csv")
    save_markdown(rows, out / "comparison.md")
    save_bar_chart(rows, out / "comparison_macro_f1.png")
    print(f"\nSaved -> {out / 'comparison.csv'}, comparison.md, comparison_macro_f1.png")


if __name__ == "__main__":
    main()
