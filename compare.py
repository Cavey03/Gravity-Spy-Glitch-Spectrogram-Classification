"""Combine the models' metrics.json files into one comparison.

    python compare.py

Writes outputs/comparison/: comparison.md, comparison.csv, model_comparison.png

Refuses to build a table from models evaluated on different splits or with
different blank-image settings -- those would be different test sets, and it is
an easy mistake when three people run things independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, Config
from utils.plotting import plot_model_comparison

NON_MODEL_DIRS = {"comparison", "data_inspection"}
HEADLINE = [("macro_f1", "macro-F1"), ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced acc."), ("weighted_f1", "weighted-F1"),
            ("macro_precision", "macro-P"), ("macro_recall", "macro-R")]


def discover(output_root: Path, wanted: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    names = wanted or sorted(d.name for d in output_root.iterdir()
                             if d.is_dir() and d.name not in NON_MODEL_DIRS)
    results, missing = {}, []
    for name in names:
        path = output_root / name / "metrics.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        elif wanted:
            missing.append(name)
    if missing:
        print("[compare] no metrics.json for: %s (run evaluate.py)" % ", ".join(missing))
    return results


def check_consistent(results: Dict[str, Dict[str, Any]]) -> str:
    """Every model must have been scored on the same test set."""
    for field, label in (("split", "split"), ("drop_blank_images", "drop-blank-images")):
        values = {n: m.get(field) for n, m in results.items()}
        if len(set(values.values())) > 1:
            raise SystemExit(
                "Refusing to compare: models disagree on %s, so they were not "
                "evaluated on the same test set.\n" % label
                + "\n".join("  %-16s %s" % kv for kv in sorted(values.items()))
                + "\nRe-run evaluate.py so every model matches.")
    return next(iter(results.values())).get("split", "unknown")


def markdown_table(results: Dict[str, Dict[str, Any]], split: str) -> str:
    order = sorted(results, key=lambda n: -results[n]["macro_f1"])
    lines = ["| Model | " + " | ".join(l for _, l in HEADLINE) + " | Params | Epochs |",
             "|---|" + "---:|" * (len(HEADLINE) + 2)]
    for name in order:
        m = results[name]
        lines.append("| `%s` | %s | %s | %s |"
                     % (name,
                        " | ".join("%.4f" % m.get(k, 0.0) for k, _ in HEADLINE),
                        "{:,}".format(m.get("n_parameters", 0)),
                        m.get("epochs_trained", "-")))
    return "\n".join(lines) + (
        "\n\n*Evaluated on the shared **%s** split, sorted by macro-F1.*\n\n"
        "**Read macro-F1, not accuracy.** The class distribution is skewed "
        "enough that accuracy stays around 97%% regardless of how the rare "
        "classes are handled.\n" % split)


def per_class_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    names = sorted(results, key=lambda n: -results[n]["macro_f1"])
    support = {c: next((m["per_class"][c]["support"] for m in results.values()
                        if c in m["per_class"]), 0) for c in CLASSES}
    lines = ["| Class | Support | " + " | ".join("`%s` F1" % n for n in names) + " |",
             "|---|---:|" + "---:|" * len(names)]
    for cls in sorted(CLASSES, key=lambda c: -support[c]):
        cells = " | ".join("%.3f" % results[n]["per_class"][cls]["f1"] for n in names)
        lines.append("| %s | %d | %s |" % (cls, support[cls], cells))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare evaluated models.")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--output-root", type=Path, default=None)
    args = p.parse_args(argv)

    output_root = Path(args.output_root or Config().output_root)
    results = discover(output_root, args.models)
    if not results:
        print("error: no evaluated models under %s" % output_root, file=sys.stderr)
        return 2

    split = check_consistent(results)
    out_dir = output_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = markdown_table(results, split)
    (out_dir / "comparison.md").write_text(
        "# Gravity Spy -- model comparison\n\n## Headline metrics\n\n" + table
        + "\n## Per-class F1\n\n" + per_class_markdown(results) + "\n",
        encoding="utf-8")

    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + [k for k, _ in HEADLINE] + ["n_parameters", "epochs"])
        for name in sorted(results, key=lambda n: -results[n]["macro_f1"]):
            m = results[name]
            w.writerow([name] + ["%.6f" % m.get(k, 0.0) for k, _ in HEADLINE]
                       + [m.get("n_parameters", ""), m.get("epochs_trained", "")])

    plot_model_comparison(results, out_dir / "model_comparison.png")

    print(table)
    print("written to %s" % out_dir)
    if len(results) < 3:
        print("\n(%d of 3 models evaluated so far.)" % len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
