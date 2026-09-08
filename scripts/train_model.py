#!/usr/bin/env python3
"""Train the Phase 2 per-window classifier and report how it generalizes.

    python scripts/train_model.py data/*.csv
    python scripts/train_model.py data/*.csv --out models/admuter.joblib

Evaluation is leave-one-session-out, always. A random train/test split on this
data is not a weak measure, it is a broken one: rows are consecutive 1-second
windows, so window N and window N+1 come from the same second of the same ad
and differ by almost nothing. Shuffling puts one in train and the other in
test, and the model scores brilliantly for having memorised its own training
set. Grouping by ``session_id`` is the only split that answers the question
that matters — how does this do on a recording it has never seen?

The feature set is fixed at four columns and deliberately small. They are the
ones whose direction is consistent across every annotated session; the spectral
and silence-timing features invert between genres, so a model that leans on
them fits the corpus rather than the problem.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.config import Config, ConfigError  # noqa: E402
from admuter.ml_detector import SCHEMA_VERSION, validate_columns  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

# Order is part of the saved contract: the scaler and the coefficients are
# positional, so this list is written to the bundle and read back verbatim.
FEATURE_COLUMNS = ["rms_dbfs", "crest_db", "delta_rms_db", "delta_crest_db"]

THRESHOLDS = (0.5, 0.7, 0.85, 0.9)


class TrainingError(RuntimeError):
    """Anything that should stop training rather than produce a quiet lie."""


def load_rows(paths: list[Path], baseline_min_windows: int) -> tuple:
    """Read the CSVs into X, y, groups — dropping windows we cannot trust.

    Rows below ``baseline_min_windows`` are discarded because two of the four
    features are differences against a baseline that has not settled yet. They
    are not missing data, they are wrong data, and the live detector refuses to
    fire on them for the same reason.
    """
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    dropped_warmup = 0
    dropped_missing = 0

    for path in paths:
        if not path.exists():
            raise TrainingError(f"no such dataset: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("label") not in ("ad", "content"):
                    continue
                count = row.get("baseline_count") or ""
                if not count or int(float(count)) < baseline_min_windows:
                    dropped_warmup += 1
                    continue
                try:
                    values = [float(row[c]) for c in FEATURE_COLUMNS]
                except (KeyError, TypeError, ValueError):
                    dropped_missing += 1
                    continue
                if not all(np.isfinite(v) for v in values):
                    dropped_missing += 1
                    continue
                X.append(values)
                y.append(1 if row["label"] == "ad" else 0)
                groups.append(row["session_id"])

    if not X:
        raise TrainingError("no usable rows after filtering")
    print(
        f"{len(X)} rows from {len(set(groups))} sessions "
        f"({sum(y)} ad / {len(y) - sum(y)} content, {sum(y) / len(y):.1%} positive)"
    )
    if dropped_warmup:
        print(f"  dropped {dropped_warmup} rows below baseline_count "
              f"{baseline_min_windows} (deltas unreliable)")
    if dropped_missing:
        print(f"  dropped {dropped_missing} rows with missing/non-finite features")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(groups)


def make_pipeline():
    """Scaler + logistic regression.

    Scaling matters here even though the model is linear: dBFS values sit
    around -25 while the deltas hover near 0, and an unscaled fit lets the
    large-magnitude column dominate the regularisation penalty.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        # class_weight balances a ~12% positive rate. Without it the model can
        # score 88% accuracy by answering "content" to everything.
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def evaluate_loso(X, y, groups) -> list[dict]:
    """Train on every session but one, test on the one left out. Repeat."""
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    folds = []
    for held_out in sorted(set(groups)):
        test = groups == held_out
        train = ~test
        if len(set(y[train])) < 2 or len(set(y[test])) < 2:
            print(f"  skipping fold {held_out}: only one class present")
            continue
        pipeline = make_pipeline()
        pipeline.fit(X[train], y[train])
        probability = pipeline.predict_proba(X[test])[:, 1]
        fold = {
            "session_id": held_out,
            "n": int(test.sum()),
            "positives": int(y[test].sum()),
            "auc": float(roc_auc_score(y[test], probability)),
            "at": {},
        }
        for threshold in THRESHOLDS:
            predicted = (probability >= threshold).astype(int)
            fold["at"][threshold] = {
                "precision": float(
                    precision_score(y[test], predicted, zero_division=0)
                ),
                "recall": float(recall_score(y[test], predicted, zero_division=0)),
                "predicted_positive": int(predicted.sum()),
            }
        folds.append(fold)
    return folds


def report(folds: list[dict]) -> None:
    if not folds:
        print("no folds to report")
        return
    print(f"\n{'leave-one-session-out':<34}{'n':>7}{'ads':>7}{'AUC':>8}")
    print("-" * 56)
    for f in folds:
        print(f"{f['session_id']:<34}{f['n']:>7}{f['positives']:>7}{f['auc']:>8.3f}")
    aucs = [f["auc"] for f in folds]
    print(f"{'mean / worst':<34}{'':>7}{'':>7}"
          f"{sum(aucs) / len(aucs):>8.3f}  (worst {min(aucs):.3f})")

    for threshold in THRESHOLDS:
        print(f"\n  threshold {threshold:.2f}")
        print(f"    {'session':<32}{'precision':>11}{'recall':>9}")
        for f in folds:
            at = f["at"][threshold]
            print(f"    {f['session_id']:<32}{at['precision']:>10.1%}{at['recall']:>9.1%}")
        precisions = [f["at"][threshold]["precision"] for f in folds]
        recalls = [f["at"][threshold]["recall"] for f in folds]
        print(f"    {'worst session':<32}{min(precisions):>10.1%}{min(recalls):>9.1%}")


def save(pipeline, session_ids: list[str], out: Path) -> Path:
    import joblib
    import sklearn

    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        # Verbatim, and read back verbatim: the scaler and coefficients are
        # positional, so a reordered list predicts confidently and wrongly.
        "feature_columns": list(FEATURE_COLUMNS),
        "training_session_ids": sorted(session_ids),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    joblib.dump(bundle, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", type=Path,
                        help="dataset CSVs from scripts/build_dataset.py")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--out", type=Path, default=Path("models/admuter.joblib"))
    parser.add_argument("--no-save", action="store_true",
                        help="evaluate only; do not write a model file")
    args = parser.parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # Fails loudly now rather than at load time on the Pi.
    validate_columns(FEATURE_COLUMNS)

    try:
        X, y, groups = load_rows(
            args.datasets, config.detection.baseline_min_windows
        )
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    folds = evaluate_loso(X, y, groups)
    report(folds)

    if args.no_save:
        return 0
    pipeline = make_pipeline()
    pipeline.fit(X, y)
    path = save(pipeline, sorted(set(groups.tolist())), args.out)
    coefficients = pipeline.named_steps["model"].coef_[0]
    print(f"\nfinal model trained on all {len(set(groups))} sessions -> {path}")
    print("  standardised coefficients (sign says which way the feature votes):")
    for name, weight in zip(FEATURE_COLUMNS, coefficients):
        print(f"    {name:<18}{weight:>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
