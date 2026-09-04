#!/usr/bin/env python3
"""Turn annotated WAV chunks into a per-window training CSV.

Each chunk is replayed through the *same* window slicer and feature extractor
the live service uses, so a row here is bit-for-bit what the detector would have
seen at that moment. Anything else and the classifier trains on one distribution
and gets deployed against another, which fails silently.

Labels come from Audacity's exported label files: ``start<TAB>end<TAB>text``, in
seconds from the start of that chunk. A window is an ad when its *midpoint*
falls inside an ``ad`` span; ``skip`` spans are dropped entirely.

    python scripts/build_dataset.py -i samples/movie -o dataset.csv
    python scripts/build_dataset.py -i samples/movie --stats-only

Every WAV must have a matching ``.ads.txt``. A missing one is an error, not an
empty label set: "I forgot this chunk" and "this chunk has no ads" must not look
the same. Export an empty file to mean the latter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import wav_windows  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402
from admuter.detector import Baseline  # noqa: E402
from admuter.features import compute_features  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
LABEL_SUFFIX = ".ads.txt"
AD_LABELS = {"ad", "ad-cont", "advert", "commercial"}
SKIP_LABELS = {"skip", "ignore", "unknown", "?"}


class DatasetError(RuntimeError):
    """Anything that should stop the build rather than produce a quiet lie."""


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    label: str

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def parse_labels(path: Path) -> list[Span]:
    """Read an Audacity label export. An empty file is valid: no ads."""
    spans: list[Span] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            # Audacity always writes three columns; anything else is a file
            # that was hand-edited or exported from something we don't know.
            raise DatasetError(
                f"{path.name}:{lineno}: expected 'start<TAB>end<TAB>label', "
                f"got {line!r}"
            )
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise DatasetError(f"{path.name}:{lineno}: bad timestamp ({exc})") from exc
        label = parts[2].strip().lower()
        if end < start:
            raise DatasetError(
                f"{path.name}:{lineno}: end {end} precedes start {start}"
            )
        if label not in AD_LABELS and label not in SKIP_LABELS:
            raise DatasetError(
                f"{path.name}:{lineno}: unrecognised label {parts[2]!r}; "
                f"expected one of {sorted(AD_LABELS | SKIP_LABELS)}"
            )
        spans.append(Span(start, end, label))

    spans.sort(key=lambda s: s.start)
    for a, b in zip(spans, spans[1:]):
        if b.start < a.end:
            raise DatasetError(
                f"{path.name}: spans overlap ({a.start:.3f}-{a.end:.3f} and "
                f"{b.start:.3f}-{b.end:.3f}); a window cannot be two classes"
            )
    return spans


def classify(midpoint: float, spans: list[Span]) -> str:
    """'ad', 'skip', or 'content' for a window centred at `midpoint`."""
    for span in spans:
        if span.contains(midpoint):
            return "skip" if span.label in SKIP_LABELS else "ad"
    return "content"


def pair_chunks(indir: Path) -> list[tuple[Path, Path]]:
    """Match every WAV with its label file, or fail loudly."""
    wavs = sorted(indir.glob("*.wav"))
    if not wavs:
        raise DatasetError(f"no .wav files in {indir}")
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for wav in wavs:
        labels = wav.with_suffix("").with_suffix(LABEL_SUFFIX)
        if not labels.exists():
            # Also accept the plain stem form, e.g. 20260816T152023.ads.txt
            alt = wav.parent / (wav.stem + LABEL_SUFFIX)
            labels = alt if alt.exists() else labels
        if not labels.exists():
            missing.append(wav.name)
        else:
            pairs.append((wav, labels))
    if missing:
        raise DatasetError(
            "no label file for: " + ", ".join(missing) + "\n"
            "Export an empty label file to declare a chunk ad-free; a missing "
            "file means it was never annotated."
        )
    return pairs


def read_session(indir: Path) -> dict:
    path = indir / "session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"session.json is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def blank_if_none(value: float | None) -> object:
    """Undefined baselines become empty cells, which pandas reads as NaN.

    Emitting 0.0 would be a lie the model would happily learn from.
    """
    return "" if value is None else round(float(value), 6)


@dataclass
class SessionState:
    """Causal state that survives chunk boundaries.

    Chunking is an artifact of ``--max-file-time``, not something the deployed
    service ever experiences: it sees one unbroken stream. Resetting the
    baseline every ten minutes would manufacture twelve warm-up periods that
    exist nowhere in production — and worse, a chunk that opens mid-ad would
    seed its baseline from ad audio, inverting the sign of the deltas that
    matter most. So this is threaded through the chunks in timestamp order and
    reset only between sessions.
    """

    baseline: Baseline
    last_gap_at: float | None = None
    elapsed: float = 0.0  # seconds of audio consumed by earlier chunks


def build_rows(
    wav: Path,
    spans: list[Span],
    config: Config,
    session_id: str,
    state: SessionState,
) -> tuple[list[dict], dict[str, int]]:
    """Replay one chunk into labelled rows plus a per-class tally."""
    detection = config.detection
    window_seconds = config.audio.window_seconds
    baseline = state.baseline
    rows: list[dict] = []
    counts = {"ad": 0, "content": 0, "skip": 0}
    last_gap_at = state.last_gap_at
    consumed = 0.0

    for window in wav_windows(wav, window_seconds):
        features = compute_features(
            window.samples,
            window.sample_rate,
            silence_dbfs=detection.silence_dbfs,
            frame_seconds=config.audio.frame_seconds,
        )
        midpoint = window.timestamp + features.duration_seconds / 2.0
        label = classify(midpoint, spans)
        counts[label] += 1

        # --- deltas against the causal baseline (state BEFORE this window) ---
        delta_rms = delta_crest = delta_centroid = None
        if baseline.count > 0:
            delta_rms = features.rms_dbfs - float(baseline.rms_dbfs)
            # Detector sign convention: positive means squashed vs baseline.
            delta_crest = float(baseline.crest_db) - features.crest_db
            if float(baseline.centroid_hz) > 1.0:
                delta_centroid = (
                    features.spectral_centroid_hz - float(baseline.centroid_hz)
                ) / float(baseline.centroid_hz)

        # Gap timing is measured in session time, so a seam near the end of one
        # chunk still counts for the opening windows of the next.
        session_seconds = state.elapsed + window.timestamp
        if features.max_silence_run_seconds >= detection.min_gap_seconds:
            last_gap_at = session_seconds
        since_gap = (
            None if last_gap_at is None else session_seconds - last_gap_at
        )

        if label != "skip":
            row: dict = {
                "session_id": session_id,
                "chunk": wav.name,
                "window_index": window.index,
                "timestamp": round(window.timestamp, 3),
                "session_seconds": round(session_seconds, 3),
                "label": label,
                "is_ad": int(label == "ad"),
            }
            row.update(
                {
                    k: (int(v) if isinstance(v, bool) else round(float(v), 6))
                    for k, v in features.as_dict().items()
                }
            )
            row["baseline_rms_dbfs"] = blank_if_none(baseline.rms_dbfs)
            row["baseline_crest_db"] = blank_if_none(baseline.crest_db)
            row["baseline_centroid_hz"] = blank_if_none(baseline.centroid_hz)
            row["baseline_count"] = baseline.count
            row["delta_rms_db"] = blank_if_none(delta_rms)
            row["delta_crest_db"] = blank_if_none(delta_crest)
            row["delta_centroid_ratio"] = blank_if_none(delta_centroid)
            row["seconds_since_gap"] = blank_if_none(since_gap)
            rows.append(row)

        # --- advance causal state AFTER the row is emitted ---
        # The baseline updates on every non-silent window regardless of label:
        # a deployed classifier has no ad-state feedback, so training must not
        # assume any. See the note in the module docstring's sibling README.
        if not features.is_silence:
            baseline.update(features, detection.baseline_alpha)
        consumed = window.timestamp + features.duration_seconds

    state.last_gap_at = last_gap_at
    state.elapsed += consumed
    return rows, counts


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, default=Path("."),
                        help="directory of WAVs + .ads.txt files")
    parser.add_argument("-o", "--out", type=Path, default=Path("dataset.csv"))
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--session-id", help="override session.json / directory name")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing CSV instead of replacing it")
    parser.add_argument("--stats-only", action="store_true",
                        help="report the class balance and exit without writing")
    args = parser.parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        pairs = pair_chunks(args.indir)
        session = read_session(args.indir)
        session_id = (
            args.session_id
            or session.get("session_id")
            or args.indir.resolve().name
        )

        all_rows: list[dict] = []
        totals = {"ad": 0, "content": 0, "skip": 0}
        # One baseline for the whole session; pair_chunks returns the WAVs in
        # filename order, which for %Y%m%dT%H%M%S stamps is chronological.
        state = SessionState(baseline=Baseline())
        print(f"session {session_id}: {len(pairs)} chunks")
        for wav, label_path in pairs:
            spans = parse_labels(label_path)
            rows, counts = build_rows(wav, spans, config, session_id, state)
            for key, value in counts.items():
                totals[key] += value
            ad_spans = [s for s in spans if s.label in AD_LABELS]
            print(
                f"  {wav.name}: {counts['ad']:>4} ad / {counts['content']:>4} "
                f"content / {counts['skip']:>3} skip   ({len(ad_spans)} span(s))"
            )
            all_rows.extend(rows)
    except DatasetError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    kept = totals["ad"] + totals["content"]
    if kept == 0:
        print("no usable windows", file=sys.stderr)
        return 1
    share = 100.0 * totals["ad"] / kept
    print(
        f"\n{kept} windows kept ({totals['skip']} skipped): "
        f"{totals['ad']} ad / {totals['content']} content — {share:.1f}% positive"
    )
    if share < 10.0:
        print(
            "  imbalanced: a model predicting 'content' every time scores "
            f"{100 - share:.1f}% accuracy. Use precision/recall, not accuracy."
        )
    warm = config.detection.baseline_min_windows
    cold = sum(1 for r in all_rows if r["baseline_count"] < warm)
    if cold:
        print(
            f"  {cold} rows have baseline_count < {warm} (deltas unreliable); "
            "filter on baseline_count before training."
        )

    if args.stats_only:
        return 0

    mode = "a" if args.append and args.out.exists() else "w"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {args.out} ({len(all_rows)} rows, {len(all_rows[0])} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())