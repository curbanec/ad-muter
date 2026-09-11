#!/usr/bin/env python3
"""Score the real detector + controller against annotated chunks.

Window-level accuracy is the wrong lens for this system. The controller only
mutes after ``confirm_windows`` consecutive ad-like windows and holds the mute
through ``min_ad_seconds``, so isolated bad windows never reach the TV. What
matters is the mute spans that come out the other end.

This runs the actual HeuristicDetector and Controller over a session's chunks —
in timestamp order, with state carried across chunk boundaries the way the live
service experiences it — and compares the resulting mute spans to the labels.

    python scripts/score_detector.py -i samples/movie
    python scripts/score_detector.py -i samples/movie --set detection.baseline_alpha=0.01
    python scripts/score_detector.py -i samples/movie --sweep detection.ad_crest_delta_db=-99,0,2
    python scripts/score_detector.py -i samples/movie -i samples/sitcom   # pooled
    python scripts/score_detector.py --loso samples/                      # per session

Three numbers decide whether it works:

* **breaks caught** — how many labelled breaks produced a mute at all.
* **latency** — seconds from break start to mute. Your budget is 1-2s.
* **false mute seconds** — audio muted that was not an ad. This is the one that
  ruins the experience; a missed ad is merely annoying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import AudioWindow, wav_windows  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402
from admuter.controller import Controller  # noqa: E402
from admuter.detector import HeuristicDetector  # noqa: E402
from admuter.logging_setup import setup_logging  # noqa: E402
from admuter.ml_detector import load_voter  # noqa: E402
from admuter.transcript import load_transcript_voter  # noqa: E402
from admuter.fingerprint import load_fingerprint_voter  # noqa: E402

from build_dataset import (  # noqa: E402
    AD_LABELS,
    DatasetError,
    Span,
    pair_chunks,
    parse_labels,
    read_session,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
# A mute that starts slightly before a break, or lingers slightly after, is not
# really a mistake — the seam itself is dead air either way.
EDGE_GRACE_SECONDS = 2.0


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


@dataclass
class RecordingRoku:
    """Stands in for the TV; records when a mute would have started/stopped."""

    is_muted: bool = False
    timestamp: float = 0.0
    spans: list[list[float]] = field(default_factory=list)

    def mute(self) -> bool:
        if not self.is_muted:
            self.is_muted = True
            self.spans.append([self.timestamp, self.timestamp])
        return True

    def unmute(self) -> bool:
        if self.is_muted:
            self.is_muted = False
            self.spans[-1][1] = self.timestamp
        return True

    def active_app(self):
        # Never consulted: the app gate is forced off for offline replay, since
        # there is no TV to ask and the annotations already say what was on.
        return None

    def close(self) -> None:
        pass


def wav_duration(path) -> float:
    """A chunk's exact length in seconds.

    Both clocks in this file -- the one that timestamps windows and the one
    that places label spans -- advance by this and only this. They used to
    disagree: the window clock credited the final, partial window of each chunk
    a whole window_seconds, while the label clock used the exact frame count.
    Every chunk that is not a whole number of windows pushed them apart, and
    the error accumulated across a session, so labels and mutes were compared
    on two different timelines.
    """
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


class SessionSource:
    """Every chunk's windows, renumbered onto one continuous session clock."""

    def __init__(self, pairs, window_seconds: float, roku: RecordingRoku) -> None:
        self.pairs = pairs
        self.window_seconds = window_seconds
        self.roku = roku
        self.duration = 0.0

    def windows(self):
        elapsed = 0.0
        index = 0
        for wav, _ in self.pairs:
            for window in wav_windows(wav, self.window_seconds):
                t = elapsed + window.timestamp
                self.roku.timestamp = t
                yield AudioWindow(
                    samples=window.samples,
                    sample_rate=window.sample_rate,
                    index=index,
                    timestamp=t,
                    # Only the very first window is a genuine stream start; a
                    # chunk boundary is a recording artifact, not a restart.
                    stream_restarted=index == 0,
                )
                index += 1
            # Exact chunk length, not the last window's nominal end.
            elapsed += wav_duration(wav)
        self.duration = elapsed

    def stop(self) -> None:
        pass


def session_spans(pairs, window_seconds: float) -> list[Span]:
    """Label spans shifted onto the same continuous session clock."""
    spans: list[Span] = []
    elapsed = 0.0
    for wav, label_path in pairs:
        for span in parse_labels(label_path):
            if span.label in AD_LABELS:
                spans.append(Span(span.start + elapsed, span.end + elapsed, "ad"))
        elapsed += wav_duration(wav)
    # Stitch spans that meet at a chunk boundary back into one break.
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start - merged[-1].end < 2.0:
            merged[-1] = Span(merged[-1].start, span.end, "ad")
        else:
            merged.append(span)
    return merged


def overlap(a: Span, b: tuple[float, float]) -> float:
    return max(0.0, min(a.end, b[1]) - max(a.start, b[0]))


def score(pairs, config: Config) -> dict:
    roku = RecordingRoku()
    source = SessionSource(pairs, config.audio.window_seconds, roku)
    detector = HeuristicDetector(
        config.detection,
        config.audio.window_seconds,
        ml_voter=load_voter(config.detection),
        # Synchronous: replay outruns realtime, so a threaded recogniser would
        # drop nearly every window and score nothing at all.
        transcript_voter=load_transcript_voter(config.detection, synchronous=True),
        fingerprint_voter=load_fingerprint_voter(config.detection),
    )
    controller = Controller(source, detector, roku, config)
    controller.run()

    duration = source.duration
    mutes = [(s[0], s[1] if s[1] > s[0] else duration) for s in roku.spans]
    breaks = session_spans(pairs, config.audio.window_seconds)

    ad_seconds = sum(b.end - b.start for b in breaks)
    muted_seconds = sum(e - s for s, e in mutes)
    covered = sum(sum(overlap(b, m) for m in mutes) for b in breaks)

    false_muted = 0.0
    for s, e in mutes:
        padded = [Span(b.start - EDGE_GRACE_SECONDS, b.end + EDGE_GRACE_SECONDS, "ad")
                  for b in breaks]
        inside = sum(overlap(p, (s, e)) for p in padded)
        false_muted += max(0.0, (e - s) - inside)

    caught = []
    for b in breaks:
        hits = [m for m in mutes if overlap(b, m) > 0]
        caught.append((b, min((m[0] for m in hits), default=None)))

    return {
        "duration": duration,
        "breaks": caught,
        "mutes": mutes,
        "ad_seconds": ad_seconds,
        "muted_seconds": muted_seconds,
        "covered": covered,
        "false_muted": false_muted,
    }


def report(result: dict, label: str = "") -> None:
    if label:
        print(f"\n=== {label} ===")
    breaks = result["breaks"]
    hit = sum(1 for _, start in breaks if start is not None)
    print(f"  breaks caught: {hit}/{len(breaks)}")
    for span, start in breaks:
        if start is None:
            print(f"    {clock(span.start)}-{clock(span.end)}  MISSED")
        else:
            print(
                f"    {clock(span.start)}-{clock(span.end)}  muted at "
                f"{clock(start)}  (latency {start - span.start:+.0f}s)"
            )
    ad = result["ad_seconds"]
    cov = 100.0 * result["covered"] / ad if ad else 0.0
    print(f"  ad audio muted:    {result['covered']:.0f}s of {ad:.0f}s ({cov:.0f}%)")
    print(f"  FALSE MUTE:        {result['false_muted']:.0f}s of content")
    extra = [m for m in result["mutes"]
             if all(not (m[0] < s.end and m[1] > s.start) for s, _ in breaks)]
    for s, e in extra:
        print(f"    spurious mute {clock(s)}-{clock(e)} ({e - s:.0f}s)")


def report_totals(results: list[tuple[str, dict]]) -> None:
    """Pool the per-session outcomes into the numbers you tune against.

    Sessions are separate recordings scored independently, so only the outcomes
    add up -- never the clocks. False mute is also given per hour: sessions
    differ in length, and a raw total silently weights the longest one.
    """
    breaks = [b for _, result in results for b in result["breaks"]]
    hit = sum(1 for _, start in breaks if start is not None)
    latencies = [start - span.start for span, start in breaks if start is not None]
    ad = sum(result["ad_seconds"] for _, result in results)
    covered = sum(result["covered"] for _, result in results)
    false_muted = sum(result["false_muted"] for _, result in results)
    duration = sum(result["duration"] for _, result in results)

    print(f"\n  == TOTAL: {len(results)} sessions, {duration / 60:.0f} min ==")
    print(f"    breaks caught:     {hit}/{len(breaks)}")
    if latencies:
        print(f"    latency:           median {statistics.median(latencies):+.0f}s"
              f"  worst {max(latencies):+.0f}s")
    cov = 100.0 * covered / ad if ad else 0.0
    print(f"    ad audio muted:    {covered:.0f}s of {ad:.0f}s ({cov:.0f}%)")
    per_hour = 3600.0 * false_muted / duration if duration else 0.0
    print(f"    FALSE MUTE:        {false_muted:.0f}s ({per_hour:.1f}s per hour)")


# --------------------------------------------------------------------------- #
# Leave-one-session-out
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionScore:
    """One session's numbers, kept per-session on purpose.

    Pooling hides the failure that matters. A config scoring well on four
    sessions and badly on a fifth is worse than one scoring moderately on all
    five, because the fifth is what a new service or genre looks like -- and
    the pooled average is dominated by whichever session ran longest.
    """

    session_id: str
    indir: str
    chunks: int
    breaks_caught: int
    breaks_total: int
    # Recall and precision are reported side by side and never combined. An F1
    # would let a config trade away the one that ruins the experience (muting
    # real dialogue) for the one that merely annoys (missing an ad).
    recall_pct: float
    precision_pct: float | None
    ad_seconds: float
    muted_seconds: float
    false_muted_seconds: float
    content_hours: float
    false_mute_per_content_hour: float
    duration_seconds: float


def discover_sessions(parent: Path) -> list[Path]:
    """Subdirectories of *parent* that hold audio. Anything else is not ours."""
    if not parent.is_dir():
        raise DatasetError(f"--loso: {parent} is not a directory")
    found = [d for d in sorted(parent.iterdir()) if d.is_dir() and any(d.glob("*.wav"))]
    if not found:
        raise DatasetError(f"--loso: no subdirectory of {parent} contains a .wav")
    return found


def has_ad_spans(pairs) -> bool:
    """Does this session label any ads at all?

    An empty label file legitimately means "no ads in this chunk", so a whole
    session of them is a session with nothing to score -- recall is undefined
    and false mute has no breaks to be measured against. Skip it loudly rather
    than reporting a meaningless 0/0.
    """
    return any(
        span.label in AD_LABELS
        for _, label_path in pairs
        for span in parse_labels(label_path)
    )


def summarise(session_id: str, indir: Path, chunks: int, result: dict) -> SessionScore:
    """Turn one score() result into the per-session row, changing nothing."""
    ad = result["ad_seconds"]
    muted = result["muted_seconds"]
    duration = result["duration"]
    breaks = result["breaks"]
    # False mute is normalised against *content* time, not wall time: a session
    # that is one-fifth ads has less content to be wrong about, and dividing by
    # the whole duration would flatter it.
    content_hours = max(duration - ad, 0.0) / 3600.0
    return SessionScore(
        session_id=session_id,
        indir=str(indir),
        chunks=chunks,
        breaks_caught=sum(1 for _, start in breaks if start is not None),
        breaks_total=len(breaks),
        recall_pct=100.0 * result["covered"] / ad if ad else 0.0,
        precision_pct=100.0 * result["covered"] / muted if muted else None,
        ad_seconds=ad,
        muted_seconds=muted,
        false_muted_seconds=result["false_muted"],
        content_hours=content_hours,
        false_mute_per_content_hour=(
            result["false_muted"] / content_hours if content_hours else 0.0
        ),
        duration_seconds=duration,
    )


def report_loso(rows: list[SessionScore]) -> None:
    """One row per session, then the spread. The spread is the headline."""
    print(f"\n  {'session':<30}{'breaks':>9}{'recall':>9}{'prec':>8}"
          f"{'false mute':>13}{'per content hr':>16}")
    print(f"  {'-' * 84}")
    # Worst first: the session at the top is the one the config has to answer
    # for, and reading order should not depend on how the directories sorted.
    for r in sorted(rows, key=lambda r: r.false_mute_per_content_hour, reverse=True):
        prec = f"{r.precision_pct:.0f}%" if r.precision_pct is not None else "n/a"
        print(
            f"  {r.session_id:<30}"
            f"{f'{r.breaks_caught}/{r.breaks_total}':>9}"
            f"{r.recall_pct:>8.0f}%"
            f"{prec:>8}"
            f"{r.false_muted_seconds:>12.0f}s"
            f"{r.false_mute_per_content_hour:>15.0f}s"
        )

    per_hour = [r.false_mute_per_content_hour for r in rows]
    worst = max(rows, key=lambda r: r.false_mute_per_content_hour)
    best = min(rows, key=lambda r: r.false_mute_per_content_hour)
    spread = worst.false_mute_per_content_hour - best.false_mute_per_content_hour
    print(f"\n  TUNE AGAINST THIS -- worst session, not the average:")
    print(f"    worst:  {worst.session_id} at "
          f"{worst.false_mute_per_content_hour:.0f}s false mute per content hour")
    print(f"    spread: {spread:.0f}s per content hour across {len(rows)} sessions "
          f"(best {best.session_id} at {min(per_hour):.0f}s)")
    caught = sum(r.breaks_caught for r in rows)
    total = sum(r.breaks_total for r in rows)
    worst_recall = min(rows, key=lambda r: r.recall_pct)
    print(f"    breaks: {caught}/{total} overall, worst session "
          f"{worst_recall.session_id} at {worst_recall.recall_pct:.0f}% recall")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def write_result_file(
    rows: list[SessionScore], config_path: Path, label: str, outdir: Path
) -> Path:
    """Stamp the run so a number can be traced back to the code that made it."""
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = outdir / f"{stamp}.json"
    try:
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError:
        config_sha = None
    per_hour = [r.false_mute_per_content_hour for r in rows]
    payload = {
        "timestamp": stamp,
        "git_sha": _git_sha(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_label": label,
        "session_ids": [r.session_id for r in rows],
        "sessions": [asdict(r) for r in rows],
        "worst_false_mute_per_content_hour": max(per_hour) if per_hour else None,
        "spread_false_mute_per_content_hour": (
            max(per_hour) - min(per_hour) if per_hour else None
        ),
    }
    # A colliding timestamp means two runs in the same second; keep both.
    suffix = 1
    while path.exists():
        path = outdir / f"{stamp}-{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    import dataclasses
    sections = {n: dataclasses.asdict(getattr(config, n))
                for n in ("audio", "detection", "controller", "roku", "logging")}
    for override in overrides:
        dotted, _, raw = override.partition("=")
        section, _, key = dotted.partition(".")
        if section not in sections or key not in sections[section]:
            raise SystemExit(f"--set: unknown option {dotted!r}")
        current = sections[section][key]
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        sections[section][key] = value
    return Config.from_dict(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, action="append", dest="indirs",
                        metavar="DIR",
                        help="directory of WAVs + .ads.txt files; repeat the flag "
                             "to score several sessions together")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value")
    parser.add_argument("--sweep", metavar="section.key=v1,v2,v3",
                        help="score once per value of one setting")
    parser.add_argument("--loso", type=Path, metavar="PARENT",
                        help="leave-one-session-out: score every session "
                             "subdirectory of PARENT on its own and report the "
                             "spread. The number to tune against is the worst "
                             "session, not the pooled average")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        metavar="DIR",
                        help="where --loso writes its stamped JSON "
                             "(default: results/)")
    parser.add_argument("--log-level", default="ERROR",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.loso is not None and args.indirs:
        print("error: --loso and -i are mutually exclusive", file=sys.stderr)
        return 2

    indirs = args.indirs or [Path(".")]
    try:
        # The app gate guards the live service against muting things it was
        # never tuned on. Offline there is no TV to query, so force it off --
        # last, so it cannot be re-enabled by a --set and silently score zeros.
        base = apply_overrides(
            Config.load(args.config), args.overrides + ["roku.armed_apps_only=false"]
        )
        if args.loso is not None:
            loso: list[tuple[str, Path, list]] = []
            for d in discover_sessions(args.loso):
                try:
                    pairs = pair_chunks(d)
                except DatasetError as exc:
                    print(f"warning: skipping {d.name} — {exc}", file=sys.stderr)
                    continue
                if not has_ad_spans(pairs):
                    print(f"warning: skipping {d.name} — no labelled ad breaks",
                          file=sys.stderr)
                    continue
                sid = read_session(d).get("session_id") or d.resolve().name
                loso.append((sid, d, pairs))
            if not loso:
                raise DatasetError(f"no scorable session under {args.loso}")
            sessions = [(sid, pairs) for sid, _, pairs in loso]
        else:
            # One entry per directory. Each is its own recording, so score()
            # below gets a fresh detector and baseline for each; carrying state
            # across unrelated sessions would let one night's audio set the
            # other's floor.
            sessions = [
                (read_session(d).get("session_id") or d.resolve().name, pair_chunks(d))
                for d in indirs
            ]
    except (ConfigError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    chunks = sum(len(pairs) for _, pairs in sessions)
    print(f"{chunks} chunks from {len(sessions)} session(s)")

    def run_loso(config: Config, label: str) -> None:
        if label:
            print(f"\n=== {label} ===")
        rows: list[SessionScore] = []
        for sid, d, pairs in loso:
            print(f"\n  [{sid}]")
            result = score(pairs, config)
            report(result)
            rows.append(summarise(sid, d, len(pairs), result))
        report_loso(rows)
        print(f"\n  wrote {write_result_file(rows, args.config, label, args.results_dir)}")

    def run_pooled(config: Config, label: str) -> None:
        results = [(name, score(pairs, config)) for name, pairs in sessions]
        if len(results) == 1:
            report(results[0][1], label)
            return
        if label:
            print(f"\n=== {label} ===")
        for name, result in results:
            print(f"\n  [{name}]")
            report(result)
        report_totals(results)

    run = run_loso if args.loso is not None else run_pooled

    if not args.sweep:
        run(base, "current config")
        return 0

    dotted, _, sweep_values = args.sweep.partition("=")
    values = sweep_values.split(",")
    skipped = 0
    for value in values:
        try:
            config = apply_overrides(base, [f"{dotted}={value}"])
        except ValueError as exc:
            # ConfigError subclasses ValueError, so this covers both a value the
            # validator rejects and a scalar that will not coerce. A sweep that
            # straddles a validation boundary should still report the values
            # that are valid rather than dying partway with a traceback.
            print(f"\n=== {dotted}={value} ===")
            print(f"  SKIPPED — {exc}")
            skipped += 1
            continue
        run(config, f"{dotted}={value}")
    if skipped:
        print(f"\n{skipped} of {len(values)} values skipped as invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
