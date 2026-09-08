#!/usr/bin/env python3
"""Sweep one config value and score every session separately at each setting.

``score_detector.py --sweep`` already varies a setting, but it reports the
pooled total, and a pooled total is the wrong lens for a threshold decision:
it is dominated by whichever session ran longest and it hides the session that
fails. This drives ``--loso`` instead, so every value is judged by its *worst*
session.

    python scripts/sweep_loso.py --loso ~/Movies
    python scripts/sweep_loso.py --loso ~/Movies \\
        --key detection.ad_crest_delta_db --values -99,0.0,1.0,2.0

Each value is scored against a throwaway copy of the config with that one key
replaced, so the real config.yaml is never written to. The copy is a plain YAML
dump and loses the comments; that is fine for something that exists for the
length of one subprocess, and it means the sha256 stamped into each result file
actually distinguishes the runs.

Every run costs a full decode of every session, so a seven-value sweep over
five sessions is hours of audio. Expect it to be slow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"
SCORER = ROOT / "scripts" / "score_detector.py"

# The crest test is the reason this script exists: config.yaml disables it at
# -99 on the strength of one session, and four later sessions disagree.
DEFAULT_KEY = "detection.ad_crest_delta_db"
DEFAULT_VALUES = "-99,0.0,0.5,1.0,1.5,2.0,3.0"


def config_with(base: dict, dotted: str, value: str) -> dict:
    """A copy of *base* with one dotted key replaced, coerced to the old type."""
    section, _, key = dotted.partition(".")
    if section not in base or not isinstance(base[section], dict):
        raise SystemExit(f"--key: no section {section!r} in the config")
    if key not in base[section]:
        raise SystemExit(f"--key: no {key!r} under {section!r}")
    current = base[section][key]
    if isinstance(current, bool):
        coerced: object = value.strip().lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int) and not isinstance(current, bool):
        coerced = int(value)
    elif isinstance(current, float):
        coerced = float(value)
    else:
        coerced = value
    patched = {name: dict(body) if isinstance(body, dict) else body
               for name, body in base.items()}
    patched[section][key] = coerced
    return patched


def score_once(
    config: dict, loso_dir: Path, results_dir: Path, python: str
) -> dict | None:
    """Run the scorer against a temp config; return the result JSON it wrote."""
    before = set(results_dir.glob("*.json")) if results_dir.is_dir() else set()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        proc = subprocess.run(
            [python, str(SCORER), "--loso", str(loso_dir),
             "-c", str(path), "--results-dir", str(results_dir)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return None
    fresh = sorted(set(results_dir.glob("*.json")) - before)
    if not fresh:
        sys.stderr.write("scorer wrote no result file\n")
        return None
    return json.loads(fresh[-1].read_text(encoding="utf-8"))


def render(key: str, runs: list[tuple[str, dict]]) -> None:
    """Per-session false mute per content hour, one row per swept value."""
    if not runs:
        print("no successful runs")
        return
    ids = [s["session_id"] for s in runs[0][1]["sessions"]]
    short = [i.replace("20260", "").replace("-hulu-", "/").replace("-netflix-", "/")
             for i in ids]

    print(f"\n{key} — false mute, seconds per hour of content")
    head = f"{'value':>8}" + "".join(f"{s:>18}" for s in short) + f"{'WORST':>10}{'breaks':>9}"
    print(head)
    print("-" * len(head))
    for value, payload in runs:
        by_id = {s["session_id"]: s for s in payload["sessions"]}
        cells = "".join(
            f"{by_id[i]['false_mute_per_content_hour']:>17.0f}s" for i in ids
        )
        caught = sum(s["breaks_caught"] for s in payload["sessions"])
        total = sum(s["breaks_total"] for s in payload["sessions"])
        worst = payload["worst_false_mute_per_content_hour"]
        print(f"{value:>8}{cells}{worst:>9.0f}s{f'{caught}/{total}':>9}")

    print(f"\n{key} — recall %, per session")
    print(head.replace("WORST", "WORST").rsplit("breaks", 1)[0].rstrip())
    for value, payload in runs:
        by_id = {s["session_id"]: s for s in payload["sessions"]}
        cells = "".join(f"{by_id[i]['recall_pct']:>17.0f}%" for i in ids)
        worst = min(s["recall_pct"] for s in payload["sessions"])
        print(f"{value:>8}{cells}{worst:>9.0f}%")

    best = min(runs, key=lambda r: r[1]["worst_false_mute_per_content_hour"])
    print(f"\nlowest worst-session false mute: {key}={best[0]} at "
          f"{best[1]['worst_false_mute_per_content_hour']:.0f}s per content hour")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loso", type=Path, required=True, metavar="PARENT",
                        help="parent directory of session subdirectories")
    parser.add_argument("--key", default=DEFAULT_KEY, metavar="section.key")
    parser.add_argument("--values", default=DEFAULT_VALUES,
                        help="comma-separated values to sweep")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "sweep",
                        help="where the per-value result JSON goes")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path,
                        help="also write the collected runs here as JSON")
    args = parser.parse_args(argv)

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, dict]] = []
    for n, value in enumerate(values, 1):
        print(f"[{n}/{len(values)}] {args.key}={value} …", flush=True)
        payload = score_once(
            config_with(base, args.key, value),
            args.loso, args.results_dir, args.python,
        )
        if payload is None:
            print(f"  FAILED at {value}", flush=True)
            continue
        payload["swept_key"] = args.key
        payload["swept_value"] = value
        runs.append((value, payload))
        worst = payload["worst_false_mute_per_content_hour"]
        caught = sum(s["breaks_caught"] for s in payload["sessions"])
        total = sum(s["breaks_total"] for s in payload["sessions"])
        print(f"  worst {worst:.0f}s/content-hr, breaks {caught}/{total}", flush=True)

    render(args.key, runs)
    if args.out:
        args.out.write_text(
            json.dumps([p for _, p in runs], indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0 if runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
