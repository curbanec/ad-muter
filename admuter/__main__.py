"""Entry point: ``python -m admuter``.

Wire-up only — every policy decision lives in config.yaml.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from . import __version__
from .capture import AudioCapture, CaptureError
from .config import Config, ConfigError
from .controller import Controller
from .detector import HeuristicDetector
from .logging_setup import build_feature_logger, setup_logging
from .roku import RokuClient

log = logging.getLogger("admuter")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m admuter",
        description="Mute Netflix ads on a Roku TV by listening to its optical output.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="override logging.level from the config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="detect and log, but never send keypresses to the TV",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="load and validate the config, print it, and exit",
    )
    parser.add_argument("--version", action="version", version=f"admuter {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    setup_logging(args.log_level or config.logging.level)

    if args.print_config:
        print(config)
        return 0

    roku = RokuClient(config.roku, dry_run=args.dry_run)
    info = roku.device_info()
    if info:
        log.info(
            "connected to %s (%s, serial %s)",
            info.get("friendly-device-name") or info.get("user-device-name") or "Roku",
            info.get("model-name", "?"),
            info.get("serial-number", "?"),
        )
    else:
        log.warning(
            "no response from %s yet — continuing anyway (TV may be off)",
            config.roku.base_url,
        )
    if args.dry_run:
        log.warning("dry-run: mute/unmute commands will be logged, not sent")

    capture = AudioCapture(config.audio)
    detector = HeuristicDetector(config.detection, config.audio.window_seconds)
    feature_logger = build_feature_logger(config.logging)
    if feature_logger is not None:
        log.info("feature log -> %s (%s)", feature_logger.path, feature_logger.format)

    controller = Controller(capture, detector, roku, config, feature_logger)

    def handle_signal(signum: int, _frame: object) -> None:
        log.info("received %s", signal.Signals(signum).name)
        controller.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        controller.run()
    except CaptureError as exc:
        log.error("capture unavailable: %s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - handler normally catches this
        log.info("interrupted")
    finally:
        roku.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
