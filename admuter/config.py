"""Load and validate config.yaml into typed dataclasses.

Every tunable in the system lives here. Unknown keys are rejected rather than
ignored — a typo in a threshold name should fail loudly at startup, not silently
leave the default in place while you spend an evening wondering why tuning did
nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when the config file is malformed, misspelled, or out of range."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AudioConfig:
    """Capture settings for the USB SPDIF receiver."""

    device: str = "plughw:CARD=Receiver,DEV=0"
    sample_rate: int = 48000
    channels: int = 2
    window_seconds: float = 1.0
    frame_seconds: float = 0.05
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 30.0

    @property
    def frames_per_window(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))

    def validate(self) -> None:
        if not self.device:
            raise ConfigError("audio.device must not be empty")
        if self.sample_rate <= 0:
            raise ConfigError("audio.sample_rate must be positive")
        if self.channels < 1:
            raise ConfigError("audio.channels must be >= 1")
        if self.window_seconds <= 0:
            raise ConfigError("audio.window_seconds must be positive")
        if not 0 < self.frame_seconds <= self.window_seconds:
            raise ConfigError(
                "audio.frame_seconds must be positive and <= audio.window_seconds"
            )
        if self.retry_initial_seconds <= 0:
            raise ConfigError("audio.retry_initial_seconds must be positive")
        if self.retry_max_seconds < self.retry_initial_seconds:
            raise ConfigError(
                "audio.retry_max_seconds must be >= audio.retry_initial_seconds"
            )


@dataclass(frozen=True)
class DetectionConfig:
    """Heuristic thresholds. See the tuning guide in README.md."""

    # Near-silence
    silence_dbfs: float = -60.0

    # Transition cue: a short silent gap followed by an abrupt shift
    require_transition_cue: bool = True
    min_gap_seconds: float = 0.2
    max_gap_seconds: float = 1.5
    cue_grace_seconds: float = 3.0
    loudness_jump_db: float = 3.0
    centroid_shift_ratio: float = 0.25

    # Ad profile, measured against the trailing content baseline
    ad_loudness_delta_db: float = 2.0
    # Hysteresis: once in an ad, loudness only has to clear this lower bar to
    # stay in it. NaN is the "unset" sentinel — __post_init__ replaces it with
    # ad_loudness_delta_db - 2.0 — so the annotation stays a plain float and
    # _coerce keeps working.
    ad_stay_loudness_delta_db: float = math.nan
    ad_crest_delta_db: float = 2.0

    # Absolute loudness profile. NaN (the default) leaves this off and the ad
    # profile stays purely relative to the baseline, as it always was.
    #
    # Set it and the loudness half of the profile switches from "how far above
    # the trailing baseline" to "how loud, full stop". The delta test defeats
    # itself on long breaks: the baseline is an EMA of recent audio, so two
    # minutes into an ad it has drifted up to meet the ad and the delta decays
    # toward zero exactly when the ad is still playing. Absolute loudness does
    # not decay, because it measures a property of the ad itself -- ads are
    # mastered to the loudness ceiling.
    #
    # -23.0 was the best single split on all three annotated sessions
    # independently (netflix movie -23.5, hulu new-girl -23.0, hulu schitts-creek
    # -23.5), which is why a fixed number is defensible here at all. It assumes a
    # stable capture level; S/PDIF PCM out is a fixed digital level, so TV volume
    # does not move it, but a different adapter or receiver would.
    ad_absolute_dbfs: float = math.nan
    # Hysteresis partner for the above, same idea as ad_stay_loudness_delta_db.
    # Unset -> ad_absolute_dbfs - 2.0.
    ad_stay_absolute_dbfs: float = math.nan
    # Windows of trailing rms_dbfs to take a median over before the absolute
    # test. 1 disables smoothing. Per-window loudness overlaps content 54% of
    # the time; a 15-window median cuts that to 17% because ads arrive in
    # contiguous blocks and content spikes do not. Past ~30 the window starts
    # straddling ad/content boundaries and separation gets worse again.
    loudness_median_windows: int = 1

    # Slow-moving content baseline
    baseline_alpha: float = 0.05
    baseline_min_windows: int = 15

    # Phase 2 voter. Off in every sense by default: no path means the detector
    # never even constructs one, and behaviour is bit-for-bit the heuristic.
    #
    # With a path but ml_vote_enabled false the model runs in SHADOW MODE -- its
    # opinion is computed and logged on every window but cannot move the
    # decision. That is the only honest way to find out what it would have done
    # to a live stream, and it is where a new model should live until its
    # disagreement rate has been looked at.
    ml_model_path: str = ""
    ml_vote_enabled: bool = False
    # Deliberately high. The voter is a veto on entry (heuristic AND ml), so a
    # low threshold here mostly costs missed breaks; it is the price of the
    # model being wrong in the confident direction.
    ml_threshold: float = 0.85

    # Transcript voter. Empty path means no ASR thread is started at all.
    # Recognition lags the audio by 10-20s, so this votes only on STAY, never
    # on entry -- see admuter/transcript.py. asr_threshold is a summed phrase
    # weight, not a probability.
    asr_model_path: str = ""
    asr_vote_enabled: bool = False
    asr_threshold: float = 3.0
    asr_decay_seconds: float = 30.0
    asr_min_phrases: int = 2

    # Ad lifetime
    ad_end_windows: int = 2
    min_ad_seconds: float = 5.0
    max_ad_seconds: float = 120.0

    def __post_init__(self) -> None:
        if math.isnan(self.ad_stay_loudness_delta_db):
            # Frozen dataclass: bypass the immutability guard for the derived default.
            object.__setattr__(
                self, "ad_stay_loudness_delta_db", self.ad_loudness_delta_db - 2.0
            )
        if math.isnan(self.ad_stay_absolute_dbfs) and not math.isnan(
            self.ad_absolute_dbfs
        ):
            object.__setattr__(
                self, "ad_stay_absolute_dbfs", self.ad_absolute_dbfs - 2.0
            )

    def validate(self) -> None:
        if self.silence_dbfs >= 0:
            raise ConfigError("detection.silence_dbfs must be negative (dBFS)")
        if self.min_gap_seconds <= 0:
            raise ConfigError("detection.min_gap_seconds must be positive")
        if self.max_gap_seconds < self.min_gap_seconds:
            raise ConfigError(
                "detection.max_gap_seconds must be >= detection.min_gap_seconds"
            )
        if self.cue_grace_seconds <= 0:
            raise ConfigError("detection.cue_grace_seconds must be positive")
        if self.loudness_jump_db < 0:
            raise ConfigError("detection.loudness_jump_db must be >= 0")
        if self.centroid_shift_ratio < 0:
            raise ConfigError("detection.centroid_shift_ratio must be >= 0")
        if self.ad_stay_loudness_delta_db > self.ad_loudness_delta_db:
            raise ConfigError(
                "detection.ad_stay_loudness_delta_db must be <= "
                "detection.ad_loudness_delta_db (it is the lower, 'stay' threshold)"
            )
        if not math.isnan(self.ad_absolute_dbfs):
            if self.ad_absolute_dbfs >= 0:
                raise ConfigError("detection.ad_absolute_dbfs must be negative (dBFS)")
            if self.ad_stay_absolute_dbfs > self.ad_absolute_dbfs:
                raise ConfigError(
                    "detection.ad_stay_absolute_dbfs must be <= "
                    "detection.ad_absolute_dbfs (it is the lower, 'stay' threshold)"
                )
        elif not math.isnan(self.ad_stay_absolute_dbfs):
            raise ConfigError(
                "detection.ad_stay_absolute_dbfs is set but "
                "detection.ad_absolute_dbfs is not; the stay threshold alone does "
                "nothing"
            )
        if self.loudness_median_windows < 1:
            raise ConfigError("detection.loudness_median_windows must be >= 1")
        if self.ml_vote_enabled and not self.ml_model_path:
            raise ConfigError(
                "detection.ml_vote_enabled is true but detection.ml_model_path "
                "is empty; there is no model to vote with"
            )
        if not 0.0 < self.ml_threshold <= 1.0:
            raise ConfigError("detection.ml_threshold must be in (0, 1]")
        if self.asr_vote_enabled and not self.asr_model_path:
            raise ConfigError(
                "detection.asr_vote_enabled is true but detection.asr_model_path "
                "is empty; there is no recogniser to vote with"
            )
        if self.asr_threshold <= 0:
            raise ConfigError("detection.asr_threshold must be positive")
        if self.asr_decay_seconds <= 0:
            raise ConfigError("detection.asr_decay_seconds must be positive")
        if self.asr_min_phrases < 1:
            raise ConfigError("detection.asr_min_phrases must be >= 1")
        if not 0 < self.baseline_alpha <= 1:
            raise ConfigError("detection.baseline_alpha must be in (0, 1]")
        if self.baseline_min_windows < 1:
            raise ConfigError("detection.baseline_min_windows must be >= 1")
        if self.ad_end_windows < 1:
            raise ConfigError("detection.ad_end_windows must be >= 1")
        if self.min_ad_seconds < 0:
            raise ConfigError("detection.min_ad_seconds must be >= 0")
        if self.max_ad_seconds <= self.min_ad_seconds:
            raise ConfigError(
                "detection.max_ad_seconds must be > detection.min_ad_seconds"
            )


@dataclass(frozen=True)
class ControllerConfig:
    """State-machine behaviour on top of the detector's raw events."""

    confirm_windows: int = 2
    max_mute_seconds: float = 130.0
    app_check_seconds: float = 10.0

    def validate(self) -> None:
        if self.confirm_windows < 1:
            raise ConfigError("controller.confirm_windows must be >= 1")
        if self.max_mute_seconds <= 0:
            raise ConfigError("controller.max_mute_seconds must be positive")
        if self.app_check_seconds < 0:
            raise ConfigError("controller.app_check_seconds must be >= 0")


@dataclass(frozen=True)
class RokuConfig:
    """Roku External Control Protocol endpoint."""

    host: str = "192.168.0.12"
    port: int = 8060
    timeout_seconds: float = 2.0
    # Detection only runs while one of the armed apps is on screen. This is a
    # safety gate, not a Netflix preference: unarmed, the detector would mute
    # against music apps, game consoles on HDMI, live TV and screensavers, none
    # of which it was tuned on. Set armed_apps_only false to run everywhere.
    armed_apps_only: bool = True
    armed_app_ids: list[str] = field(default_factory=lambda: ["12", "2285", "13"])
    armed_app_names: list[str] = field(
        default_factory=lambda: ["Netflix", "Hulu", "Prime Video"]
    )
    assume_muted_at_start: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def validate(self) -> None:
        if not self.host:
            raise ConfigError("roku.host must not be empty")
        if not 0 < self.port < 65536:
            raise ConfigError("roku.port must be a valid TCP port")
        if self.timeout_seconds <= 0:
            raise ConfigError("roku.timeout_seconds must be positive")
        if self.armed_apps_only and not (self.armed_app_ids or self.armed_app_names):
            raise ConfigError(
                "roku.armed_apps_only is true but no roku.armed_app_ids or "
                "roku.armed_app_names given, so detection could never arm"
            )


@dataclass(frozen=True)
class LoggingConfig:
    """Console logging plus the optional Phase 2 training-data dump."""

    level: str = "INFO"
    feature_log_enabled: bool = False
    feature_log_path: str = "logs/features.jsonl"
    feature_log_format: str = "jsonl"
    # Log every single detector decision at INFO (one line per window), rather
    # than deduplicating unchanged NO_CHANGE reasons down to DEBUG.
    verbose_decisions: bool = False
    # How often to emit a steady-state INFO line anyway; 0 disables.
    decision_heartbeat_windows: int = 60

    _LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    _FORMATS = ("jsonl", "csv")

    def validate(self) -> None:
        if self.decision_heartbeat_windows < 0:
            raise ConfigError("logging.decision_heartbeat_windows must be >= 0")
        if self.level.upper() not in self._LEVELS:
            raise ConfigError(
                f"logging.level must be one of {self._LEVELS}, got {self.level!r}"
            )
        if self.feature_log_format not in self._FORMATS:
            raise ConfigError(
                f"logging.feature_log_format must be one of {self._FORMATS}, "
                f"got {self.feature_log_format!r}"
            )
        if self.feature_log_enabled and not self.feature_log_path:
            raise ConfigError(
                "logging.feature_log_path must be set when feature logging is enabled"
            )


@dataclass(frozen=True)
class Config:
    """Top-level config object handed to every component."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    roku: RokuConfig = field(default_factory=RokuConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        self.audio.validate()
        self.detection.validate()
        self.controller.validate()
        self.roku.validate()
        self.logging.validate()
        # Cross-section sanity: the controller failsafe must outlive the
        # detector failsafe, otherwise the detector's AD_ENDED never gets a
        # chance to fire first and every ad ends via the emergency unmute.
        if self.controller.max_mute_seconds < self.detection.max_ad_seconds:
            raise ConfigError(
                "controller.max_mute_seconds must be >= detection.max_ad_seconds"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        if not isinstance(data, Mapping):
            raise ConfigError("config root must be a mapping")
        sections = {
            "audio": AudioConfig,
            "detection": DetectionConfig,
            "controller": ControllerConfig,
            "roku": RokuConfig,
            "logging": LoggingConfig,
        }
        unknown = sorted(set(data) - set(sections))
        if unknown:
            raise ConfigError(
                f"unknown config section(s): {unknown}; valid: {sorted(sections)}"
            )
        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            raw = data.get(name) or {}
            if not isinstance(raw, Mapping):
                raise ConfigError(f"config section {name!r} must be a mapping")
            kwargs[name] = _build(section_cls, raw, name)
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        if data is None:
            data = {}
        return cls.from_dict(data)


# --------------------------------------------------------------------------- #
# Generic dataclass builder
# --------------------------------------------------------------------------- #


def _build(cls: type[T], data: Mapping[str, Any], section: str) -> T:
    """Instantiate a config dataclass from a mapping, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls) if not f.name.startswith("_")}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{section}: unknown key(s) {unknown}; valid keys: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(data[name], str(spec.type), f"{section}.{name}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(value: Any, type_name: str, where: str) -> Any:
    """Coerce a YAML scalar to the annotated type, or raise ConfigError.

    Annotations arrive as strings because this module uses postponed evaluation,
    so we match on the source text of the annotation.
    """
    type_name = type_name.replace(" ", "")
    if type_name == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false, got {value!r}")
        return value
    if type_name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if type_name == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if type_name == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if type_name in ("list[str]", "List[str]"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{where} must be a list of strings, got {value!r}")
        return list(value)
    raise ConfigError(f"{where}: unsupported config type {type_name!r}")
