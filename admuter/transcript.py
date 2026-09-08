"""Ad detection from what the television is *saying*, not how loud it is.

Every other signal in this system is loudness or dynamics, and they all fail the
same way: on content mastered as hot as its own ads there is nothing left to
measure. On the reality-TV session the ad/content loudness gap is 1.05 dB with
58% overlap, and no threshold over that separates anything.

Ad copy does not have that problem. It is formulaic, and some of it is legally
mandated -- a pharmaceutical spot must say "tell your doctor", an offer must say
"restrictions apply". Those phrases essentially never occur in scripted
dialogue, and crucially they do not care how loud the show is.

WHY THIS ONLY EXTENDS MUTES, NEVER STARTS THEM
----------------------------------------------
Recognition lags. Words have to be spoken, buffered, decoded and scored, so the
evidence lands 10-20s after the break began -- long after the moment a mute
needed to start. It is therefore wired into the *stay* half of the detector's
hysteresis, where a slow, confident voter is exactly what is wanted: it holds a
mute open through the middle of a break the acoustic detector is about to drop.
Using it to enter would mute the show a quarter-minute after an ad ended.

The audio path must never block on this. ``feed`` drops samples when the
recogniser falls behind rather than backing up the capture loop; late words are
worthless and a stalled capture loop is a stopped ad muter.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# Vosk wants 16 kHz mono; the capture chain is 48 kHz stereo. 48/16 is exactly
# 3, so decimation is an integer factor and needs no resampler.
ASR_SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

# Weights say how much a phrase means, not how common it is. Regulatory
# boilerplate scores highest because it is close to exclusive to advertising:
# a drama can say "side effects", but nothing except an ad says "restrictions
# apply" or "well qualified buyers".
AD_COPY_PHRASES: dict[str, float] = {
    # --- legal / regulatory boilerplate: near-exclusive to ads ---
    "restrictions apply": 2.0,
    "terms and conditions": 2.0,
    "not available in all states": 2.5,
    "see dealer for details": 2.5,
    "well qualified buyers": 2.5,
    "results may vary": 2.0,
    "individual results may vary": 2.5,
    "while supplies last": 2.0,
    "for a limited time": 1.5,
    "limited time offer": 2.0,
    "offer ends": 1.5,
    "no purchase necessary": 2.5,
    "void where prohibited": 2.5,
    # --- pharmaceutical: much of this is legally mandated ---
    "tell your doctor": 2.5,
    "ask your doctor": 2.5,
    "call your doctor": 2.0,
    "talk to your doctor": 2.0,
    "side effects": 1.5,
    "serious side effects": 2.0,
    "side effects may include": 2.5,
    "do not take": 1.5,
    "may cause": 1.0,
    "allergic reaction": 1.5,
    "prescription": 1.0,
    "clinical studies": 1.5,
    "consult your doctor": 2.0,
    "if you are pregnant": 2.0,
    # --- insurance ---
    "free quote": 2.0,
    "switch and save": 2.5,
    "auto insurance": 2.0,
    "home insurance": 2.0,
    "life insurance": 2.0,
    "licensed agent": 2.0,
    "your deductible": 1.5,
    "coverage options": 1.5,
    # --- automotive / retail finance ---
    "zero percent": 1.5,
    "percent a p r": 2.0,
    "cash back": 1.5,
    "test drive": 1.5,
    "lease for": 2.0,
    "down payment": 1.5,
    # --- direct response ---
    "call now": 2.0,
    "order now": 2.0,
    "call the number on your screen": 2.5,
    "visit us at": 1.5,
    "dot com": 1.0,
    "brought to you by": 1.5,
    "ask about": 1.0,
    "available now at": 1.5,
    "in stores now": 1.5,
}

_WORD = re.compile(r"[a-z']+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    ASR output has no punctuation and inconsistent spacing, so phrases are
    matched against a flattened word stream rather than raw text.
    """
    return " ".join(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class LexiconHit:
    score: float
    phrases: tuple[str, ...]


class LexiconScorer:
    """Sums the weight of distinct ad-copy phrases found in a transcript.

    ``min_phrases`` guards the obvious false positive: a medical drama really
    does say "tell your doctor". One phrase is a coincidence, and scores zero;
    two or more distinct phrases inside one short window is ad copy.
    """

    def __init__(
        self,
        phrases: dict[str, float] | None = None,
        min_phrases: int = 2,
    ) -> None:
        self.phrases = dict(phrases if phrases is not None else AD_COPY_PHRASES)
        self.min_phrases = int(min_phrases)
        self._normalised = {normalise(p): w for p, w in self.phrases.items()}

    def score(self, text: str) -> LexiconHit:
        flat = normalise(text)
        if not flat:
            return LexiconHit(0.0, ())
        found = tuple(sorted(p for p in self._normalised if p and p in flat))
        if len(found) < self.min_phrases:
            return LexiconHit(0.0, found)
        return LexiconHit(sum(self._normalised[p] for p in found), found)


# --------------------------------------------------------------------------- #
# Recogniser seam
# --------------------------------------------------------------------------- #


class Recognizer(Protocol):
    """Whatever turns 16 kHz mono PCM into words.

    Kept behind a Protocol so the queueing, decimation, decay and scoring can
    all be tested without Vosk installed -- there is no macOS wheel, so the real
    binding is only exercisable on the Pi.
    """

    def accept(self, pcm16: bytes) -> str:
        """Feed audio; return any newly finalised text (may be empty)."""

    def final(self) -> str:
        """Flush and return whatever is still pending."""


class VoskRecognizer:
    """Streaming Vosk. Imported lazily so the package stays optional."""

    def __init__(self, model_path: str, sample_rate: int = ASR_SAMPLE_RATE) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)  # Vosk is extremely chatty on stderr by default
        self._model = Model(model_path)
        self._rec = KaldiRecognizer(self._model, sample_rate)

    def accept(self, pcm16: bytes) -> str:
        import json

        if self._rec.AcceptWaveform(pcm16):
            return json.loads(self._rec.Result()).get("text", "")
        return ""

    def final(self) -> str:
        import json

        return json.loads(self._rec.FinalResult()).get("text", "")


def to_asr_pcm(samples: np.ndarray, sample_rate: int) -> bytes:
    """48 kHz stereo float32 -> 16 kHz mono int16, the only format Vosk takes.

    Decimation is by whole groups averaged together rather than by picking every
    Nth sample: the averaging is a crude box filter that takes the edge off
    aliasing, which matters because speech energy sits right where 48->16 folds.
    """
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    factor = max(1, int(round(sample_rate / ASR_SAMPLE_RATE)))
    if factor > 1:
        usable = (len(audio) // factor) * factor
        if usable == 0:
            return b""
        audio = audio[:usable].reshape(-1, factor).mean(axis=1)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------- #
# Voter
# --------------------------------------------------------------------------- #


@dataclass
class _State:
    score: float = 0.0
    phrases: tuple[str, ...] = ()
    at: float | None = None
    transcript: str = ""
    dropped: int = 0
    finals: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class TranscriptVoter:
    """Background ASR that exposes a decaying "this sounds like ad copy" score.

    Nothing here runs on the capture thread except ``feed``, which is a
    bounded, non-blocking hand-off. If recognition falls behind, audio is
    dropped and counted; the alternative is a queue that grows without limit
    and a mute decision made on words from a minute ago.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        scorer: LexiconScorer | None = None,
        threshold: float = 3.0,
        decay_seconds: float = 30.0,
        queue_size: int = 32,
        synchronous: bool = False,
    ) -> None:
        self.recognizer = recognizer
        self.scorer = scorer or LexiconScorer()
        self.threshold = float(threshold)
        self.decay_seconds = float(decay_seconds)
        # Offline replay runs far faster than realtime, so a background thread
        # would drop almost every window and the run would measure nothing.
        # Synchronous mode recognises inline instead: slow, deterministic, and
        # the only way to score whether the transcript vote actually helps.
        self.synchronous = bool(synchronous)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._state = _State()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="admuter-asr", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- capture side (must never block) -------------------------------- #

    def feed(self, samples: np.ndarray, sample_rate: int, timestamp: float) -> None:
        if self.synchronous:
            self._process((samples, sample_rate, timestamp))
            return
        try:
            self._queue.put_nowait((samples, sample_rate, timestamp))
        except queue.Full:
            with self._state.lock:
                self._state.dropped += 1

    # -- detector side --------------------------------------------------- #

    def says_ad(self, timestamp: float) -> tuple[bool, float]:
        """(above_threshold, decayed_score) for this moment.

        The score decays linearly to zero over ``decay_seconds`` from the last
        match, so evidence from an ad that ended a minute ago cannot keep a mute
        alive. Ad copy arrives in bursts; between spots there is music and no
        words at all, and the decay is what carries the vote across those gaps
        without carrying it into the show.
        """
        with self._state.lock:
            score, at = self._state.score, self._state.at
        if at is None or score <= 0.0:
            return False, 0.0
        age = timestamp - at
        if age < 0 or age >= self.decay_seconds:
            return False, 0.0
        decayed = score * (1.0 - age / self.decay_seconds)
        return decayed >= self.threshold, decayed

    @property
    def stats(self) -> dict[str, float]:
        with self._state.lock:
            return {
                "asr_dropped_windows": float(self._state.dropped),
                "asr_final_results": float(self._state.finals),
            }

    @property
    def last_transcript(self) -> str:
        with self._state.lock:
            return self._state.transcript

    # -- worker ---------------------------------------------------------- #

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            self._process(item)

    def _process(self, item) -> None:
        samples, sample_rate, timestamp = item
        try:
            pcm = to_asr_pcm(samples, sample_rate)
            if not pcm:
                return
            text = self.recognizer.accept(pcm)
        except Exception:  # noqa: BLE001 - a broken recogniser must not take
            # the detector with it; the acoustic path still works without words.
            log.exception("ASR failed; skipping this window")
            return
        if not text:
            return
        hit = self.scorer.score(text)
        with self._state.lock:
            self._state.finals += 1
            self._state.transcript = text
            if hit.score > 0.0:
                self._state.score = hit.score
                self._state.phrases = hit.phrases
                self._state.at = timestamp
        if hit.score > 0.0:
            log.info(
                "ad copy at %.0fs (score %.1f): %s",
                timestamp, hit.score, ", ".join(hit.phrases),
            )


def load_transcript_voter(
    detection, synchronous: bool = False
) -> "TranscriptVoter | None":
    """Build the voter a DetectionConfig asks for, or None. Never raises.

    Same contract as the ML voter: a missing model, a missing package or a
    broken load is logged and dropped. The acoustic detector alone is a working
    ad muter; a stopped service is not.
    """
    path = getattr(detection, "asr_model_path", "")
    if not path:
        return None
    try:
        recognizer = VoskRecognizer(path)
    except Exception as exc:  # noqa: BLE001 - includes ImportError and OSError
        level = log.error if detection.asr_vote_enabled else log.warning
        level(
            "ASR voter unavailable (%s); continuing without transcript voting%s",
            exc,
            " — detection.asr_vote_enabled is set but will have no effect"
            if detection.asr_vote_enabled else "",
        )
        return None
    voter = TranscriptVoter(
        recognizer,
        scorer=LexiconScorer(min_phrases=detection.asr_min_phrases),
        threshold=detection.asr_threshold,
        decay_seconds=detection.asr_decay_seconds,
        synchronous=synchronous,
    )
    if not synchronous:
        voter.start()
    log.info(
        "ASR voter loaded from %s (threshold %.1f, decay %.0fs, %s)",
        path, detection.asr_threshold, detection.asr_decay_seconds,
        "voting on STAY" if detection.asr_vote_enabled
        else "SHADOW MODE — logged, not counted",
    )
    return voter
