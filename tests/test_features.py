"""Feature extraction tests — synthetic buffers only, no hardware."""

from __future__ import annotations

import numpy as np
import pytest

from admuter.features import (
    DBFS_FLOOR,
    amplitude_to_dbfs,
    compute_features,
    crest_factor,
    frame_rms,
    peak_level,
    rms,
    rms_dbfs,
    silence_profile,
    spectral_centroid,
    to_mono_float32,
    zero_crossing_rate,
)

SR = 48000


def sine(freq: float, seconds: float = 1.0, amplitude: float = 0.5, sr: int = SR):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def square(freq: float, seconds: float = 1.0, amplitude: float = 0.5, sr: int = SR):
    return (amplitude * np.sign(sine(freq, seconds, 1.0, sr))).astype(np.float32)


def noise(seconds: float = 1.0, amplitude: float = 0.1, sr: int = SR, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(sr * seconds))).astype(np.float32)


def silence(seconds: float = 1.0, sr: int = SR):
    return np.zeros(int(sr * seconds), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def test_to_mono_averages_channels_and_scales_int16():
    stereo = np.array([[32767, -32768], [16384, 16384]], dtype=np.int16)
    mono = to_mono_float32(stereo)
    assert mono.dtype == np.float32
    assert mono.shape == (2,)
    assert mono[0] == pytest.approx(-0.5 / 32768, abs=1e-4)
    assert mono[1] == pytest.approx(0.5, abs=1e-4)


def test_to_mono_passes_float_through():
    mono = to_mono_float32(np.array([0.25, -0.25], dtype=np.float64))
    assert mono.dtype == np.float32
    assert mono.tolist() == pytest.approx([0.25, -0.25])


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #


def test_rms_of_sine_is_amplitude_over_sqrt2():
    assert rms(sine(1000, amplitude=0.5)) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)


def test_rms_dbfs_of_half_scale_sine():
    # 0.5 amplitude sine -> 0.3536 RMS -> about -9 dBFS
    assert rms_dbfs(sine(1000, amplitude=0.5)) == pytest.approx(-9.03, abs=0.1)


def test_dbfs_floors_instead_of_negative_infinity():
    assert amplitude_to_dbfs(0.0) == DBFS_FLOOR
    assert rms_dbfs(silence()) == DBFS_FLOOR
    assert np.isfinite(rms_dbfs(silence()))


def test_peak_level_tracks_the_largest_absolute_sample():
    buf = np.array([0.1, -0.8, 0.3], dtype=np.float32)
    assert peak_level(buf) == pytest.approx(0.8)


def test_empty_buffer_is_handled():
    empty = np.zeros(0, dtype=np.float32)
    assert rms(empty) == 0.0
    assert peak_level(empty) == 0.0
    assert spectral_centroid(empty, SR) == 0.0
    assert zero_crossing_rate(empty) == 0.0


# --------------------------------------------------------------------------- #
# Crest factor — the ad-compression proxy
# --------------------------------------------------------------------------- #


def test_crest_factor_of_sine_is_3db():
    buf = sine(1000, amplitude=0.5)
    crest = crest_factor(peak_level(buf), rms(buf))
    assert crest == pytest.approx(np.sqrt(2), rel=1e-2)
    assert amplitude_to_dbfs(crest) == pytest.approx(3.01, abs=0.1)


def test_square_wave_is_less_dynamic_than_noise():
    """A square wave is maximally 'compressed'; noise has real peaks."""
    sq = compute_features(square(200), SR)
    ns = compute_features(noise(amplitude=0.1), SR)
    assert sq.crest_db == pytest.approx(0.0, abs=0.2)
    assert ns.crest_db > sq.crest_db + 5.0


def test_crest_factor_of_digital_silence_is_unity():
    assert crest_factor(0.0, 0.0) == 1.0
    assert compute_features(silence(), SR).crest_db == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #


def test_spectral_centroid_lands_on_the_tone():
    # Leakage from the window's sidelobes pulls the centroid a few percent high;
    # what matters for detection is that it tracks the dominant energy.
    assert spectral_centroid(sine(1000), SR) == pytest.approx(1000, rel=0.10)
    assert spectral_centroid(sine(5000), SR) == pytest.approx(5000, rel=0.10)


def test_spectral_centroid_orders_low_below_high():
    assert spectral_centroid(sine(300), SR) < spectral_centroid(sine(4000), SR)


def test_spectral_centroid_of_silence_is_zero():
    assert spectral_centroid(silence(), SR) == 0.0


def test_zero_crossing_rate_rises_with_frequency():
    assert zero_crossing_rate(sine(100)) < zero_crossing_rate(sine(4000))


# --------------------------------------------------------------------------- #
# Silence structure
# --------------------------------------------------------------------------- #


def test_frame_rms_splits_the_buffer():
    buf = np.concatenate([np.full(10, 0.5, dtype=np.float32), np.zeros(10, np.float32)])
    frames = frame_rms(buf, 10)
    assert frames.size == 2
    assert frames[0] == pytest.approx(0.5)
    assert frames[1] == pytest.approx(0.0)


def test_frame_rms_short_buffer_is_one_frame():
    assert frame_rms(np.full(5, 0.5, dtype=np.float32), 10).size == 1


def test_silence_profile_separates_leading_interior_and_trailing():
    mask = np.array([True, True, False, True, False, False, True], dtype=bool)
    ratio, max_run, leading, trailing, interior = silence_profile(mask, 0.1)
    assert ratio == pytest.approx(4 / 7)
    assert max_run == pytest.approx(0.2)
    assert leading == pytest.approx(0.2)
    assert trailing == pytest.approx(0.1)
    assert interior == pytest.approx(0.1)


def test_silence_profile_all_silent_has_no_interior_run():
    ratio, max_run, leading, trailing, interior = silence_profile(
        np.ones(5, dtype=bool), 0.1
    )
    assert ratio == 1.0
    assert max_run == pytest.approx(0.5)
    assert leading == pytest.approx(0.5)
    assert trailing == pytest.approx(0.5)
    assert interior == 0.0


def test_silence_profile_no_silence():
    assert silence_profile(np.zeros(5, dtype=bool), 0.1) == (0.0, 0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# compute_features integration
# --------------------------------------------------------------------------- #


def test_is_silence_flag_follows_the_threshold():
    assert compute_features(silence(), SR, silence_dbfs=-60.0).is_silence
    assert not compute_features(sine(1000), SR, silence_dbfs=-60.0).is_silence
    quiet = sine(1000, amplitude=0.0005)  # about -69 dBFS
    assert compute_features(quiet, SR, silence_dbfs=-60.0).is_silence
    assert not compute_features(quiet, SR, silence_dbfs=-80.0).is_silence


def test_short_gap_inside_a_loud_window_is_visible_at_frame_resolution():
    """A 0.3s gap must be measurable even though the window is 1s and loud."""
    loud = sine(1000, seconds=0.35)
    gap = silence(0.3)
    buf = np.concatenate([loud, gap, loud])
    feats = compute_features(buf, SR, silence_dbfs=-60.0, frame_seconds=0.05)
    assert not feats.is_silence  # window-level RMS is nowhere near silent
    assert feats.interior_silence_seconds == pytest.approx(0.3, abs=0.06)
    assert feats.max_silence_run_seconds == pytest.approx(0.3, abs=0.06)
    assert feats.leading_silence_seconds == 0.0
    assert feats.trailing_silence_seconds == 0.0


def test_edge_silence_is_reported_as_leading_and_trailing():
    buf = np.concatenate([silence(0.2), sine(1000, seconds=0.5), silence(0.3)])
    feats = compute_features(buf, SR, silence_dbfs=-60.0, frame_seconds=0.05)
    assert feats.leading_silence_seconds == pytest.approx(0.2, abs=0.06)
    assert feats.trailing_silence_seconds == pytest.approx(0.3, abs=0.06)
    assert feats.interior_silence_seconds == 0.0


def test_features_round_trip_to_dict():
    row = compute_features(sine(1000), SR).as_dict()
    assert row["rms_dbfs"] == pytest.approx(-9.03, abs=0.1)
    assert set(row) >= {
        "rms_dbfs",
        "peak",
        "peak_dbfs",
        "crest_factor",
        "crest_db",
        "spectral_centroid_hz",
        "is_silence",
    }


def test_stereo_int16_window_is_accepted_end_to_end():
    mono = sine(1000, amplitude=0.5)
    stereo = np.stack([mono, mono], axis=1)
    pcm = (stereo * 32767).astype(np.int16)
    feats = compute_features(pcm, SR)
    assert feats.rms_dbfs == pytest.approx(-9.03, abs=0.1)
    assert feats.duration_seconds == pytest.approx(1.0)
