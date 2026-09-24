"""Real FFmpeg tests (skipped automatically if FFmpeg is not installed)."""
import shutil
import subprocess

import pytest

from app.config import settings
from app.utils import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make(path, channels, extra=()):
    # channel 1 = 440 Hz tone, channel 2 = silence (so we can tell them apart)
    if channels == 2:
        src = ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=8000",
               "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "2",
               "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]", "-map", "[a]"]
    else:
        src = ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=8000"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *src, *extra, str(path)], check=True)


def _peak(path):
    out = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    line = [l for l in out.splitlines() if "max_volume" in l][0]
    return float(line.split("max_volume:")[1].split("dB")[0])


def test_probe_stereo_wav(tmp_path):
    f = tmp_path / "call.wav"
    _make(f, 2)
    info = audio.probe_audio(str(f))
    assert info["channels"] == 2
    assert info["sample_rate"] == 8000
    assert 1.9 <= info["duration_seconds"] <= 2.1


def test_stereo_split_labels_and_channel_order(tmp_path):
    f = tmp_path / "call.wav"
    _make(f, 2)
    parts = audio.prepare_channels(str(f), str(tmp_path / "work"), audio.probe_audio(str(f)))
    assert [p[0] for p in parts] == ["Agent", "Customer"]
    ch1, ch2 = parts[0][1], parts[1][1]
    assert audio.probe_audio(ch1)["channels"] == 1
    assert audio.probe_audio(ch1)["sample_rate"] == 16000
    assert _peak(ch1) > -20      # tone was on channel 1
    assert _peak(ch2) < -80      # silence was on channel 2


def test_swapped_speaker_config(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "channel_1_speaker", "Customer")
    monkeypatch.setattr(settings, "channel_2_speaker", "Agent")
    f = tmp_path / "call.wav"
    _make(f, 2)
    parts = audio.prepare_channels(str(f), str(tmp_path / "w"), audio.probe_audio(str(f)))
    assert [p[0] for p in parts] == ["Customer", "Agent"]


def test_mono_gets_single_speaker(tmp_path):
    f = tmp_path / "mono.wav"
    _make(f, 1)
    parts = audio.prepare_channels(str(f), str(tmp_path / "w"), audio.probe_audio(str(f)))
    assert parts[0][0] == "Speaker" and len(parts) == 1


@pytest.mark.parametrize("ext,codec", [(".mp3", "libmp3lame"), (".m4a", "aac")])
def test_mp3_and_m4a_supported(tmp_path, ext, codec):
    f = tmp_path / f"call{ext}"
    try:
        _make(f, 2, extra=("-c:a", codec))
    except subprocess.CalledProcessError:
        pytest.skip(f"{codec} encoder not available in this ffmpeg")
    info = audio.probe_audio(str(f))
    assert info["channels"] == 2
    parts = audio.prepare_channels(str(f), str(tmp_path / "w"), info)
    assert len(parts) == 2


def test_supported_extensions():
    assert audio.is_supported("a.wav") and audio.is_supported("a.MP3") and audio.is_supported("a.m4a")
    assert not audio.is_supported("a.txt") and not audio.is_supported("")


def test_corrupt_file_raises_audio_error(tmp_path):
    f = tmp_path / "bad.mp3"
    f.write_bytes(b"not audio")
    with pytest.raises(audio.AudioError):
        audio.probe_audio(str(f))
