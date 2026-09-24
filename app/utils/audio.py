"""Audio helpers built on FFmpeg / ffprobe.

- probe_audio: duration, channel count, sample rate
- prepare_channels: convert to 16 kHz mono WAV per speaker
  * stereo -> channel 1 and channel 2 become separate files, labelled from config
  * mono   -> one file labelled MONO_SPEAKER ("Speaker")

Speaker diarization for mono audio (e.g. pyannote) can be added later by
replacing prepare_channels for the mono case; nothing else needs to change.
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

from app.config import settings

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


class AudioError(Exception):
    """Raised when FFmpeg cannot read or convert a file."""


def _run(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioError(f"{cmd[0]} is not installed") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-3:]
        raise AudioError(f"{cmd[0]} failed: {' | '.join(tail)}") from exc
    return proc.stdout


def is_supported(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in SUPPORTED_EXTENSIONS


def probe_audio(path: str) -> dict:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,codec_name:format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(out or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioError("No audio stream found in file")
    stream = streams[0]
    duration = data.get("format", {}).get("duration")
    return {
        "duration_seconds": round(float(duration), 2) if duration not in (None, "N/A") else None,
        "channels": int(stream.get("channels") or 1),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "codec": stream.get("codec_name"),
    }


def prepare_channels(path: str, work_dir: str, info: dict) -> List[Tuple[str, str]]:
    """Return [(speaker_label, wav_path), ...] ready for Whisper (16 kHz mono PCM)."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    channels = info.get("channels", 1)

    if channels >= 2:
        if channels > 2:
            log.warning("File has %s channels; only channels 1 and 2 are used", channels)
        parts = []
        for index in (0, 1):
            speaker = settings.speaker_for_channel(index)
            out = work / f"channel{index + 1}.wav"
            _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
                  "-af", f"pan=mono|c0=c{index}", "-ar", "16000", "-acodec", "pcm_s16le", str(out)])
            parts.append((speaker, str(out)))
        return parts

    out = work / "mono.wav"
    _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
          "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out)])
    return [(settings.mono_speaker, str(out))]
