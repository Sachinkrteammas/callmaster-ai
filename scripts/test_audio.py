"""Transcribe ONE real recording directly with Whisper and print the details.

Run on the GPU server:
    source scripts/env.sh app
    python scripts/test_audio.py data/audio/sample.mp3

Note: this loads a second copy of Whisper for the test (~3 GB GPU memory).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, setup_logging  # noqa: E402
from app.pipeline import transcribe_audio  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/test_audio.py <path-to-audio>")
        sys.exit(1)
    setup_logging("test_audio")
    path = sys.argv[1]
    t0 = time.time()
    result = transcribe_audio("manual-test", path)
    info = result["audio"]
    print("=" * 60)
    print(f"File           : {path}")
    print(f"Audio duration : {info['duration_seconds']} s")
    print(f"Channels       : {info['channels']}")
    print(f"Sample rate    : {info['sample_rate']} Hz")
    print(f"STT model      : {result['model']}")
    if info["channels"] >= 2:
        print(f"Channel 1      : {settings.channel_1_speaker}")
        print(f"Channel 2      : {settings.channel_2_speaker}")
    for speaker, meta in result["languages"].items():
        print(f"Language ({speaker}): {meta['language']} (p={meta['language_probability']})")
    print(f"Processing time: {time.time() - t0:.1f} s (includes model load)")
    print("=" * 60)
    print(result["transcript"] or "(empty transcript)")


if __name__ == "__main__":
    main()
