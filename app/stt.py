"""Speech-to-text with faster-whisper.

The transcript is kept EXACTLY as Whisper produced it: no cleaning, no
translation, no removal of repeated words or fillers like haan / nahi / sir.
"""
import logging
import time
from typing import Iterable, List, Tuple

from app.config import settings

log = logging.getLogger(__name__)


def make_segment(speaker: str, start: float, end: float, text: str) -> dict:
    return {
        "speaker": speaker,
        "start": round(float(start), 2),
        "end": round(float(end), 2),
        "text": (text or "").strip(),
    }


def merge_segments(per_channel: Iterable[List[dict]]) -> List[dict]:
    """Merge segments from each channel into one timeline, ordered by start time."""
    tagged = []
    for channel_index, segments in enumerate(per_channel):
        for seg in segments:
            if seg["text"]:
                tagged.append((seg["start"], channel_index, seg))
    tagged.sort(key=lambda item: (item[0], item[1]))
    return [seg for _, _, seg in tagged]


def format_transcript(segments: List[dict]) -> str:
    """[0.00s] Agent: Sir good morning sir."""
    return "\n".join(f"[{s['start']:.2f}s] {s['speaker']}: {s['text']}" for s in segments)


class Transcriber:
    """Loads Whisper once and keeps it in GPU memory for every call."""

    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    @staticmethod
    def _load():
        from faster_whisper import WhisperModel  # imported lazily so unit tests need no GPU

        log.info("Loading Whisper model=%s device=%s compute_type=%s",
                 settings.stt_model, settings.stt_device, settings.stt_compute_type)
        t0 = time.time()
        model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            download_root=settings.stt_model_dir,
        )
        log.info("Whisper loaded in %.1fs", time.time() - t0)
        return model

    def transcription_options(self) -> dict:
        options = {
            "language": settings.stt_language,  # None = auto-detect (Hindi/English/Hinglish)
            "task": "transcribe",               # never translate
            "beam_size": settings.stt_beam_size,
            "best_of": settings.stt_best_of,
            "vad_filter": settings.stt_vad_filter,
            "condition_on_previous_text": settings.stt_condition_on_previous_text,
        }
        if settings.vocab:
            mode = settings.stt_vocab_mode if settings.stt_vocab_mode in ("hotwords", "initial_prompt") else "hotwords"
            options[mode] = settings.vocab
        return options

    def transcribe_channel(self, wav_path: str, speaker: str) -> Tuple[List[dict], dict]:
        segments, info = self.model.transcribe(wav_path, **self.transcription_options())
        result = [make_segment(speaker, s.start, s.end, s.text) for s in segments]  # consumes generator
        meta = {
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 3),
        }
        return result, meta
