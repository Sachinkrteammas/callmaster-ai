"""The transcript must be exactly what Whisper heard: no cleaning, no translation."""
from app.config import settings
from app.stt import Transcriber, format_transcript, make_segment, merge_segments
from tests.conftest import FakeWhisperModel


def test_segment_object_shape():
    seg = make_segment("Agent", 0, 2.5, " Sir good morning sir. ")
    assert seg == {"speaker": "Agent", "start": 0.0, "end": 2.5, "text": "Sir good morning sir."}


def test_text_format():
    segs = [make_segment("Agent", 0, 2.5, "Sir good morning sir."), make_segment("Customer", 2.5, 3, "Hello.")]
    assert format_transcript(segs) == "[0.00s] Agent: Sir good morning sir.\n[2.50s] Customer: Hello."


def test_merge_orders_by_timestamp_across_channels():
    agent = [make_segment("Agent", 0.0, 2, "Sir good morning sir."), make_segment("Agent", 3.2, 4, "Ma'am good morning ma'am.")]
    customer = [make_segment("Customer", 2.1, 3, "Hello."), make_segment("Customer", 5.1, 6, "Haan.")]
    merged = merge_segments([agent, customer])
    assert [s["speaker"] for s in merged] == ["Agent", "Customer", "Agent", "Customer"]
    assert [s["start"] for s in merged] == [0.0, 2.1, 3.2, 5.1]


def test_empty_segments_dropped():
    merged = merge_segments([[make_segment("Agent", 0, 1, "  ")], [make_segment("Customer", 1, 2, "Ok")]])
    assert len(merged) == 1


def _run_fake(text):
    model = FakeWhisperModel({"x.wav": [(0.0, 1.0, text)]})
    segs, _ = Transcriber(model=model).transcribe_channel("/tmp/x.wav", "Agent")
    return segs[0]["text"]


def test_english_preserved():
    assert _run_fake("Sir good morning sir. Hello madam. Thank you.") == "Sir good morning sir. Hello madam. Thank you."


def test_hindi_devanagari_preserved():
    assert _run_fake("हां मैडम। नहीं मैडम। आपका नंबर क्या है?") == "हां मैडम। नहीं मैडम। आपका नंबर क्या है?"


def test_hinglish_preserved():
    assert _run_fake("Madam website pe login nahi ho raha.") == "Madam website pe login nahi ho raha."


def test_names_numbers_repetition_and_fillers_preserved():
    text = "Chetan Arjun Varun Om Shankar 1 2 10 100 number number number haan nahi yes no okay madam sir"
    assert _run_fake(text) == text


def test_real_callmaster_style_transcript_preserved():
    text = "Company madam? चेतन. अर्जुन वरुण Wrong number number number number madam website log नहीं"
    assert _run_fake(text) == text


def test_options_auto_language_and_no_translation(monkeypatch):
    monkeypatch.setattr(settings, "stt_language", None)
    monkeypatch.setattr(settings, "vocab", None)
    opts = Transcriber(model=FakeWhisperModel()).transcription_options()
    assert opts["language"] is None          # auto-detect
    assert opts["task"] == "transcribe"      # never translate
    assert opts["beam_size"] == settings.stt_beam_size
    assert "hotwords" not in opts and "initial_prompt" not in opts


def test_vocab_is_passed_to_whisper(monkeypatch):
    monkeypatch.setattr(settings, "vocab", "Dialdesk, Callmaster, login")
    monkeypatch.setattr(settings, "stt_vocab_mode", "hotwords")
    model = FakeWhisperModel()
    Transcriber(model=model).transcribe_channel("/tmp/a.wav", "Agent")
    assert model.last_kwargs["hotwords"] == "Dialdesk, Callmaster, login"

    monkeypatch.setattr(settings, "stt_vocab_mode", "initial_prompt")
    Transcriber(model=model).transcribe_channel("/tmp/a.wav", "Agent")
    assert model.last_kwargs["initial_prompt"] == "Dialdesk, Callmaster, login"


def test_language_metadata_returned():
    _, meta = Transcriber(model=FakeWhisperModel()).transcribe_channel("/tmp/a.wav", "Agent")
    assert meta == {"language": "hi", "language_probability": 0.87}
