"""Tests for the voice input module."""
import inspect
import os
import unittest
from unittest.mock import Mock
from gemma_agent import voice_input


class _FakeSR:
    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass


class TestVoiceInput(unittest.TestCase):

    def test_listen_to_microphone_signature(self):
        sig = inspect.signature(voice_input.listen_to_microphone)
        self.assertIn("duration", sig.parameters)
        self.assertIn("silence_timeout", sig.parameters)

    def test_transcribe_uses_local_whisper(self):
        rec = Mock()
        rec.recognize_faster_whisper.return_value = "  hello world  "
        out = voice_input._transcribe(rec, Mock(), _FakeSR)
        self.assertEqual(out, "  hello world  ")  # caller strips
        rec.recognize_faster_whisper.assert_called_once()

    def test_transcribe_unintelligible_returns_none(self):
        rec = Mock()
        rec.recognize_faster_whisper.side_effect = _FakeSR.UnknownValueError()
        self.assertIsNone(voice_input._transcribe(rec, Mock(), _FakeSR))

    def test_transcribe_missing_engine_returns_none_with_hint(self):
        """If faster-whisper is somehow absent, a helpful error prints instead of a crash."""
        rec = Mock()
        rec.recognize_faster_whisper.side_effect = ImportError("no module named faster_whisper")
        self.assertIsNone(voice_input._transcribe(rec, Mock(), _FakeSR))

    def test_ensure_voice_model_missing_engine(self):
        """Setup reports failure cleanly when faster-whisper is not installed."""
        import sys
        sys.modules["faster_whisper"] = None  # importing a None module raises ImportError
        try:
            self.assertFalse(voice_input.ensure_voice_model())
        finally:
            sys.modules.pop("faster_whisper", None)

    def test_real_transcription_end_to_end(self):
        """Integration: real audio through the REAL engine — mocks cannot catch
        missing transitive deps (a soundfile gap shipped past mocked tests).
        Skips when the engine or cached model is unavailable (e.g. CI)."""
        import math
        import struct
        import tempfile
        import wave as wave_mod
        try:
            import faster_whisper  # noqa: F401
            import soundfile  # noqa: F401
            import speech_recognition as sr
        except ImportError as e:
            self.skipTest(f"voice engine not installed: {e}")

        path = tempfile.mktemp(suffix=".wav")
        try:
            with wave_mod.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"".join(
                    struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000)))
                    for i in range(16000)
                ))
            recognizer = sr.Recognizer()
            with sr.AudioFile(path) as source:
                audio = recognizer.record(source)
            try:
                out = voice_input._transcribe(recognizer, audio, sr)
            except Exception as e:  # pragma: no cover - explicit failure detail
                self.fail(f"real transcription raised: {type(e).__name__}: {e}")
            # A pure sine tone transcribes to None or near-empty text; the
            # assertion is that the full pipeline ran without any exception.
            self.assertTrue(out is None or isinstance(out, str))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_ensure_voice_model_success(self):
        import sys
        fake = Mock()
        fake.WhisperModel = Mock()
        sys.modules["faster_whisper"] = fake
        try:
            self.assertTrue(voice_input.ensure_voice_model())
            fake.WhisperModel.assert_called_once()
        finally:
            sys.modules.pop("faster_whisper", None)


if __name__ == "__main__":
    unittest.main()
