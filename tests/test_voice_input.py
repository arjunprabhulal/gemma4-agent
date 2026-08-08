"""Tests for the voice input module."""
import inspect
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
