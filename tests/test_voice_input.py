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

    def test_keyboard_interrupt_cancels_recording_not_session(self):
        """Ctrl+C mid-recording must return None (capture cancelled), never
        propagate and kill the session — the user-reported ^C^C^C^C case."""
        import sys
        import types

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                raise KeyboardInterrupt

        fake_sd = types.SimpleNamespace(InputStream=lambda **kw: FakeStream())
        fake_sr = types.SimpleNamespace()
        sys.modules["sounddevice"] = fake_sd
        sys.modules["speech_recognition"] = fake_sr
        try:
            out = voice_input.listen_to_microphone(duration=1)
            self.assertIsNone(out)  # cancelled cleanly, exception did not escape
        finally:
            sys.modules.pop("sounddevice", None)
            sys.modules.pop("speech_recognition", None)

    def test_gemma_engine_transcription(self):
        """The /voice gemma engine posts input_audio to the verified endpoint."""
        import json as j
        import struct
        import tempfile
        import wave as wave_mod
        from unittest.mock import Mock, patch
        path = tempfile.mktemp(suffix=".wav")
        with wave_mod.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<h", 0) * 160)
        try:
            resp = Mock()
            resp.read.return_value = j.dumps(
                {"choices": [{"message": {"content": ' "hello there" '}}]}).encode()
            cm = Mock()
            cm.__enter__ = Mock(return_value=resp)
            cm.__exit__ = Mock(return_value=False)
            with patch("urllib.request.urlopen", return_value=cm) as opener:
                out = voice_input._transcribe_gemma(path, "http://localhost:11434")
            self.assertEqual(out, "hello there")
            req = opener.call_args[0][0]
            self.assertIn("/v1/chat/completions", req.full_url)
            payload = j.loads(req.data)
            self.assertEqual(payload["model"], "gemma4:12b")
            self.assertEqual(payload["messages"][0]["content"][0]["type"], "input_audio")
        finally:
            os.remove(path)

    def test_gemma_engine_failure_returns_none(self):
        """Endpoint failure degrades gracefully (caller falls back to Whisper)."""
        import tempfile
        from unittest.mock import patch
        path = tempfile.mktemp(suffix=".wav")
        open(path, "wb").write(b"RIFF")
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
                self.assertIsNone(voice_input._transcribe_gemma(path, "http://localhost:11434"))
        finally:
            os.remove(path)

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
