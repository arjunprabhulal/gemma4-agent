"""
Gemma Agent Voice Input Module.

Provides smart microphone recording with Voice Activity Detection (VAD)
and fully local speech-to-text recognition (Whisper via faster-whisper).
"""

import tempfile
import wave
import os
import time
import numpy as np
from typing import Optional
from gemma_agent import ui


_WHISPER_MODEL = "base.en"
# int8 is efficient on CPU and avoids ctranslate2's float16-conversion warning
_WHISPER_INIT = {"compute_type": "int8"}


def _quiet_speech_logs() -> None:
    """Silence ctranslate2's expected float16->float32 conversion warning so it
    never interleaves with the voice UX."""
    try:
        import logging
        import ctranslate2
        ctranslate2.set_log_level(logging.ERROR)
    except Exception:
        pass


def _energy(data) -> float:
    """Mean absolute amplitude, computed in int32 so int16's -32768 cannot overflow."""
    return float(np.abs(data.astype(np.int32)).mean())


def ensure_voice_model() -> bool:
    """Download/cache the local Whisper model ahead of first use.

    Called when voice mode is enabled so the one-time ~74MB download happens
    up front with a visible message, instead of silently stalling the first
    listen. No-op (fast) when the model is already cached.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        ui.print_error("Local speech engine missing. Run: pip install faster-whisper")
        return False
    _quiet_speech_logs()
    try:
        ui.print_info("🎙️  Preparing local speech model (one-time ~74MB download if not already cached)...")
        WhisperModel(_WHISPER_MODEL, **_WHISPER_INIT)
        ui.print_success("Local speech model ready — transcription runs fully on-device.")
        return True
    except Exception as e:
        ui.print_error(f"Could not prepare the speech model: {e}")
        ui.print_info("Voice input will retry the download on first use.")
        return False


def _transcribe_gemma(wav_path: str, ollama_host: str) -> Optional[str]:
    """Transcribe via Gemma 4's native audio (12B) through Ollama's OpenAI-
    compatible endpoint — the 'every network is Gemma' option. Measured ~6s
    per utterance vs ~0.8s for Whisper on an M4 Pro; chosen explicitly via
    `/voice gemma`, never by default."""
    import base64
    import json
    import urllib.request
    try:
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": "gemma4:12b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": "Transcribe exactly what was said. Output only the transcription, nothing else."},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{ollama_host.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text.strip().strip('"') or None
    except Exception as e:
        ui.print_error(f"Gemma transcription failed ({e}); is gemma4:12b pulled? Falling back to Whisper.")
        return None


def _transcribe(recognizer, audio_data, sr_module) -> Optional[str]:
    """Transcribe on-device with local Whisper (faster-whisper implementation).

    The recording never leaves the machine. First use downloads the base.en
    model (~74MB) from Hugging Face once; every run after that is fully offline.
    """
    _quiet_speech_logs()
    try:
        return recognizer.recognize_faster_whisper(
            audio_data, model=_WHISPER_MODEL, init_options=dict(_WHISPER_INIT)
        )
    except sr_module.UnknownValueError:
        return None  # audio was unintelligible
    except ImportError as e:
        # Name the ACTUAL missing module — a transitive gap (e.g. soundfile)
        # is not fixed by reinstalling faster-whisper.
        missing = getattr(e, "name", None) or "faster-whisper"
        ui.print_error(f"Voice transcription needs the '{missing}' package. Run: pip install {missing}")
        return None
    except Exception as e:
        ui.print_error(f"Local transcription failed: {e}")
        return None


def listen_to_microphone(
    silence_timeout: float = 5.0,
    pause_threshold: float = 1.5,
    max_phrase_limit: float = 30.0,
    sample_rate: int = 16000,
    duration: Optional[float] = None,
    engine: str = "whisper",
    ollama_host: str = "http://localhost:11434",
) -> Optional[str]:
    """
    Smart Voice Activity Detection (VAD):
    1. Waits up to `silence_timeout` seconds for speech to BEGIN (`duration`
       overrides this wait window; it does not cap recording length).
       If nothing is heard, cancels cleanly and returns None.
    2. Once speech starts, records until `pause_threshold` seconds of silence
       or `max_phrase_limit` seconds total.
    3. Transcribes locally via Whisper (faster-whisper); the recording never
       leaves the machine and the temp WAV is always deleted.
    """
    if duration is not None:
        silence_timeout = float(duration)
    wav_path = None
    try:
        import sounddevice as sd
        import speech_recognition as sr

        # If the agent is still speaking its previous answer, wait — otherwise the
        # microphone records our own TTS output and loops it back as a new instruction.
        ui.wait_for_speech_to_finish()

        ui.print_info(f"🎙️  Listening to microphone... Speak anytime ({silence_timeout:.0f}s initial timeout)!")

        chunk_duration = 0.1  # 100ms per audio chunk
        chunk_samples = int(sample_rate * chunk_duration)

        frames = []
        has_started = False
        speech_start_time = None
        last_sound_time = time.time()
        start_time = time.time()

        # Dynamic energy threshold calibration
        calibration_levels = []
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            # Calibrate background noise for 0.3s. Keep the audio — otherwise a
            # user who starts talking immediately loses their first syllable.
            for _ in range(3):
                data, _ = stream.read(chunk_samples)
                frames.append(data.tobytes())
                calibration_levels.append(_energy(data))

            bg_noise = np.mean(calibration_levels) if calibration_levels else 100
            # Cap the threshold: if the user is already speaking during
            # calibration, bg_noise IS their voice — an uncapped 2.5x multiple
            # would make detection impossible for the rest of the session.
            energy_threshold = max(min(bg_noise * 2.5, 3000), 300)

            while True:
                data, overflowed = stream.read(chunk_samples)
                frames.append(data.tobytes())

                energy = _energy(data)
                now = time.time()

                if energy > energy_threshold:
                    if not has_started:
                        has_started = True
                        speech_start_time = now
                        ui.print_info("🎙️  Voice detected! Recording your instruction...")
                    last_sound_time = now

                if not has_started:
                    # Cancel if the wait window passes with no speech detected
                    if now - start_time > silence_timeout:
                        return None
                else:
                    if now - last_sound_time > pause_threshold:
                        ui.print_info("⚡ Silence detected. Processing your voice instruction...")
                        break
                    if (now - speech_start_time) > max_phrase_limit:
                        ui.print_info(f"⚡ Max phrase length ({max_phrase_limit:.0f}s) reached. Processing your voice instruction...")
                        break

        if not frames or not has_started:
            return None

        # Save recorded speech to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        # Transcribe speech — 100% local either way (no audio leaves the machine)
        if engine == "gemma":
            with ui.console.status("[bold cyan]🧠 Gemma transcribing...[/bold cyan]", spinner="dots"):
                text = _transcribe_gemma(wav_path, ollama_host)
            if text:
                return text
            # graceful fallback to Whisper below
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = _transcribe(recognizer, audio_data, sr)
        return text.strip() if text else None

    except KeyboardInterrupt:
        # Ctrl+C during listening/recording cancels THIS capture, not the session
        ui.print_info("Recording cancelled.")
        return None
    except Exception as e:
        ui.print_error(f"Voice recording error: {str(e)}")
        ui.print_info("Make sure microphone permissions are granted to your terminal app.")
        return None
    finally:
        # The voice recording must never outlive the call, no matter which
        # path raised — it is a privacy guarantee, not just tidiness.
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
