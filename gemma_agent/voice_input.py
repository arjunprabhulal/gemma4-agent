"""
Gemma Agent Voice Input Module.

Provides smart microphone recording with Voice Activity Detection (VAD)
and speech-to-text recognition.
"""

import tempfile
import wave
import os
import time
import numpy as np
from typing import Optional
from gemma_agent import ui


def listen_to_microphone(
    silence_timeout: float = 5.0,
    pause_threshold: float = 1.5,
    max_phrase_limit: float = 30.0,
    sample_rate: int = 16000,
    duration: Optional[float] = None
) -> Optional[str]:
    """
    Smart Voice Activity Detection (VAD):
    1. Waits up to `silence_timeout` (5s) for speech to begin. If silent, cancels cleanly.
    2. Once you start speaking, records continuously as long as you talk.
    3. Automatically stops recording 1.5s after you finish speaking!
    """
    if duration is not None:
        silence_timeout = float(duration)
    try:
        import sounddevice as sd
        import speech_recognition as sr

        # If the agent is still speaking its previous answer, wait — otherwise the
        # microphone records our own TTS output and loops it back as a new instruction.
        ui.wait_for_speech_to_finish()

        ui.print_info("🎙️  Listening to microphone... Speak anytime (5s initial timeout)!")
        
        chunk_duration = 0.1  # 100ms per audio chunk
        chunk_samples = int(sample_rate * chunk_duration)
        
        frames = []
        has_started = False
        speech_start_time = None
        last_sound_time = time.time()
        start_time = time.time()

        # Dynamic energy threshold calibration
        calibration_frames = []
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            # Calibrate background noise for 0.3s
            for _ in range(3):
                data, _ = stream.read(chunk_samples)
                calibration_frames.append(np.abs(data).mean())
            
            bg_noise = np.mean(calibration_frames) if calibration_frames else 100
            energy_threshold = max(bg_noise * 2.5, 300)

            while True:
                data, overflowed = stream.read(chunk_samples)
                frames.append(data.tobytes())
                
                energy = np.abs(data).mean()
                now = time.time()

                if energy > energy_threshold:
                    if not has_started:
                        has_started = True
                        speech_start_time = now
                        ui.print_info("🎙️  Voice detected! Recording your instruction...")
                    last_sound_time = now

                if not has_started:
                    # Cancel if 5 seconds pass with no speech detected
                    if now - start_time > silence_timeout:
                        return None
                else:
                    # Stop recording 1.5s after you finish talking or max limit reached
                    if now - last_sound_time > pause_threshold or (now - speech_start_time) > max_phrase_limit:
                        ui.print_info("⚡ Silence detected. Processing your voice instruction...")
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

        # Transcribe speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                # 100% local offline transcription via PocketSphinx (no audio leaves the machine)
                text = recognizer.recognize_sphinx(audio_data)
                return text.strip()
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                ui.print_error("Local speech engine is not installed. Run: pip install pocketsphinx")
                return None
            except Exception:
                return None
            finally:
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

    except Exception as e:
        ui.print_error(f"Voice recording error: {str(e)}")
        ui.print_info("Make sure microphone permissions are granted to your terminal app.")
        return None
