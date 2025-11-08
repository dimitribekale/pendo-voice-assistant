import os
import torch
import soundfile as sf
import resampy
import logging
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)

class SpeechTranscriber:
    def __init__(self, model_id="openai/whisper-base"):
        """
        Initializes the transcriber, loading the model and processor,
        and setting the device to GPU if available.

        Args:
            model_id: Hugging Face model ID (default: openai/whisper-base)

        Raises:
            RuntimeError: If model fails to load
        """
        try:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing Whisper model on device: {self.device}")

            # Load processor
            try:
                self.processor = WhisperProcessor.from_pretrained(model_id)
                logger.info("Whisper processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper processor: {e}")
                raise RuntimeError(f"Could not load Whisper processor: {e}")

            # Load model
            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise RuntimeError(f"Could not load Whisper model: {e}")

            logger.info(f"✅ Successfully initialized Whisper on device: {self.device}")

        except Exception as e:
            logger.exception(f"Critical error initializing SpeechTranscriber: {e}")
            raise

    def transcribe(self, audio_file):
        """
        Transcribes an audio file of any length.

        Args:
            audio_file: Path to the audio file

        Returns:
            str: Transcribed text, or empty string if transcription fails
        """
        try:
            # Validate audio file exists
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return ""

            # Check file size (warn if empty)
            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                logger.warning(f"Audio file is empty: {audio_file}")
                return ""

            logger.info(f"Transcribing audio file: {audio_file} ({file_size} bytes)")

            # 1. Read and resample the audio file
            try:
                audio, original_sr = sf.read(audio_file)
                logger.info(f"Audio loaded: sample rate={original_sr}, shape={audio.shape}")
            except Exception as e:
                logger.error(f"Failed to read audio file: {e}")
                return ""

            target_sr = 16000
            if original_sr != target_sr:
                try:
                    audio = resampy.resample(audio, sr_orig=original_sr, sr_new=target_sr)
                    logger.info(f"Audio resampled from {original_sr}Hz to {target_sr}Hz")
                except Exception as e:
                    logger.error(f"Failed to resample audio: {e}")
                    return ""

            # 2. Handle stereo audio by converting to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                logger.info("Converted stereo audio to mono")

            # 3. Process the audio
            try:
                inputs = self.processor(
                    audio,
                    sampling_rate=target_sr,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                logger.error(f"Failed to process audio: {e}")
                return ""

            # 4. Generate transcription
            try:
                predicted_ids = self.model.generate(
                    **inputs,
                    language="en",
                    task="transcribe"
                )
            except RuntimeError as e:
                logger.error(f"Model inference failed (possible OOM): {e}")
                return ""
            except Exception as e:
                logger.error(f"Failed to generate transcription: {e}")
                return ""

            # 5. Decode token ids to text
            try:
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                logger.info(f"✅ Transcription successful: '{transcription}'")
                return transcription

            except Exception as e:
                logger.error(f"Failed to decode transcription: {e}")
                return ""

        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(f"Unexpected error during transcription: {e}")
            return ""
