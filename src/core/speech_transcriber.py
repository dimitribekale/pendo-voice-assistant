import torch
import soundfile as sf
import resampy
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechTranscriber:
    def __init__(self, model_id="openai/whisper-base"):
        """
        Initializes the transcriber, loading the model and processor,
        and setting the device to GPU if available.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        
        print(f"Successfully initialized Whisper on device: {self.device}")

    def transcribe(self, audio_file):
        """
        Transcribes an audio file of any length.
        """
        # 1. Read and resample the audio file
        audio, original_sr = sf.read(audio_file)
        target_sr = 16000
        if original_sr != target_sr:
            audio = resampy.resample(audio, sr_orig=original_sr, sr_new=target_sr)

        # 2. Handle stereo audio by converting to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 3. Process the audio without truncation
        inputs = self.processor(audio,
                                sampling_rate=target_sr,
                                return_tensors="pt"
                            ).to(self.device) # Move the whole object to the GPU/CPU

        # Unpack the inputs dictionary to pass both input_features and attention_mask
        # Also, explicitly set the language and task
        predicted_ids = self.model.generate(**inputs, language="en", task="transcribe")

        # 5. Decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print(f"ASR result: {transcription}")
        return transcription
