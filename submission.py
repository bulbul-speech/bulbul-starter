"""Participant-owned adapter. Keep the BulbulASR API unchanged."""
import numpy as np
import torch
from transformers import pipeline


class BulbulASR:
    def __init__(self, model_id: str, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.model_id = model_id
        # EXAMPLE: replace this block with your HF model loading code.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=False,
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Accept mono float32 waveform and actual sampling rate; return Arabic text."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Expected nonempty mono 1D audio")
        if sample_rate <= 0:
            raise ValueError("Invalid sample rate")
        # Whisper example: adjust for other architectures.
        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": "arabic", "task": "transcribe"},
        )
        prediction = result["text"]
        if not isinstance(prediction, str):
            raise TypeError("ASR output must be a string")
        return prediction.strip()
