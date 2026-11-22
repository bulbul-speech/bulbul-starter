import io
import numpy as np
import soundfile as sf


def audio_example_to_waveform(audio_ex, target_sr=16000):
    """Decode Hugging Face Audio(decode=False) bytes/path into mono float32 audio."""
    if not isinstance(audio_ex, dict):
        raise TypeError(f"Unexpected audio example type: {type(audio_ex)}")
    # Embedded bytes are preferable: HF path may be a nonlocal archive member.
    if audio_ex.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(audio_ex["bytes"]))
    elif audio_ex.get("path"):
        wav, sr = sf.read(audio_ex["path"])
    else:
        raise ValueError("Audio dict has neither 'path' nor 'bytes'")
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1:
        raise ValueError("Expected mono or stereo audio")
    wav = wav.astype(np.float32)
    if sr != target_sr:
        import torch
        import torchaudio
        wav_t = torch.from_numpy(wav)
        wav = torchaudio.functional.resample(wav_t, sr, target_sr).numpy()
        sr = target_sr
    return wav, sr
