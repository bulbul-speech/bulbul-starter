import argparse
import numpy as np
import soundfile as sf
from submission import BulbulASR
from audio_utils import audio_example_to_waveform


def main():
    parser = argparse.ArgumentParser(description="Smoke test Bulbul submission adapter")
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--audio", help="Optional local WAV/FLAC audio path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.audio:
        audio, sr = audio_example_to_waveform({"path": args.audio})
    else:
        # Tests the interface only; meaningful ASR requires real speech.
        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)
        print("No --audio provided: testing interface on silence only")
    model = BulbulASR(args.model_id, args.device)
    output = model.transcribe(audio, sr)
    assert isinstance(output, str), "transcribe must return a string"
    print("PASS: transcribe returned a string")
    print("Transcription:", repr(output))


if __name__ == "__main__":
    main()
