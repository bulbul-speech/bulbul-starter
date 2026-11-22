# Bulbul ASR Leaderboard — Model Submission Starter

Official starter interface for submitting **Hugging Face-hosted Arabic ASR models** to the private Bulbul evaluation benchmark.

**Paper:** https://aclanthology.org/2026.emnlp-main.1032/

## What to submit

1. Hugging Face model repository URL **and immutable revision/commit hash**.
2. Public code repository URL containing your adapted `submission.py`.
3. Casablanca WER/CER results and evaluation configuration (dataset revision, split, columns, normalization, dependencies).
4. Author name, affiliation, contact email, and GPU requirements through the official Google Form (link to be announced).

The evaluation dataset and reference transcriptions are private. Model weights must be hosted on Hugging Face. A GitHub repository may be used for adapter code. Submission code is reviewed before execution; submission does not guarantee acceptance.

## Hardware

Evaluation is limited to **one NVIDIA RTX A6000 (48 GB VRAM)**. Models must work on a single GPU. Disclose any required dependencies, CPU offloading, quantization, or special execution requirements. The organizers may decline submissions that exceed resource limits.

## Install

Use Python 3.10 or 3.11. Install PyTorch and matching torchaudio for your CUDA setup, then:

```bash
pip install -r requirements.txt
```

The example uses `openai/whisper-tiny` and Hugging Face Transformers. It is an interface demonstration, not a competitive Arabic baseline.

## Customize your model

Edit **only** `submission.py` for normal submissions. Preserve:

```python
class BulbulASR:
    def __init__(self, model_id: str, device: str = "cuda"): ...
    def transcribe(self, audio, sample_rate: int) -> str: ...
```

`audio` is a **mono float32 NumPy waveform**, resampled to **16,000 Hz** by the evaluator. Return **one unnormalized UTF-8 transcription string**. Do not compute WER or CER in `submission.py`. Do not access reference text. Do not download resources or make network requests during private inference. Package or declare dependencies and use reproducible model revisions. Do not use `trust_remote_code=True` unless explicitly approved.

## Smoke test

```bash
python test_submission.py --model-id openai/whisper-tiny --device cpu --audio path/to/speech.wav
```

Without `--audio`, the script uses two seconds of silence and validates only the interface. Replace the model ID with your Hugging Face model. For GPU tests, use `--device cuda`.

## Casablanca screening evaluation

The public screening evaluator uses `UBC-NLP/Casablanca`, `Audio(decode=False)`, and the **original Bulbul normalization**, unchanged. Casablanca dataset layouts can differ by configuration/revision: **inspect the actual schema first** and provide the correct split, config, audio, text, and dialect column names.

```python
from datasets import load_dataset
print(load_dataset("UBC-NLP/Casablanca"))
```

Example for a dataset layout with `test` split and `audio`, `text`, `dialect` columns:

```bash
python evaluate_casablanca.py \
  --dataset UBC-NLP/Casablanca \
  --split test \
  --audio-column audio --text-column text --dialect-column dialect \
  --model-id openai/whisper-tiny \
  --device cuda \
  --output casablanca_results.json
```

If Casablanca is organized as separate dialect configs or lacks a dialect column, the organizer must finalize the official dataset adapter before accepting scores. **Do not treat the example command as a verified official Casablanca protocol.** The `--max-samples` flag is for debugging only and marks output as partial. Never report partial scores as official.

Results include corpus-level (micro) WER/CER and per-dialect WER/CER. The macro dialect WER is the unweighted mean of per-dialect WERs. Use the **same** `normalization.py` on both references and predictions. The original normalizer intentionally preserves Arabic punctuation characters such as `؟` in some cases; do not alter it.

## Security and fairness

Submissions must not access hidden labels, inspect evaluation internals, transmit private audio, or contact external services during inference. Organizers must review adapters, run them in isolated, network-disabled workers, and keep scoring references outside model execution. A shared SLURM job is **not** itself a security sandbox. The public kit contains no private Bulbul data.

## License and publication

Model ownership stays with submitters. Model-specific licenses must permit evaluation. Organizers will seek consent to publish model names, affiliations, and aggregate scores. This starter kit's code license should be selected by repository maintainers before public release.
