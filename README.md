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

The Casablanca dataset (`UBC-NLP/Casablanca`) contains **eight dialect configurations** with `validation` and `test` splits. Each configuration has `audio` and `transcription` columns (not `text` or a per-row `dialect` column). The script evaluates each dialect separately using your `BulbulASR` adapter, decodes audio using `Audio(decode=False)`, resamples to 16 kHz, and scores with the **original Bulbul normalization, unchanged**.

### Choose your dataset source

The evaluator supports **both** the Hugging Face dataset (default) and an existing local Parquet download. Both use the same model adapter, audio decoder, original Bulbul normalizer, and scoring logic. The local directory must contain dialect subfolders such as `Algeria/test-00000-of-00002.parquet`.

**Option A — Hugging Face (no `--data-dir`):**

```bash
python evaluate_casablanca.py --model-id openai/whisper-tiny --device cuda --split test --output casablanca_results.json
```

**Option B — Local Parquet shards:**

```bash
python evaluate_casablanca.py \
  --data-dir /path/to/Casablanca \
  --model-id openai/whisper-tiny \
  --device cuda \
  --split test \
  --output casablanca_results.json
```

`--data-dir` overrides `--dataset`. If a local split is missing, evaluation fails rather than silently downloading data. The JSON records whether the source was `local` or `huggingface`. To compare scores, use the same dataset revision and split in both modes.

Run the **full public test evaluation**:

```bash
python evaluate_casablanca.py \
  --model-id openai/whisper-tiny \
  --device cuda \
  --split test \
  --output casablanca_results.json
```

For a **quick smoke test**, run a single dialect and only a few samples:

```bash
python evaluate_casablanca.py \
  --model-id openai/whisper-tiny \
  --device cuda \
  --dialect Algeria \
  --max-samples 5 \
  --output casablanca_smoke.json
```

The output contains per-dialect WER/CER, overall corpus-level (micro) WER/CER, and macro-averaged dialect WER/CER, all as percentages. A restricted run is flagged `partial_evaluation: true` and **must not** be reported as a full benchmark score. Authors should disclose whether Casablanca was used in training or tuning. Dataset and model revisions should be recorded for reproducibility; the script does not yet pin them automatically.

The original `normalization.py` intentionally retains some Arabic-block punctuation such as `؟`. Do not modify the normalizer to improve scores.

**Status:** Dataset schema was checked against the public Hugging Face viewer. Synthetic integration test included; running it requires the listed dependencies. Actual full Whisper inference over Casablanca has not been executed in this starter's build environment.

## Security and fairness

Submissions must not access hidden labels, inspect evaluation internals, transmit private audio, or contact external services during inference. Organizers must review adapters, run them in isolated, network-disabled workers, and keep scoring references outside model execution. A shared SLURM job is **not** itself a security sandbox. The public kit contains no private Bulbul data.

## License and publication

Model ownership stays with submitters. Model-specific licenses must permit evaluation. Organizers will seek consent to publish model names, affiliations, and aggregate scores. This starter kit's code license should be selected by repository maintainers before public release.
