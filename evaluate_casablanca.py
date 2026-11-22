"""Public screening benchmark. No private Bulbul references are included."""
import argparse
import json
from pathlib import Path
import jiwer
from datasets import load_dataset, Audio
from tqdm import tqdm
from submission import BulbulASR
from audio_utils import audio_example_to_waveform
from normalization import normalize_ar


def score(refs, hyps):
    if not refs:
        return None
    # Jiwer corpus WER/CER: total errors / total reference units.
    return {"wer": round(100 * jiwer.wer(refs, hyps), 4),
            "cer": round(100 * jiwer.cer(refs, hyps), 4), "samples": len(refs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="UBC-NLP/Casablanca")
    p.add_argument("--split", default="test")
    p.add_argument("--model-id", default="openai/whisper-tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--text-column", default="text")
    p.add_argument("--dialect-column", default="dialect")
    p.add_argument("--config", default=None, help="Optional HF dataset config/subset")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default="casablanca_results.json")
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max-samples must be positive")
    kwargs = {"split": args.split}
    if args.config:
        kwargs["name"] = args.config
    ds = load_dataset(args.dataset, **kwargs)
    required = {args.audio_column, args.text_column}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Missing dataset columns {sorted(missing)}. Available: {ds.column_names}. Inspect Casablanca schema and set column flags/config.")
    if args.dialect_column not in ds.column_names:
        raise ValueError(f"Missing dialect column '{args.dialect_column}'. Available: {ds.column_names}. Provide --dialect-column or evaluate a single dialect with --config and an appropriate column.")
    ds = ds.cast_column(args.audio_column, Audio(decode=False))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    model = BulbulASR(args.model_id, args.device)
    per_dialect = {}
    all_refs, all_hyps = [], []
    for row in tqdm(ds, desc="Casablanca"):
        wav, sr = audio_example_to_waveform(row[args.audio_column])
        hyp = model.transcribe(wav, sr)
        if not isinstance(hyp, str):
            raise TypeError("transcribe() must return str")
        ref = normalize_ar(row[args.text_column])
        hyp = normalize_ar(hyp)
        key = str(row[args.dialect_column])
        per_dialect.setdefault(key, {"references": [], "predictions": []})
        per_dialect[key]["references"].append(ref)
        per_dialect[key]["predictions"].append(hyp)
        all_refs.append(ref)
        all_hyps.append(hyp)
    if not all_refs:
        raise ValueError("Dataset split is empty")
    scores = {k: score(v["references"], v["predictions"])
              for k, v in sorted(per_dialect.items())}
    result = {"model_id": args.model_id, "dataset": args.dataset,
              "split": args.split, "config": args.config,
              "partial_evaluation": args.max_samples is not None,
              "overall_micro": score(all_refs, all_hyps),
              "macro_dialect_wer": round(sum(x["wer"] for x in scores.values()) / len(scores), 4),
              "by_dialect": scores,
              "normalization": "Bulbul original normalize_ar"}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
