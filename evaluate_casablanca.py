"""Evaluate a BulbulASR adapter on Casablanca's eight public test subsets.

Scores are corpus-level percentages. --max-samples is for smoke tests only.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import jiwer
from datasets import Audio, load_dataset
from tqdm import tqdm

from audio_utils import audio_example_to_waveform
from normalization import normalize_ar
from submission import BulbulASR

DIALECTS = (
    "Algeria", "Egypt", "Jordan", "Mauritania",
    "Morocco", "Palestine", "UAE", "Yemen",
)


def score(references, predictions):
    if not references:
        raise ValueError("Cannot score an empty evaluation set")
    return {
        "wer": round(100 * jiwer.wer(references, predictions), 4),
        "cer": round(100 * jiwer.cer(references, predictions), 4),
        "samples": len(references),
    }


def load_casablanca(dialect, split="test", data_dir=None, dataset="UBC-NLP/Casablanca", dataset_loader=load_dataset):
    """Load one dialect from local Parquet shards or the Hugging Face Hub."""
    if data_dir is not None:
        folder = Path(data_dir).expanduser() / dialect
        files = sorted(folder.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} Parquet shards found in {folder}")
        ds = dataset_loader("parquet", data_files={split: [str(f) for f in files]}, split=split)
    else:
        ds = dataset_loader(dataset, name=dialect, split=split)
    required = {"audio", "transcription"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"{dialect}: missing columns {sorted(missing)}; available: {ds.column_names}")
    return ds.cast_column("audio", Audio(decode=False))


def evaluate(args, model=None, dataset_loader=load_dataset):
    model = model if model is not None else BulbulASR(args.model_id, args.device)
    dialects = [args.dialect] if args.dialect else list(DIALECTS)
    by_dialect = {}
    all_references, all_predictions = [], []

    for dialect in dialects:
        ds = load_casablanca(
            dialect=dialect, split=args.split,
            data_dir=getattr(args, "data_dir", None),
            dataset=args.dataset, dataset_loader=dataset_loader,
        )
        if args.max_samples is not None:
            ds = ds.select(range(min(len(ds), args.max_samples)))
        refs, hyps = [], []
        for row in tqdm(ds, desc=dialect):
            waveform, sample_rate = audio_example_to_waveform(row["audio"], target_sr=16000)
            prediction = model.transcribe(waveform, sample_rate)
            if not isinstance(prediction, str):
                raise TypeError(f"{dialect}: transcribe() must return a string")
            refs.append(normalize_ar(row["transcription"]))
            hyps.append(normalize_ar(prediction))
        by_dialect[dialect] = score(refs, hyps)
        all_references.extend(refs)
        all_predictions.extend(hyps)

    result = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "source": "local" if getattr(args, "data_dir", None) else "huggingface",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()) if getattr(args, "data_dir", None) else None,
        "split": args.split,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dialects_evaluated": dialects,
        "partial_evaluation": args.max_samples is not None or args.dialect is not None,
        "max_samples_per_dialect": args.max_samples,
        "normalization": "Bulbul original normalize_ar (unchanged)",
        "overall_micro": score(all_references, all_predictions),
        "macro_dialect_wer": round(sum(v["wer"] for v in by_dialect.values()) / len(by_dialect), 4),
        "macro_dialect_cer": round(sum(v["cer"] for v in by_dialect.values()) / len(by_dialect), 4),
        "by_dialect": by_dialect,
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="UBC-NLP/Casablanca", help="Hugging Face dataset ID (default source)")
    p.add_argument("--data-dir", default=None, help="Local Casablanca root containing Algeria/, Egypt/, etc. with Parquet shards; overrides --dataset")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--model-id", default="openai/whisper-tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dialect", choices=DIALECTS, help="Run only one dialect (debugging)")
    p.add_argument("--max-samples", type=int, help="Limit samples PER dialect (debugging)")
    p.add_argument("--output", default="casablanca_results.json")
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max-samples must be positive")
    result = evaluate(args)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["partial_evaluation"] or args.split != "test":
        print("NOTE: Not an official full-test Casablanca result.")


if __name__ == "__main__":
    main()
