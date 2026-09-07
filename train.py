"""CLI version of notebooks/02_train_xlm-roberta.ipynb."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from baseline.common import (
    ENTITY_LABELS,
    TAGS,
    TokenizedNerDataset,
    load_fast_tokenizer,
    read_records,
    validate_window,
)
from baseline.predict import _build_windows, _decode_records, _predict_token_scores

ROOT = Path(__file__).resolve().parent
DEFAULT_AUGMENTATIONS = (
    ROOT / "data/augmentation/hard_negatives.jsonl",
    ROOT / "data/augmentation/morphology.jsonl",
    ROOT / "data/augmentation/case.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-R and select the best model by exact-span micro-F1."
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/dev.jsonl")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/xlm_roberta"
    )
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augmentations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use data/augmentation/*.jsonl (default: enabled).",
    )
    parser.add_argument(
        "--augmentation",
        type=Path,
        action="append",
        default=[],
        help="Additional augmentation JSONL; may be repeated.",
    )
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument(
        "--allow-nonempty-output",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "train-batch-size": args.train_batch_size,
        "eval-batch-size": args.eval_batch_size,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "max-length": args.max_length,
        "max-grad-norm": args.max_grad_norm,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_records", "max_dev_records"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def read_train(
    train_path: Path, augmentations: Iterable[Path], limit: int | None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in (train_path, *augmentations):
        if limit is not None and len(records) >= limit:
            break
        if not path.is_file():
            raise FileNotFoundError(path)
        remaining = None if limit is None else limit - len(records)
        for record in read_records(path, require_entities=True, limit=remaining):
            if record["hash"] in hashes:
                raise ValueError(f"duplicate hash across train files: {record['hash']}")
            hashes.add(record["hash"])
            records.append(record)
    return records


def metric_values(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def exact_span_metrics(
    gold_records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    def keyed(records: list[dict[str, Any]]) -> dict[str, set[tuple[str, int, int]]]:
        return {
            record["hash"]: {
                (entity["label"], entity["start"], entity["end"])
                for entity in record["entities"]
            }
            for record in records
        }

    gold, predicted = keyed(gold_records), keyed(predictions)
    if set(gold) != set(predicted):
        raise ValueError("gold and prediction hash sets differ")
    counts = {label: Counter(tp=0, fp=0, fn=0) for label in ENTITY_LABELS}
    for record_hash, gold_entities in gold.items():
        predicted_entities = predicted[record_hash]
        for label in ENTITY_LABELS:
            gold_label = {item for item in gold_entities if item[0] == label}
            pred_label = {item for item in predicted_entities if item[0] == label}
            counts[label]["tp"] += len(gold_label & pred_label)
            counts[label]["fp"] += len(pred_label - gold_label)
            counts[label]["fn"] += len(gold_label - pred_label)
    by_label = {
        label: metric_values(values["tp"], values["fp"], values["fn"])
        for label, values in counts.items()
    }
    micro = metric_values(
        sum(v["tp"] for v in counts.values()),
        sum(v["fp"] for v in counts.values()),
        sum(v["fn"] for v in counts.values()),
    )
    macro = {
        key: sum(by_label[label][key] for label in ENTITY_LABELS) / len(ENTITY_LABELS)
        for key in ("precision", "recall", "f1")
    }
    return {"matching": "exact label/start/end", "by_label": by_label, "micro": micro, "macro": macro}


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"{'scope':<8} {'precision':>10} {'recall':>10} {'f1':>10}")
    print("-" * 42)
    for label in (*ENTITY_LABELS, "micro"):
        values = metrics["micro"] if label == "micro" else metrics["by_label"][label]
        print(f"{label:<8} {values['precision']:10.4f} {values['recall']:10.4f} {values['f1']:10.4f}")
    values = metrics["macro"]
    print(f"{'macro':<8} {values['precision']:10.4f} {values['recall']:10.4f} {values['f1']:10.4f}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    windows: list[Any],
    batch_size: int,
    device: torch.device,
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    scores = _predict_token_scores(
        model, tokenizer, windows, len(records), batch_size=batch_size, device=device
    )
    return _decode_records(records, scores, id2label)


def save_best(
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    metadata: dict[str, Any],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path:
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_json(model_dir / "baseline_config.json", metadata["model_config"])
    write_json(model_dir / "dev_metrics.json", metrics)
    write_jsonl(model_dir / "dev_predictions.jsonl", predictions)
    torch.save(
        {
            **metadata,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        model_dir / "training_state.pt",
    )
    return model_dir


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_nonempty_output:
        raise ValueError(f"output is not empty: {output_dir}; use another path or --allow-nonempty-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    augmentation_paths = [p.expanduser().resolve() for p in args.augmentation]
    if args.augmentations:
        augmentation_paths = [*DEFAULT_AUGMENTATIONS, *augmentation_paths]
    train_records = read_train(
        args.train.expanduser().resolve(), augmentation_paths, args.max_train_records
    )
    dev_records = read_records(
        args.dev.expanduser().resolve(), require_entities=True, limit=args.max_dev_records
    )
    tokenizer = load_fast_tokenizer(args.model_name)
    validate_window(tokenizer, args.max_length, args.stride)
    train_dataset = TokenizedNerDataset(
        train_records, tokenizer, max_length=args.max_length, stride=args.stride,
        description="Tokenize train",
    )
    dev_windows = _build_windows(
        dev_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )

    id2label = dict(enumerate(TAGS))
    label2id = {tag: index for index, tag in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(TAGS), id2label=id2label, label2id=label2id
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(updates_per_epoch * args.epochs * args.warmup_ratio),
        num_training_steps=updates_per_epoch * args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model_config = {
        "schema_version": 1, "base_model": args.model_name, "tags": list(TAGS),
        "max_length": args.max_length, "stride": args.stride, "seed": args.seed,
    }
    run_config = {
        **vars(args), "train": str(args.train), "dev": str(args.dev),
        "output_dir": str(output_dir), "augmentation": [str(p) for p in augmentation_paths],
        "device": str(device), "amp": use_amp, "train_records": len(train_records),
        "dev_records": len(dev_records), "train_windows": len(train_dataset),
        "dev_windows": len(dev_windows),
    }
    write_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    best_f1, best_dir, global_update = -1.0, None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = labeled_tokens = 0
        progress = tqdm(loader, desc=f"Train epoch {epoch}", unit="batch")
        for batch_index, batch in enumerate(progress, 1):
            batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}
            group_start = ((batch_index - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(loader) - group_start)
            context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                loss = model(**batch).loss
            scaler.scale(loss / group_size).backward()
            token_count = int((batch["labels"] != -100).sum())
            weighted_loss += float(loss.detach()) * token_count
            labeled_tokens += token_count
            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
            running_loss = weighted_loss / max(labeled_tokens, 1)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", avg=f"{running_loss:.4f}")
        train_history.append({"epoch": epoch, "loss": running_loss, "global_update": global_update})

        predictions = predict(
            model, tokenizer, dev_records, dev_windows, args.eval_batch_size, device, id2label
        )
        metrics = exact_span_metrics(dev_records, predictions)
        current_f1 = float(metrics["micro"]["f1"])
        eval_history.append({
            "epoch": epoch, "global_update": global_update, "train_loss": running_loss,
            "micro_f1": current_f1,
            **{f"{label}_f1": metrics["by_label"][label]["f1"] for label in ENTITY_LABELS},
        })
        print(f"\nEpoch {epoch}: train_loss={running_loss:.6f}")
        print_metrics(metrics)
        write_json(output_dir / "train_history.json", train_history)
        write_json(output_dir / "eval_history.json", eval_history)
        write_jsonl(output_dir / "latest_dev_predictions.jsonl", predictions)
        write_json(output_dir / "latest_dev_metrics.json", metrics)
        if current_f1 > best_f1:
            best_f1 = current_f1
            metadata = {
                "epoch": epoch, "global_update": global_update, "micro_f1": best_f1,
                "model_config": model_config, "train_history": train_history,
                "eval_history": eval_history,
            }
            best_dir = save_best(
                output_dir, model, tokenizer, optimizer, scheduler, scaler,
                metadata, predictions, metrics,
            )
            write_json(output_dir / "best_checkpoint.json", {
                "epoch": epoch, "micro_f1": best_f1, "checkpoint": str(best_dir)
            })
            print(f"New best checkpoint: {best_dir} (micro-F1={best_f1:.4f})")
        model.train()

    print(f"Training finished in {(time.time() - started) / 60:.1f} minutes")
    print(f"Best exact-span micro-F1: {best_f1:.4f}; checkpoint: {best_dir}")
    if best_dir is None:
        raise RuntimeError("training finished without a checkpoint")
    return best_dir


def main() -> int:
    try:
        run(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
