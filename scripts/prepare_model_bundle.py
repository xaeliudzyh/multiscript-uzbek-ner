from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


TAGS = (
    "O",
    "B-ORG",
    "I-ORG",
    "B-NAME",
    "I-NAME",
    "B-GEO",
    "I-GEO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a Hugging Face model directory around model.safetensors."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model"),
        help="Directory containing model.safetensors.",
    )
    parser.add_argument(
        "--base-model",
        default="FacebookAI/xlm-roberta-large",
        help="Original Hugging Face base model used for fine-tuning.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load the full local model after preparation to verify tensor shapes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}\n"
            "Put your trained model.safetensors into the model directory first."
        )

    print("Base model:", args.base_model)
    print("Model dir: ", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer is required for offset_mapping.")
    tokenizer.save_pretrained(model_dir)

    config = AutoConfig.from_pretrained(args.base_model)
    id2label = {i: tag for i, tag in enumerate(TAGS)}
    label2id = {tag: i for i, tag in id2label.items()}

    config.num_labels = len(TAGS)
    config.id2label = id2label
    config.label2id = label2id
    config.architectures = ["XLMRobertaForTokenClassification"]
    config.save_pretrained(model_dir)

    baseline_config = {
        "schema_version": 1,
        "base_model": args.base_model,
        "tags": list(TAGS),
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
    }
    (model_dir / "baseline_config.json").write_text(
        json.dumps(baseline_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPrepared model bundle:")
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            print(" -", path.name)

    if args.validate:
        print("\nValidating checkpoint against reconstructed config...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        print("OK")
        print("Model class:", model.__class__.__name__)
        print("num_labels:", model.config.num_labels)
        print("id2label:", model.config.id2label)

    print(
        "\nIMPORTANT: keep all generated tokenizer/config files inside model/. "
        "Runtime must not download anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
