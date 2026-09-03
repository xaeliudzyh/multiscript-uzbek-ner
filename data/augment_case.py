"""Create lower-case and upper-case variants of entity surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from augmentation_common import JsonObject, read_jsonl, replace_entities, unique_records, write_jsonl, write_report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate case augmentation for entity surfaces.")
    parser.add_argument("--input", type=Path, default=root / "train.jsonl")
    parser.add_argument("--output", type=Path, default=root / "augmentation" / "case.jsonl")
    parser.add_argument(
        "--mode",
        choices=("per-record", "per-entity"),
        default="per-record",
        help="per-record changes all entities at once; per-entity creates the maximum candidate pool.",
    )
    return parser.parse_args()


def has_case(surface: str) -> bool:
    return any(char.islower() or char.isupper() for char in surface)


def generate_per_record(records: list[JsonObject]) -> Iterator[JsonObject]:
    for record in records:
        for target_case, transform in (("lower", str.lower), ("upper", str.upper)):
            replacements = {}
            changed = []
            for index, entity in enumerate(record["entities"]):
                surface = record["text"][entity["start"] : entity["end"]]
                replacement = transform(surface)
                if has_case(surface) and replacement != surface:
                    replacements[index] = replacement
                    changed.append(index)
            if replacements:
                yield replace_entities(
                    record,
                    replacements,
                    kind="case",
                    details={"mode": "per-record", "target_case": target_case, "entity_indices": changed},
                )


def generate_per_entity(records: list[JsonObject]) -> Iterator[JsonObject]:
    for record in records:
        for index, entity in enumerate(record["entities"]):
            surface = record["text"][entity["start"] : entity["end"]]
            if not has_case(surface):
                continue
            for target_case, replacement in (("lower", surface.lower()), ("upper", surface.upper())):
                if replacement == surface:
                    continue
                yield replace_entities(
                    record,
                    {index: replacement},
                    kind="case",
                    details={
                        "mode": "per-entity",
                        "target_case": target_case,
                        "entity_indices": [index],
                        "source_surface": surface,
                        "replacement_surface": replacement,
                    },
                )


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input)
    generated = generate_per_record(records) if args.mode == "per-record" else generate_per_entity(records)
    count = write_jsonl(args.output, unique_records(generated))
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "generated_records": count,
        "mode": args.mode,
        "note": "The pool should be subsampled; keep every original training record.",
    }
    write_report(args.output.with_suffix(".report.json"), report)
    print(f"Generated {count} case-augmented records ({args.mode}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
